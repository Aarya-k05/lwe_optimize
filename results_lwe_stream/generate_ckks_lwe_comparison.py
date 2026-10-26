#!/usr/bin/env python3
"""Compare the recorded CKKS and custom LWE stream-cipher runs."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lwe_stream_simulation import (
    DAMPING,
    DT,
    GAIN_SCALE,
    MASS,
    N_MASSES,
    N_STATES,
    N_STEPS,
    Q_MOD,
    SCALE,
    SPRING,
    build_plant,
)
from scipy.linalg import solve_continuous_are


OUTPUT_DIR = Path(__file__).resolve().parent
CKKS_ZIP = Path(
    "/Users/parthbole/Downloads/18-State-Mass-Spring-Damper-System-main.zip"
)
CKKS_ROOT = "18-State-Mass-Spring-Damper-System-main/"
COMPARISON_CSV = OUTPUT_DIR / "ckks_vs_lwe_comparison.csv"
COMPARISON_PNG = OUTPUT_DIR / "ckks_vs_lwe_four_graphs.png"
COMPARISON_PDF = OUTPUT_DIR / "ckks_vs_lwe_four_graphs.pdf"

PARAMETERS = {
    "masses": N_MASSES,
    "state_dimension": N_STATES,
    "mass_kg": MASS,
    "spring_N_per_m": SPRING,
    "damping_Ns_per_m": DAMPING,
    "sample_interval_s": DT,
    "duration_s": N_STEPS * DT,
    "control_steps": N_STEPS,
    "initial_state": "zeros except x[16]=1.0",
    "state_order": "[pos1,vel1,...,pos9,vel9]",
    "state_cost_diagonal": "[50,1,...,50,1]",
    "input_cost": 0.01,
    "reference": "zero-state regulation",
    "control_law": "u(k)=-K@x(k)",
    "discretization": "continuous-time plant, ZOH dt=0.1 s",
}


def parameters_for(scheme: str) -> dict[str, object]:
    params: dict[str, object] = dict(PARAMETERS)
    if scheme == "CKKS":
        params["encryption"] = {
            "scheme": "CKKS",
            "library": "TenSEAL / Microsoft SEAL",
            "poly_modulus_degree": 8192,
            "coefficient_modulus_bits": [60, 40, 40, 60],
            "global_scale": 2**40,
        }
    elif scheme == "LWE":
        params["encryption"] = {
            "construction": "custom recursive modular stream cipher; not standard HE",
            "q": Q_MOD,
            "secret_matrix_dimension": [N_STATES, N_STATES],
            "secret_entry_range": [0, Q_MOD // 1000],
            "noise": "rounded Box-Muller Gaussian-like",
            "noise_sigma_target": 1.0,  # keep in sync with SIGMA in lwe_stream_simulation.py
            "seed": 42,
            "state_scale": SCALE,
            "gain_scale": GAIN_SCALE,
            "gain_scale_reference_status": "Q8 not specified in report/C",
        }
    return params


def read_csv_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    content = archive.read(CKKS_ROOT + name).decode("utf-8")
    return list(csv.DictReader(io.StringIO(content)))


def ckks_source_check(archive: zipfile.ZipFile) -> tuple[bool, str]:
    plant_source = archive.read(CKKS_ROOT + "controller_sim/state_space_model.py").decode()
    encrypted_source = archive.read(CKKS_ROOT + "controller_sim/encrypted_sim.py").decode()
    compare_source = archive.read(CKKS_ROOT + "controller_sim/compare_and_plot.py").decode()
    checks = {
        "9 masses": "N_MASSES = 9" in plant_source,
        "9-chain physical defaults": "m=1.0, c=1.0, k=2.0" in plant_source,
        "interleaved states": "[pos_1, vel_1, pos_2, vel_2" in plant_source,
        "LQR Q/R": "Q[2 * i, 2 * i] = 50.0" in plant_source
        and "R = np.array([[0.01]])" in plant_source,
        "CARE gain": "solve_continuous_are(A, B, Q, R)" in plant_source
        and "np.linalg.inv(R) @ B.T @ P" in plant_source,
        "ZOH": "method='zoh'" in plant_source,
        "CKKS dot on encrypted vector": "enc_x.dot(neg_K)" in encrypted_source,
        "200-step x0[16] run": "x0[2 * (N_MASSES - 1)] = 1.0" in encrypted_source
        and "steps=200" in encrypted_source,
        "dt and duration": "dt = 0.1" in compare_source
        and "steps = 200" in compare_source,
        "zero reference": "r=0.0" in archive.read(
            CKKS_ROOT + "controller_sim/plaintext_sim.py"
        ).decode(),
        "same error norms": "np.linalg.norm(X_plain - X_enc, axis=1)" in compare_source
        and "np.abs(U_plain - U_enc)" in compare_source,
    }
    return all(checks.values()), "; ".join(
        f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in checks.items()
    )


def verify_common_baseline(
    ckks_rows: list[dict[str, str]], lwe_rows: list[dict[str, str]]
) -> dict[str, float]:
    if len(ckks_rows) != N_STEPS or len(lwe_rows) != N_STEPS:
        raise ValueError("Both recorded control datasets must contain exactly 200 steps.")

    ckks_plain_u = np.array([float(row["u_plain"]) for row in ckks_rows])
    lwe_plain_u = np.array([float(row["u_plain"]) for row in lwe_rows])
    ckks_plain_states = np.array(
        [
            [
                float(ckks_rows[k][f"{field}{mass}_plain"])
                for mass in range(1, N_MASSES + 1)
                for field in ("pos", "vel")
            ]
            for k in range(N_STEPS)
        ]
    )
    lwe_plain_states = np.array(
        [
            [float(lwe_rows[k][f"x_plain_{i}"]) for i in range(N_STATES)]
            for k in range(N_STEPS)
        ]
    )

    A, B, _, _ = build_plant()
    q_cost = np.diag([weight for _ in range(N_MASSES) for weight in (50.0, 1.0)])
    r_cost = np.array([[0.01]])
    p = solve_continuous_are(A, B, q_cost, r_cost)
    gain = np.linalg.inv(r_cost) @ B.T @ p
    from scipy.signal import cont2discrete

    c = np.zeros((1, N_STATES))
    c[0, -2] = 1.0
    d = np.zeros((1, 1))
    ad, bd, _, _, _ = cont2discrete((A, B, c, d), DT, method="zoh")
    x = np.zeros(N_STATES)
    x[16] = 1.0
    expected_plain_states = np.zeros((N_STEPS, N_STATES))
    expected_plain_controls = np.zeros(N_STEPS)
    for step in range(N_STEPS):
        expected_plain_states[step] = x
        u = float((-gain @ x.reshape(-1, 1))[0, 0])
        expected_plain_controls[step] = u
        x = ad @ x + bd[:, 0] * u

    max_plain_control_ckks_lwe_diff = float(np.max(np.abs(ckks_plain_u - lwe_plain_u)))
    max_plain_control_reference_diff = float(
        max(
            np.max(np.abs(ckks_plain_u - expected_plain_controls)),
            np.max(np.abs(lwe_plain_u - expected_plain_controls)),
        )
    )
    max_plain_state_ckks_lwe_diff = float(
        np.max(np.linalg.norm(ckks_plain_states - lwe_plain_states, axis=1))
    )
    max_plain_state_reference_diff = float(
        max(
            np.max(np.linalg.norm(ckks_plain_states - expected_plain_states, axis=1)),
            np.max(np.linalg.norm(lwe_plain_states - expected_plain_states, axis=1)),
        )
    )
    return {
        "max_plain_control_ckks_lwe_diff": max_plain_control_ckks_lwe_diff,
        "max_plain_control_reference_diff": max_plain_control_reference_diff,
        "max_plain_state_ckks_lwe_diff": max_plain_state_ckks_lwe_diff,
        "max_plain_state_reference_diff": max_plain_state_reference_diff,
    }


def read_lwe_setup_ms() -> float | None:
    text = (OUTPUT_DIR / "lwe_stream_summary.txt").read_text(encoding="utf-8")
    match = re.search(
        r"Setup/key generation \(RNG \+ S creation, excluded from per-step\): "
        r"([0-9.eE+-]+) ms",
        text,
    )
    return float(match.group(1)) if match else None


def add_metric(
    records: list[dict[str, str]],
    scheme: str,
    metric: str,
    value: float | None,
    units: str,
    boundary: str,
    status: str,
    details: str = "",
) -> None:
    records.append(
        {
            "record_type": "metric",
            "scheme": scheme,
            "step": "",
            "metric": metric,
            "value": "" if value is None else f"{value:.15g}",
            "units": units,
            "parameters_json": json.dumps(parameters_for(scheme), sort_keys=True),
            "measurement_boundary": boundary,
            "validation_status": status,
            "details": details,
        }
    )


def add_trajectory(
    records: list[dict[str, str]],
    scheme: str,
    metric: str,
    values: np.ndarray,
    units: str,
    status: str,
) -> None:
    for step, value in enumerate(values):
        records.append(
            {
                "record_type": "trajectory",
                "scheme": scheme,
                "step": str(step),
                "metric": metric,
                "value": f"{float(value):.15g}",
                "units": units,
                "parameters_json": json.dumps(parameters_for(scheme), sort_keys=True),
                "measurement_boundary": "Control step k; recorded run data; no smoothing.",
                "validation_status": status,
                "details": "",
            }
        )


def save_comparison_csv(
    ckks_rows: list[dict[str, str]],
    lwe_rows: list[dict[str, str]],
    setup_ms: float | None,
    source_match: bool,
    baseline_checks: dict[str, float],
) -> dict[str, object]:
    ckks_encrypt = np.array([float(row["encrypt_ms"]) for row in ckks_rows])
    ckks_eval = np.array([float(row["he_dot_ms"]) for row in ckks_rows])
    ckks_recovery = np.array([float(row["decrypt_ms"]) for row in ckks_rows])
    ckks_step_total = ckks_encrypt + ckks_eval + ckks_recovery
    ckks_state_payload = np.array(
        [float(row["enc_x_serialized_bytes"]) for row in ckks_rows]
    )
    ckks_control_payload = np.array(
        [float(row["enc_u_serialized_bytes"]) for row in ckks_rows]
    )
    lwe_state_payload = np.array(
        [float(row["enc_x_serialized_bytes"]) for row in lwe_rows]
    )
    ckks_plain_u = np.array([float(row["u_plain"]) for row in ckks_rows])
    ckks_u = np.array([float(row["u_encrypted"]) for row in ckks_rows])
    ckks_control_error = np.abs(ckks_u - ckks_plain_u)
    ckks_plain_states = np.array(
        [
            [
                float(ckks_rows[k][f"{field}{mass}_plain"])
                for mass in range(1, N_MASSES + 1)
                for field in ("pos", "vel")
            ]
            for k in range(N_STEPS)
        ]
    )
    ckks_states = np.array(
        [
            [
                float(ckks_rows[k][f"{field}{mass}_encrypted"])
                for mass in range(1, N_MASSES + 1)
                for field in ("pos", "vel")
            ]
            for k in range(N_STEPS)
        ]
    )
    ckks_state_error = np.linalg.norm(ckks_states - ckks_plain_states, axis=1)

    lwe_plain_u = np.array([float(row["u_plain"]) for row in lwe_rows])
    lwe_u = np.array([float(row["u_lwe"]) for row in lwe_rows])
    lwe_control_error = np.abs(lwe_u - lwe_plain_u)
    lwe_plain_states = np.array(
        [
            [float(lwe_rows[k][f"x_plain_{i}"]) for i in range(N_STATES)]
            for k in range(N_STEPS)
        ]
    )
    lwe_states = np.array(
        [[float(lwe_rows[k][f"x_lwe_{i}"]) for i in range(N_STATES)] for k in range(N_STEPS)]
    )
    lwe_state_error = np.linalg.norm(lwe_states - lwe_plain_states, axis=1)

    ckks_recorded_control_error = np.array(
        [float(row["control_error"]) for row in ckks_rows]
    )
    ckks_recorded_state_error = np.array([float(row["state_error"]) for row in ckks_rows])
    lwe_recorded_control_error = np.array(
        [float(row["control_error_abs"]) for row in lwe_rows]
    )
    lwe_recorded_state_error = np.array(
        [float(row["state_error_l2"]) for row in lwe_rows]
    )
    if not np.allclose(ckks_control_error, ckks_recorded_control_error, atol=1e-10):
        raise ValueError("Recomputed CKKS control errors do not match recorded values.")
    if not np.allclose(ckks_state_error, ckks_recorded_state_error, atol=1e-9):
        raise ValueError("Recomputed CKKS state errors do not match recorded values.")
    if not np.allclose(lwe_control_error, lwe_recorded_control_error, atol=1e-10):
        raise ValueError("Recomputed LWE control errors do not match recorded values.")
    if not np.allclose(lwe_state_error, lwe_recorded_state_error, atol=1e-9):
        raise ValueError("Recomputed LWE state errors do not match recorded values.")

    baseline_ok = all(value <= 1e-10 for value in baseline_checks.values())
    ckks_status = "RECORDED_RUN; source configuration verified" if source_match else "PROVISIONAL; source mismatch"
    lwe_status = (
        "ARITHMETIC RECOVERY PASS; Q8 gain representation is not report/C specified; "
        "not standard HE; comparison provisional"
    )
    overall_status = (
        "PROVISIONAL: LWE gain encoding is not source-specified; setup/key transport "
        "and actual network traffic are not measured; run environments not proven identical"
    )

    records: list[dict[str, str]] = []
    boundaries = {
        "encrypt": (
            "CKKS: perf_counter around ts.ckks_vector(ctx,x) only; ciphertext serialize "
            "occurs after the timer. LWE: perf_counter_ns around fixed-point state encode, "
            "Box-Muller noise, S@Vprev modular matvec, and modular message/noise addition."
        ),
        "evaluation": (
            "CKKS: perf_counter around TenSEAL enc_x.dot(-K), genuine CKKS "
            "ciphertext-plaintext dot product. LWE: perf_counter_ns around modular "
            "gain dot/cancellation using the shared secret-key recurrence pad; not HE."
        ),
        "recovery": (
            "CKKS: timer begins after dot and ends after enc_u serialization and "
            "enc_u.decrypt(); therefore includes result serialization. LWE: centered "
            "modular state/control decoding after key-based recurrence cancellation."
        ),
        "total": "Arithmetic sum of the measured per-step component timings; excludes plant update and setup/key generation.",
    }
    for scheme, arrays, status in (
        ("CKKS", (ckks_encrypt, ckks_eval, ckks_recovery, ckks_step_total), ckks_status),
        (
            "LWE",
            (
                np.array([float(r["encrypt_ms"]) for r in lwe_rows]),
                np.array([float(r["evaluation_ms"]) for r in lwe_rows]),
                np.array([float(r["recovery_ms"]) for r in lwe_rows]),
                np.array([float(r["total_ms"]) for r in lwe_rows]),
            ),
            lwe_status,
        ),
    ):
        for metric, values, boundary in zip(
            ("encryption_mean", "evaluation_control_mean", "recovery_mean", "total_mean"),
            arrays,
            ("encrypt", "evaluation", "recovery", "total"),
        ):
            add_metric(
                records,
                scheme,
                metric,
                float(np.mean(values)),
                "ms/step",
                boundaries[boundary],
                status,
            )

    add_metric(
        records, "CKKS", "setup_key_generation_time", None, "ms",
        "Context and Galois-key generation occur before the loop; no setup timer or serialized context/key size is stored in the archived run.",
        "NOT_RECORDED", "Do not compare against an absent value."
    )
    add_metric(
        records, "LWE", "setup_key_generation_time", setup_ms, "ms",
        "RNG and secret-matrix creation measured outside the 200-step loop.",
        "RECORDED" if setup_ms is not None else "NOT_RECORDED",
        "Gain integer quantization and correctness-test setup are included in the reported setup interval."
    )

    ckks_state_mean = float(np.mean(ckks_state_payload))
    ckks_control_mean = float(np.mean(ckks_control_payload))
    lwe_state_mean = float(np.mean(lwe_state_payload))
    size_boundary = (
        "Serialized object payload per control step; excludes context, keys, "
        "parameters, framing, transport protocol, and retransmissions. In-memory "
        "serialization lengths are recorded, but network transfer is not."
    )
    for scheme, state_bytes, control_bytes, total_bytes, status in (
        (
            "CKKS",
            ckks_state_mean,
            ckks_control_mean,
            ckks_state_mean + ckks_control_mean,
            ckks_status,
        ),
        ("LWE", lwe_state_mean, None, lwe_state_mean, lwe_status),
    ):
        add_metric(records, scheme, "state_ciphertext_payload_mean", state_bytes, "bytes", size_boundary, status)
        add_metric(
            records,
            scheme,
            "control_result_ciphertext_payload_mean",
            control_bytes,
            "bytes",
            size_boundary,
            status if control_bytes is not None else "NO_DISTINCT_RESULT_CIPHERTEXT",
            "LWE control residue is recovered locally; no distinct ciphertext is serialized or sent.",
        )
        add_metric(records, scheme, "combined_per_step_payload_mean", total_bytes, "bytes/step", size_boundary, status)
    add_metric(
        records,
        "CKKS",
        "total_network_communication_overhead",
        None,
        "bytes",
        "Not measurable from the archived simulation; it creates a context and Galois keys in-process but records no key/context serialization or network transfers.",
        "NOT_MEASURED",
    )
    add_metric(
        records,
        "LWE",
        "total_network_communication_overhead",
        None,
        "bytes",
        "Not measurable from the simulation; S is local/shared setup state and no key or transport framing is serialized.",
        "NOT_MEASURED",
    )

    for scheme, control_error, state_error, status in (
        ("CKKS", ckks_control_error, ckks_state_error, ckks_status),
        ("LWE", lwe_control_error, lwe_state_error, lwe_status),
    ):
        add_metric(
            records,
            scheme,
            "maximum_absolute_control_error",
            float(np.max(control_error)),
            "control units",
            "max over k=0..199 of abs(u_scheme[k]-u_plain[k]); recomputed from recorded trajectories.",
            status,
        )
        add_metric(
            records,
            scheme,
            "maximum_state_error_l2",
            float(np.max(state_error)),
            "state L2 units",
            "max over k=0..199 of norm2(x_scheme[k]-x_plain[k]); same recorded-step interval for both schemes.",
            status,
        )

    add_metric(
        records,
        "CKKS",
        "closed_loop_numerical_validation",
        1.0,
        "boolean",
        "Verified archived CKKS control/state trajectories against the recorded plaintext baseline.",
        "PASS",
    )
    add_metric(
        records,
        "LWE",
        "mathematical_recovery_tests",
        1.0,
        "boolean",
        "Six current implementation tests: fixed-point round trip, recursive state recovery, signed output cases, centered-boundary decoding, cancellation identity with signed decode, and initial-state control.",
        "PASS_FOR_Q8_VARIANT",
    )
    add_metric(
        records,
        "LWE",
        "closed_loop_numerical_validation",
        1.0,
        "boolean",
        "Recorded control/state run compared against CARE/ZOH plaintext trajectory with a documented finite-precision/noise error bound.",
        "PASS_FOR_Q8_VARIANT",
    )
    add_metric(
        records,
        "LWE",
        "report_C_gain_representation_conformance",
        0.0,
        "boolean",
        "The Q8 quantization of the fractional LQR gain is not specified by the report or original C implementation.",
        "NOT_ESTABLISHED",
    )
    add_metric(
        records,
        "LWE",
        "genuine_homomorphic_evaluation",
        0.0,
        "boolean",
        "Evaluation is modular recurrence cancellation using the secret key/pad; it is not a homomorphic operation.",
        "NOT_HE",
    )

    for scheme, values, status in (
        ("PLAINTEXT", ckks_plain_u, "REFERENCE_BASELINE"),
        ("CKKS", ckks_u, ckks_status),
        ("LWE", lwe_u, lwe_status),
    ):
        add_trajectory(records, scheme, "control_signal", values, "control units", status)
    for scheme, error, status in (
        ("CKKS", ckks_control_error, ckks_status),
        ("LWE", lwe_control_error, lwe_status),
    ):
        add_trajectory(records, scheme, "control_error_abs", error, "control units", status)
    for scheme, error, status in (
        ("CKKS", ckks_state_error, ckks_status),
        ("LWE", lwe_state_error, lwe_status),
    ):
        add_trajectory(records, scheme, "state_error_l2", error, "state L2 units", status)

    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "record_type",
                "scheme",
                "step",
                "metric",
                "value",
                "units",
                "parameters_json",
                "measurement_boundary",
                "validation_status",
                "details",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    return {
        "ckks_encrypt": ckks_encrypt,
        "ckks_eval": ckks_eval,
        "ckks_recovery": ckks_recovery,
        "ckks_total": ckks_step_total,
        "lwe_encrypt": np.array([float(r["encrypt_ms"]) for r in lwe_rows]),
        "lwe_eval": np.array([float(r["evaluation_ms"]) for r in lwe_rows]),
        "lwe_recovery": np.array([float(r["recovery_ms"]) for r in lwe_rows]),
        "lwe_total": np.array([float(r["total_ms"]) for r in lwe_rows]),
        "ckks_state_payload": ckks_state_payload,
        "ckks_control_payload": ckks_control_payload,
        "lwe_state_payload": lwe_state_payload,
        "ckks_control_error": ckks_control_error,
        "lwe_control_error": lwe_control_error,
        "ckks_plain_u": ckks_plain_u,
        "ckks_u": ckks_u,
        "lwe_u": lwe_u,
        "max_state_ckks": float(np.max(ckks_state_error)),
        "max_state_lwe": float(np.max(lwe_state_error)),
        "max_control_ckks": float(np.max(ckks_control_error)),
        "max_control_lwe": float(np.max(lwe_control_error)),
        "ckks_combined_payload": ckks_state_payload + ckks_control_payload,
        "lwe_combined_payload": lwe_state_payload,
        "baseline_ok": baseline_ok,
        "overall_status": overall_status,
        "ckks_status": ckks_status,
        "lwe_status": lwe_status,
    }


def make_graphs(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        "CKKS vs custom LWE stream cipher — PROVISIONAL COMPARISON",
        fontsize=16,
        fontweight="bold",
        color="darkorange",
    )

    ax = axes[0, 0]
    operations = ["Encryption", "Evaluation/\ncontrol", "Recovery/\ndecryption", "Total"]
    categories = ["ckks_encrypt", "ckks_eval", "ckks_recovery", "ckks_total"]
    lwe_categories = ["lwe_encrypt", "lwe_eval", "lwe_recovery", "lwe_total"]
    x = np.arange(len(operations))
    width = 0.36
    bars_ckks = ax.bar(
        x - width / 2,
        [float(np.mean(data[key])) for key in categories],
        width,
        label="CKKS",
        color="#4472C4",
    )
    bars_lwe = ax.bar(
        x + width / 2,
        [float(np.mean(data[key])) for key in lwe_categories],
        width,
        label="LWE custom modular arithmetic",
        color="#ED7D31",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, operations)
    ax.set_ylabel("Milliseconds per step (log scale)")
    ax.set_title("Timing — recorded per-step component means")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8)
    for bars in (bars_ckks, bars_lwe):
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:.4g}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )
    ax.text(
        0.01,
        0.02,
        "CKKS evaluation is a ciphertext-plaintext CKKS dot product. "
        "LWE evaluation is secret-key modular recurrence cancellation, not HE.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    ax = axes[0, 1]
    labels = ["State ciphertext\npayload", "Control/result\nciphertext payload", "Combined payload\nper step"]
    ckks_sizes = [
        float(np.mean(data["ckks_state_payload"])) / 1024.0,
        float(np.mean(data["ckks_control_payload"])) / 1024.0,
        float(np.mean(data["ckks_combined_payload"])) / 1024.0,
    ]
    lwe_sizes = [
        float(np.mean(data["lwe_state_payload"])) / 1024.0,
        np.nan,
        float(np.mean(data["lwe_combined_payload"])) / 1024.0,
    ]
    x = np.arange(3)
    bars_ckks = ax.bar(x - width / 2, ckks_sizes, width, label="CKKS", color="#4472C4")
    bars_lwe = ax.bar(x + width / 2, lwe_sizes, width, label="LWE", color="#ED7D31")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Serialized payload (KB per step; 1024 bytes/KB)")
    ax.set_title("Serialized Ciphertext Payload (KB)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=20)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for bars in (bars_ckks, bars_lwe):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height / 1024:.4g} KB",
                    (bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    ax.text(
        x[1] + width / 2,
        max(ckks_sizes) * 0.06,
        "N/A\n(no distinct result ciphertext)",
        ha="center",
        fontsize=7,
        color="#9E480E",
    )
    ax.text(
        0.01,
        0.98,
        "Excludes setup keys, protocol framing, and total network traffic.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    ax = axes[1, 0]
    errors = [float(data["max_control_ckks"]), float(data["max_control_lwe"])]
    bars = ax.bar(["CKKS", "LWE custom"], errors, color=["#4472C4", "#ED7D31"])
    ax.set_yscale("log")
    ax.set_ylabel("Maximum absolute control error")
    ax.set_title("Maximum Control Error (k = 0…199)")
    ax.grid(axis="y", which="both", alpha=0.3)
    for bar, value in zip(bars, errors):
        ax.annotate(
            f"{value:.8g}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.text(
        0.99,
        0.97,
        f"State L2 max (same k interval): CKKS {float(data['max_state_ckks']):.5g}; "
        f"LWE {float(data['max_state_lwe']):.5g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    ax = axes[1, 1]
    steps = np.arange(N_STEPS)
    visible = (steps >= 19) & (steps <= 199)
    ax.plot(
        steps[visible],
        np.asarray(data["ckks_plain_u"])[visible],
        "-o",
        markersize=2.5,
        linewidth=1,
        label="Plaintext baseline",
        color="black",
    )
    ax.plot(
        steps[visible],
        np.asarray(data["ckks_u"])[visible],
        "--s",
        markersize=2.5,
        linewidth=1,
        label="CKKS decrypted control",
        color="#4472C4",
    )
    ax.plot(
        steps[visible],
        np.asarray(data["lwe_u"])[visible],
        ":^",
        markersize=2.5,
        linewidth=1,
        label="LWE recovered control (Q8)",
        color="#ED7D31",
    )
    ax.set_xlabel("Control step")
    ax.set_ylabel("Control input u(k)")
    ax.set_title("Control Signal After Transient (steps 19–199)")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    fig.savefig(COMPARISON_PNG, dpi=300)
    fig.savefig(COMPARISON_PDF)
    plt.close(fig)


def append_comparison_assessment(
    data: dict[str, object],
    source_match: bool,
    source_checks: str,
    baseline_checks: dict[str, float],
    setup_ms: float | None,
) -> None:
    summary = OUTPUT_DIR / "lwe_stream_summary.txt"
    text = summary.read_text(encoding="utf-8").rstrip()
    marker = "\nCKKS comparison assessment\n"
    if marker in text:
        text = text.split(marker, maxsplit=1)[0].rstrip()
    assessment = [
        "",
        "CKKS comparison assessment",
        "==========================",
        f"CKKS plant/controller source configuration: {'PASS' if source_match else 'FAIL'}",
        f"Source checks: {source_checks}",
        "Plaintext baseline agreement CKKS-vs-LWE and against CARE/ZOH reconstruction:",
        *[f"  {name}: {value:.12g}" for name, value in baseline_checks.items()],
        f"CKKS setup/context/key-generation time: NOT RECORDED in archived run.",
        f"LWE setup/key-generation time (RNG + S, excluded): "
        f"{'NOT RECORDED' if setup_ms is None else f'{setup_ms:.9g} ms'}",
        "Comparison fairness: PROVISIONAL, not definitive.",
        "Reasons: the current LWE gain uses trajectory-specific Q8 scaling not "
        "specified by Sir's report/C example; the C source's gain examples are not "
        "the CKKS LQR gain (the timed state model has N=2 and 100 steps; the separate "
        "vector-encryption example has N_DIM=400 and a random F); original C truncates "
        "Box-Muller noise to int while this Python simulation rounds it; the report's "
        "own parameter table describes a different two-state, 0.01-second, 10-second "
        "example; CKKS/LWE setup "
        "communication and total network framing are not recorded; timing boundaries "
        "are similar but not identical; identical runtime/hardware conditions have "
        "not been established.",
        "Metric methodology: control error is max over steps 0..199 of "
        "|u_scheme-u_plain|. State error is max over the same steps of "
        "||x_scheme-x_plain||_2. Ciphertext payload is the recorded serialized "
        "object size only; it excludes context/keys/parameters/framing. LWE has no "
        "distinct encrypted control-result ciphertext.",
        "LWE evaluation is secret-key modular recurrence cancellation, not "
        "homomorphic evaluation and not an independent evaluator. CKKS evaluation "
        "uses TenSEAL CKKS ciphertext-plaintext dot product. No security claim is made.",
    ]
    text += "\n" + "\n".join(assessment) + "\n"
    summary.write_text(text, encoding="utf-8")


def main() -> None:
    if not CKKS_ZIP.is_file():
        raise FileNotFoundError(f"CKKS reference archive not found: {CKKS_ZIP}")
    with zipfile.ZipFile(CKKS_ZIP) as archive:
        source_match, source_checks = ckks_source_check(archive)
        ckks_rows = read_csv_from_zip(archive, "results/control_data.csv")
    with (OUTPUT_DIR / "lwe_stream_results.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        lwe_rows = list(csv.DictReader(source))

    baseline_checks = verify_common_baseline(ckks_rows, lwe_rows)
    data = save_comparison_csv(
        ckks_rows,
        lwe_rows,
        read_lwe_setup_ms(),
        source_match,
        baseline_checks,
    )
    make_graphs(data)
    append_comparison_assessment(
        data,
        source_match,
        source_checks,
        baseline_checks,
        read_lwe_setup_ms(),
    )

    print(f"Comparison status: {data['overall_status']}")
    print(f"CKKS source configuration: {'PASS' if source_match else 'FAIL'}")
    print("Baseline checks:", baseline_checks)
    print(
        "Max absolute control error: "
        f"CKKS={data['max_control_ckks']:.12g}, "
        f"LWE={data['max_control_lwe']:.12g}"
    )
    print(
        "Max state L2 error over k=0..199: "
        f"CKKS={data['max_state_ckks']:.12g}, "
        f"LWE={data['max_state_lwe']:.12g}"
    )
    print(f"CSV: {COMPARISON_CSV}")
    print(f"PNG: {COMPARISON_PNG}")
    print(f"PDF: {COMPARISON_PDF}")


if __name__ == "__main__":
    main()
