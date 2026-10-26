#!/usr/bin/env python3
"""Fresh simulation of the report's recursive LWE stream-cipher equations."""

from __future__ import annotations

import csv
import math
import struct
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.linalg import solve_continuous_are


OUTPUT_DIR = Path(__file__).resolve().parent
N_MASSES = 9
N_STATES = 2 * N_MASSES
N_STEPS = 200
DT = 0.1
MASS = 1.0
SPRING = 2.0
DAMPING = 1.0
Q_MOD = 2**31 - 1
SCALE = 2**16
# The report's C code stores feedback coefficients as integers while only the
# state message carries the 2^16 fixed-point scale. Q8 preserves fractional
# LQR-gain precision without making the state-times-gain result ambiguous mod q.
GAIN_SCALE = 2**8
CONTROL_SCALE = SCALE * GAIN_SCALE
SEED = 42
SIGMA = math.sqrt(N_STATES)


def build_plant() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the wall-anchored chain with interleaved position/velocity states."""
    A = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    B = np.zeros((N_STATES, 1), dtype=np.float64)
    C = np.zeros((1, N_STATES), dtype=np.float64)
    D = np.zeros((1, 1), dtype=np.float64)

    for mass_index in range(N_MASSES):
        position = 2 * mass_index
        velocity = position + 1
        A[position, velocity] = 1.0
        stiffness_total = SPRING
        damping_total = DAMPING

        if mass_index > 0:
            A[velocity, position - 2] += SPRING / MASS
            A[velocity, velocity - 2] += DAMPING / MASS
        if mass_index < N_MASSES - 1:
            stiffness_total += SPRING
            damping_total += DAMPING
            A[velocity, position + 2] += SPRING / MASS
            A[velocity, velocity + 2] += DAMPING / MASS

        A[velocity, position] -= stiffness_total / MASS
        A[velocity, velocity] -= damping_total / MASS

    B[-1, 0] = 1.0 / MASS
    C[0, -2] = 1.0
    return A, B, C, D


def centered(value: int) -> int:
    """Decode a canonical residue using the centered interval around zero."""
    residue = int(value) % Q_MOD
    return residue if residue <= Q_MOD // 2 else residue - Q_MOD


def recover_signed_control(residue: int) -> int:
    """Recover the unique signed representative in the centered interval."""
    return centered(residue)


def encode_signed_control(value: float) -> int:
    """Encode a physical control at CONTROL_SCALE if centered recovery is unique."""
    encoded = int(np.rint(value * CONTROL_SCALE))
    if abs(encoded) > Q_MOD // 2:
        raise ValueError(
            f"Control {value} exceeds the centered representable range "
            f"[-{(Q_MOD // 2) / CONTROL_SCALE}, {(Q_MOD // 2) / CONTROL_SCALE}]"
        )
    return encoded


def modular_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Use Python integer accumulators so matrix products cannot overflow."""
    result = np.empty(matrix.shape[0], dtype=np.int64)
    for row_index, row in enumerate(matrix):
        accumulator = sum(int(a) * int(b) for a, b in zip(row, vector))
        result[row_index] = accumulator % Q_MOD
    return result


def modular_dot(vector: np.ndarray, values: np.ndarray) -> int:
    """Compute a modular dot product with arbitrary-precision accumulation."""
    return sum(int(a) * int(b) for a, b in zip(vector, values)) % Q_MOD


def reported_control_residue(
    gain_int: np.ndarray, ciphertext: np.ndarray, previous_pad: np.ndarray
) -> int:
    """Apply the report's modular F*V_i - (F*S)*V_(i-1) cancellation."""
    return (
        modular_dot(gain_int, ciphertext) - modular_dot(gain_int, previous_pad)
    ) % Q_MOD


def box_muller_integer_noise(rng: np.random.Generator) -> np.ndarray:
    """Sample rounded, integer-valued Gaussian-like noise via Box–Muller."""
    uniforms = np.maximum(rng.random((N_STATES, 2)), np.finfo(np.float64).tiny)
    z = np.sqrt(-2.0 * np.log(uniforms[:, 0])) * np.cos(
        2.0 * np.pi * uniforms[:, 1]
    )
    return np.rint(SIGMA * z).astype(np.int64)


def run_correctness_tests(
    secret: np.ndarray, gain_int: np.ndarray, K: np.ndarray
) -> dict[str, bool]:
    """Check fixed-point, recurrence, signed decoding, and initial control recovery."""
    test_scale = 65536
    test_state = np.zeros(N_STATES, dtype=np.float64)
    test_state[16] = 1.0
    encoded_state = np.rint(test_state * test_scale).astype(np.int64)
    fixed_point_passed = bool(
        np.array_equal(encoded_state.astype(np.float64) / test_scale, test_state)
    )

    test_message = np.arange(-9, 9, dtype=np.int64) * (test_scale // 100)
    test_noise = np.arange(N_STATES, dtype=np.int64) - 9
    previous = np.arange(N_STATES, dtype=np.int64) * 7919 % Q_MOD
    pad = modular_matvec(secret, previous)
    test_ciphertext = (pad + test_message + test_noise) % Q_MOD
    recovered_state = np.array(
        [centered(int(c) - int(p)) for c, p in zip(test_ciphertext, pad)],
        dtype=np.int64,
    )
    recurrence_passed = bool(
        np.array_equal(recovered_state, test_message + test_noise)
    )

    signed_cases = [12.5, -0.125, 0.0, 50.0, -50.0]
    signed_cases_passed = True
    control_test_pad = modular_matvec(secret, previous)
    for value in signed_cases:
        encoded = encode_signed_control(value)
        state = np.zeros(N_STATES, dtype=np.float64)
        if value:
            state[16] = value / float(-K[0, 16])
        message = np.rint(state * test_scale).astype(np.int64)
        ciphertext = (control_test_pad + message) % Q_MOD
        residue = reported_control_residue(gain_int, ciphertext, control_test_pad)
        unwrapped = sum(int(a) * int(b) for a, b in zip(gain_int, message))
        recovered = recover_signed_control(residue)
        expected_physical = float((-K @ state.reshape(-1, 1))[0, 0])
        observed_physical = recovered / CONTROL_SCALE
        tolerance = (
            0.5 * abs(state[16]) / GAIN_SCALE
            + 0.5 * np.sum(np.abs(K.reshape(-1))) / test_scale
        )
        signed_cases_passed &= (
            abs(unwrapped) <= Q_MOD // 2
            and recovered == unwrapped
            and abs(observed_physical - expected_physical) <= tolerance + 1e-12
            and abs(observed_physical - value) <= tolerance + 1e-12
        )

    center_limit = Q_MOD // 2
    wraparound_passed = (
        recover_signed_control(center_limit % Q_MOD) == center_limit
        and recover_signed_control((-center_limit) % Q_MOD) == -center_limit
        and centered(center_limit + 1) == -center_limit
        and centered(-center_limit - 1) == center_limit
        and abs(center_limit + 1) > center_limit
        and abs(-center_limit - 1) > center_limit
    )

    initial_message = np.zeros(N_STATES, dtype=np.int64)
    initial_message[16] = test_scale
    initial_control_integer = sum(
        int(a) * int(b) for a, b in zip(gain_int, initial_message)
    )
    initial_control_residue = initial_control_integer % Q_MOD
    initial_control = recover_signed_control(initial_control_residue) / CONTROL_SCALE
    expected_initial_control = float((-K @ test_state.reshape(-1, 1))[0, 0])
    initial_tolerance = (
        0.5 * np.sum(np.abs(test_state)) / GAIN_SCALE
        + 0.5 * np.sum(np.abs(K.reshape(-1))) / test_scale
    )
    initial_control_passed = (
        abs(initial_control_integer) <= center_limit
        and abs(initial_control - expected_initial_control)
        <= initial_tolerance + 1e-12
    )

    # Test the report's actual cancellation identity and physical-unit result.
    report_residue = reported_control_residue(gain_int, test_ciphertext, pad)
    expected_control_integer = sum(
        int(a) * int(b) for a, b in zip(gain_int, test_message + test_noise)
    )
    report_equation_passed = (
        report_residue == expected_control_integer % Q_MOD
        and abs(expected_control_integer) <= center_limit
        and recover_signed_control(report_residue) == expected_control_integer
    )
    return {
        "fixed_point": fixed_point_passed,
        "recursive_state": recurrence_passed,
        "signed_control": bool(signed_cases_passed),
        "modular_wraparound": bool(wraparound_passed),
        "report_equation": bool(report_equation_passed),
        "initial_control": bool(initial_control_passed),
    }


def serialize_ciphertext(ciphertext: np.ndarray) -> bytes:
    """Serialize each canonical q-residue as one big-endian unsigned 32-bit word."""
    return struct.pack(f">{len(ciphertext)}I", *(int(value) for value in ciphertext))


def write_csv(rows: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / "lwe_stream_results.csv"
    if not rows:
        raise RuntimeError("No simulation rows were recorded.")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_graphs(
    rows: list[dict[str, object]],
    timing_means_ms: dict[str, float],
    ciphertext_kb: float,
    max_control_error: float,
    max_state_error: float,
    state_valid: bool,
    control_valid: bool,
) -> None:
    steps = np.array([int(row["step"]) for row in rows])
    plain_u = np.array([float(row["u_plain"]) for row in rows])
    stream_u = np.array([float(row["u_lwe"]) for row in rows])
    failed = not (state_valid and control_valid)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        "LWE custom stream-cipher arithmetic — VALIDATION FAILED"
        if failed
        else "LWE custom stream-cipher arithmetic — recovery passed; comparison provisional",
        fontsize=15,
        fontweight="bold",
        color="darkred" if failed else "darkgreen",
    )

    ax = axes[0, 0]
    timing_labels = ["Encrypt", "Eval/control\n(not HE)", "Decrypt/\nrecovery", "TOTAL"]
    timing_values = [
        timing_means_ms["encrypt"],
        timing_means_ms["evaluation"],
        timing_means_ms["recovery"],
        timing_means_ms["total"],
    ]
    bars = ax.bar(timing_labels, timing_values, color=["#4472C4", "#ED7D31", "#70AD47", "#5B9BD5"])
    ax.set_title("LWE Stream Cipher — Timing (average of 200 steps)")
    ax.set_ylabel("Milliseconds per step")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, timing_values):
        ax.annotate(
            f"{value:.6g}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.text(
        0.99,
        0.97,
        "Setup/key generation excluded; evaluation uses secret-key recurrence cancellation.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    ax = axes[0, 1]
    bars = ax.bar(
        ["State ciphertext\nto controller", "Encrypted control/result\nto actuator"],
        [ciphertext_kb, 0],
        color=["#4472C4", "#A5A5A5"],
    )
    ax.set_title("Ciphertext Size")
    ax.set_ylabel("Serialized size (KB; 1024 bytes/KB)")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        f"{ciphertext_kb:.6g} KB",
        (bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2,
        max(ciphertext_kb * 0.08, 1e-5),
        "N/A",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax = axes[1, 0]
    bar = ax.bar(["Maximum |u_LWE - u_plain|"], [max_control_error], color="#C00000")[0]
    ax.set_title("Maximum Control Error")
    ax.set_ylabel("Absolute error")
    ax.grid(axis="y", alpha=0.3)
    if max_control_error > 0:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max_control_error / 10.0)
    ax.annotate(
        f"{max_control_error:.9g}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
    ax.text(
        0.98,
        0.96,
        f"Maximum state error (L2): {max_state_error:.9g}\n"
        f"State recovery: {'PASS' if state_valid else 'FAIL'}; "
        f"control recovery: {'PASS' if control_valid else 'FAIL'}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )

    ax = axes[1, 1]
    after_transient = (steps >= 19) & (steps <= 199)
    ax.plot(
        steps[after_transient],
        plain_u[after_transient],
        "-o",
        markersize=3,
        linewidth=1,
        label="Plaintext",
    )
    ax.plot(
        steps[after_transient],
        stream_u[after_transient],
        "--s",
        markersize=3,
        linewidth=1,
        label="LWE report equation (Q8 gain encoding)",
    )
    ax.set_title("Control Signal After Transient")
    ax.set_xlabel("Control step")
    ax.set_ylabel("Control input u(k)")
    ax.grid(True, alpha=0.35)
    ax.legend()

    fig.savefig(OUTPUT_DIR / "lwe_stream_four_graphs.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "lwe_stream_four_graphs.pdf")
    plt.close(fig)


def write_summary(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    ciphertext_bytes: int,
    timing_means_ms: dict[str, float],
    max_state_error: float,
    max_control_error: float,
    state_valid: bool,
    control_valid: bool,
    correctness_tests: dict[str, bool],
    noise_min: int,
    noise_max: int,
    noise_std: float,
    max_control_tolerance: float,
    max_abs_control: float,
    max_abs_scaled_control: int,
    setup_ms: float,
) -> None:
    overall_valid = state_valid and control_valid and all(correctness_tests.values())
    notes = (
        "No integer overflow was observed; all modular accumulations use Python "
        "arbitrary-precision integers."
    )
    if not control_valid:
        notes += (
            "\nAt least one recovered control exceeded its documented comparison "
            "tolerance or the centered representable range."
        )
    if not state_valid:
        notes += (
            "\nState recovery failed for at least one sample; centered decoding was "
            "outside the documented fixed-point/noise tolerance."
        )

    lines = [
        "Fresh LWE-based recursive stream-cipher simulation",
        "===================================================",
        f"State dimension: {A.shape[0]} (expected 18)",
        f"Control steps: {N_STEPS} (expected 200)",
        "",
        "Plant and controller",
        "--------------------",
        f"Masses: {N_MASSES}; mass per body: {MASS} kg",
        f"Spring constant: {SPRING} N/m; damping: {DAMPING} Ns/m",
        "Topology: first mass fixed to wall; nearest-neighbour springs/dampers; "
        "last mass free.",
        "Actuator: force on mass 9; observed state: all 18 states.",
        "State order: [pos1, vel1, pos2, vel2, ..., pos9, vel9]",
        f"Sampling interval: {DT} s; duration: {DT * N_STEPS} s",
        "Initial state: all zero except x0[16] = 1.0 m; initial velocities zero.",
        "Reference: zero-state regulation; control law u(k) = -K @ x(k).",
        "Continuous-time LQR: Q = diag(50,1,...,50,1), R = [[0.01]].",
        "K = inv(R) @ B.T @ P, P = solve_continuous_are(A,B,Q,R).",
        "Discretization: scipy.signal.cont2discrete, ZOH, dt=0.1 s.",
        f"Continuous plant matrices: A {A.shape}, B {B.shape}",
        f"Input gain K: {np.array2string(K, precision=12, separator=', ')}",
        "",
        "Cryptographic construction",
        "--------------------------",
        f"Modulus q: {Q_MOD}",
        f"Secret matrix: {N_STATES}x{N_STATES}, uniform integers in "
        f"[0, q//1000) = [0, {Q_MOD // 1000})",
        f"Noise: Box-Muller Gaussian-like integer samples; target sigma=sqrt(18)="
        f"{SIGMA:.12g}; sampled rounded noise min={noise_min}, max={noise_max}, "
        f"standard deviation={noise_std:.12g}",
        f"Random seed: {SEED}",
        f"State scale: {SCALE}; m_int=round(x*{SCALE})",
        f"Feedback representation: round((-K)*{GAIN_SCALE}); gain scale={GAIN_SCALE} "
        f"(Q8, chosen in this simulation and not specified in the report/C example), "
        f"while the report's C implementation stores F as integer coefficients "
        f"and applies the 2^16 scale to the state message. Effective control scale: "
        f"{CONTROL_SCALE}.",
        "Ciphertext recurrence: V[k]=(S@V[k-1]+m[k]+e[k]) mod q; V[-1]=0.",
        "Control equation implemented as documented: "
        "(-K)_int@V[k] - (-K)_int@(S@V[k-1] mod q), modulo q, then centered decode.",
        f"Centered signed output range: ±{(Q_MOD // 2) / CONTROL_SCALE:.12g} control units. "
        "A Q8 gain scale is used to preserve fractional gain accuracy while keeping "
        "the entire measured control trajectory within that range; the gain matrix K "
        "itself is still computed from the specified continuous-time CARE.",
        f"Measured maximum |u_LWE|: {max_abs_control:.12g}; maximum absolute unwrapped "
        f"control integer: {max_abs_scaled_control} (center limit {Q_MOD // 2}).",
        f"At this measured range, floor((q//2)/(state_scale*max|u|)) = "
        f"{int((Q_MOD // 2) / (SCALE * max_abs_control))}; Q8 is the largest "
        "power-of-two gain scale below that bound (Q9 would wrap). This is a "
        "trajectory-specific mathematical range choice, not a Sir-specified parameter.",
        "Evaluation classification: custom modular recurrence cancellation using S; "
        "this is not genuine homomorphic evaluation.",
        "",
        "Measured per-step timing (mean over 200 recorded steps)",
        "--------------------------------------------------------",
        f"Encrypt (state encode + noise + recurrence): {timing_means_ms['encrypt']:.9g} ms",
        f"Evaluation/control (reported modular equation): "
        f"{timing_means_ms['evaluation']:.9g} ms",
        f"Decrypt/recovery (centered state/control decode): "
        f"{timing_means_ms['recovery']:.9g} ms",
        f"Total (sum of measured operations): {timing_means_ms['total']:.9g} ms",
        f"Setup/key generation (RNG + S creation, excluded from per-step): "
        f"{setup_ms:.9g} ms.",
        "",
        "Serialized sizes",
        "-----------------",
        f"Serialized state ciphertext payload (would be sent to a controller): "
        f"{ciphertext_bytes} bytes "
        f"({ciphertext_bytes / 1024:.9g} KB), measured from the serialized "
        "big-endian uint32 representation. No network transfer is implemented or timed.",
        "Encrypted control/result sent to actuator: N/A; this implementation "
        "decodes the reported modular result locally and sends no result ciphertext.",
        "",
        "Numerical validation",
        "--------------------",
        f"Fixed-point state encoding/decoding: "
        f"{'PASS' if correctness_tests['fixed_point'] else 'FAIL'}",
        f"Recursive ciphertext/state recovery: "
        f"{'PASS' if correctness_tests['recursive_state'] else 'FAIL'}",
        f"Signed control recovery (positive, negative, zero, large): "
        f"{'PASS' if correctness_tests['signed_control'] else 'FAIL'}",
        f"Centered modular interval boundaries/wraparound: "
        f"{'PASS' if correctness_tests['modular_wraparound'] else 'FAIL'}",
        f"Report control equation and physical signed recovery: "
        f"{'PASS' if correctness_tests['report_equation'] else 'FAIL'}",
        f"Initial x[16]=1 control recovery against plaintext: "
        f"{'PASS' if correctness_tests['initial_control'] else 'FAIL'}",
        f"Every run state recovered within fixed-point plus sampled-noise bound: "
        f"{'PASS' if state_valid else 'FAIL'}",
        f"Control recovery against plaintext baseline: {'PASS' if control_valid else 'FAIL'}",
        f"Overall recovery validation: {'PASS' if overall_valid else 'FAIL'}",
        "Report/C gain-representation conformance: NOT ESTABLISHED (Q8 is "
        "simulation-specific; report/C use an integer F but do not define the "
        "mapping from the specified fractional LQR K to F).",
        "Standard LWE or homomorphic-encryption security: NOT CLAIMED. The "
        "implementation evaluates by secret-key modular recurrence cancellation.",
        f"Maximum documented per-step control-error tolerance: "
        f"{max_control_tolerance:.12g}",
        f"Maximum state error max_k ||x_LWE-x_plain||_2: {max_state_error:.12g}",
        f"Maximum control error max_k |u_LWE-u_plain|: {max_control_error:.12g}",
        f"Numerical/overflow issues: {notes}",
        "",
        "Failure caveat",
        "---------------",
        "The report does not specify the Q8 gain representation used here. Its C "
        "example uses integer coefficients in F and does not implement the 18-state "
        "LQR gain. Therefore arithmetic/control recovery for this explicit variant "
        "passes, but reference conformance and a definitive LWE-vs-CKKS comparison "
        "are not established. The simulated evaluation has access to the secret "
        "matrix/pad, so it can recover state values; this is not privacy-preserving "
        "untrusted-controller evaluation. The custom scheme is not standard HE and "
        "makes no security claim.",
        "When validation fails, the generated plots are diagnostic measurements only "
        "and must not be interpreted as successful encrypted control.",
    ]
    (OUTPUT_DIR / "lwe_stream_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    if N_STATES != 18 or N_STEPS != 200:
        raise RuntimeError("Configured state dimension or control-step count is wrong.")

    A, B, C, D = build_plant()
    Q = np.diag([weight for _ in range(N_MASSES) for weight in (50.0, 1.0)])
    R = np.array([[0.01]], dtype=np.float64)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    Ad, Bd, _, _, _ = signal.cont2discrete((A, B, C, D), DT, method="zoh")

    if A.shape != (18, 18) or B.shape != (18, 1) or K.shape != (1, 18):
        raise RuntimeError("Unexpected plant or controller matrix dimensions.")
    if not np.all(np.isfinite(Ad)) or not np.all(np.isfinite(Bd)):
        raise RuntimeError("Plant discretization produced non-finite values.")

    setup_start = time.perf_counter_ns()
    rng = np.random.default_rng(SEED)
    secret = rng.integers(0, Q_MOD // 1000, size=(N_STATES, N_STATES), dtype=np.int64)
    gain_int = np.rint(-K.reshape(N_STATES) * GAIN_SCALE).astype(np.int64)
    setup_ns = time.perf_counter_ns() - setup_start
    correctness_tests = run_correctness_tests(secret, gain_int, K)

    plaintext_states = np.zeros((N_STEPS + 1, N_STATES), dtype=np.float64)
    lwe_states = np.zeros((N_STEPS + 1, N_STATES), dtype=np.float64)
    plaintext_states[0, 16] = 1.0
    lwe_states[0, 16] = 1.0
    previous_ciphertext = np.zeros(N_STATES, dtype=np.int64)
    plaintext_controls = np.empty(N_STEPS, dtype=np.float64)
    lwe_controls = np.empty(N_STEPS, dtype=np.float64)
    rows: list[dict[str, object]] = []
    encrypt_ns: list[int] = []
    evaluation_ns: list[int] = []
    recovery_ns: list[int] = []
    ciphertext_size_bytes: list[int] = []
    all_noise: list[int] = []
    state_valid = True
    control_valid = True
    max_control_tolerance = 0.0
    max_abs_scaled_control = 0

    for step in range(N_STEPS):
        plain_x = plaintext_states[step]
        lwe_x = lwe_states[step]
        plain_u = float((-K @ plain_x.reshape(-1, 1))[0, 0])
        plaintext_controls[step] = plain_u

        start = time.perf_counter_ns()
        message = np.rint(lwe_x * SCALE).astype(np.int64)
        noise = box_muller_integer_noise(rng)
        all_noise.extend(int(value) for value in noise)
        current_pad = modular_matvec(secret, previous_ciphertext)
        ciphertext = (current_pad + message + noise) % Q_MOD
        encrypt_ns.append(time.perf_counter_ns() - start)
        serialized_ciphertext_bytes = len(serialize_ciphertext(ciphertext))
        ciphertext_size_bytes.append(serialized_ciphertext_bytes)

        # Equation (3.11): subtract the predicted prior-state term modulo q.
        start = time.perf_counter_ns()
        control_residue = reported_control_residue(
            gain_int, ciphertext, current_pad
        )
        evaluation_ns.append(time.perf_counter_ns() - start)

        start = time.perf_counter_ns()
        recovered_state_int = np.array(
            [
                centered(int(value) - int(pad))
                for value, pad in zip(ciphertext, current_pad)
            ],
            dtype=np.int64,
        )
        recovered_state = recovered_state_int.astype(np.float64) / SCALE
        control_fixed_centered = recover_signed_control(control_residue)
        lwe_u = control_fixed_centered / CONTROL_SCALE
        recovery_ns.append(time.perf_counter_ns() - start)
        lwe_controls[step] = lwe_u

        # Recovery tolerance covers half-unit message quantization and observed
        # integer noise; no plaintext state is substituted into the control path.
        state_tolerance = (np.abs(noise).astype(np.float64) + 1.0) / SCALE
        state_ok = bool(
            np.all(np.abs(recovered_state - lwe_x) <= state_tolerance + 1e-12)
        )
        state_valid = state_valid and state_ok
        control_error = abs(lwe_u - plain_u)
        unwrapped_control_integer = sum(
            int(a) * int(b) for a, b in zip(gain_int, message + noise)
        )
        control_centered_limit = Q_MOD // 2
        max_abs_scaled_control = max(
            max_abs_scaled_control, abs(unwrapped_control_integer)
        )
        control_ok = (
            abs(unwrapped_control_integer) <= control_centered_limit
            and control_fixed_centered == unwrapped_control_integer
        )
        # Bound coefficient quantization, inter-run state divergence, state
        # encoding quantization, and the report's F*e noise term separately.
        gain_quantized = gain_int.astype(np.float64) / GAIN_SCALE
        control_error_bound = (
            np.sum(np.abs(gain_quantized + K.reshape(-1)) * np.abs(plain_x))
            + np.sum(np.abs(gain_quantized) * np.abs(lwe_x - plain_x))
            + 0.5 * np.sum(np.abs(gain_quantized)) / SCALE
            + abs(unwrapped_control_integer - sum(
                int(a) * int(b) for a, b in zip(gain_int, message)
            )) / CONTROL_SCALE
        )
        max_control_tolerance = max(max_control_tolerance, control_error_bound)
        control_ok = control_ok and control_error <= control_error_bound + 1e-10
        control_valid = control_valid and bool(control_ok)

        # The plaintext and stream plants advance independently from their own
        # measured control inputs.
        plaintext_states[step + 1] = (
            Ad @ plain_x + Bd[:, 0] * plain_u
        )
        lwe_states[step + 1] = Ad @ lwe_x + Bd[:, 0] * lwe_u

        row: dict[str, object] = {
            "step": step,
            "time_s": step * DT,
            "u_plain": plain_u,
            "u_lwe": lwe_u,
            "state_error_l2": float(np.linalg.norm(lwe_x - plain_x)),
            "control_error_abs": control_error,
            "encrypt_ms": encrypt_ns[-1] / 1e6,
            "evaluation_ms": evaluation_ns[-1] / 1e6,
            "recovery_ms": recovery_ns[-1] / 1e6,
            "total_ms": (encrypt_ns[-1] + evaluation_ns[-1] + recovery_ns[-1]) / 1e6,
            "state_recovery_pass": state_ok,
            "control_recovery_pass": bool(control_ok),
            "enc_x_serialized_bytes": serialized_ciphertext_bytes,
        }
        row.update({f"x_plain_{i}": float(value) for i, value in enumerate(plain_x)})
        row.update({f"x_lwe_{i}": float(value) for i, value in enumerate(lwe_x)})
        rows.append(row)

        # Keep the exact recurrence state, not a centered/decoded state.
        previous_ciphertext = ciphertext

    # Include the terminal state in the maximum state-error definition.
    state_errors = np.linalg.norm(lwe_states - plaintext_states, axis=1)
    control_errors = np.abs(lwe_controls - plaintext_controls)
    max_state_error = float(np.max(state_errors))
    max_control_error = float(np.max(control_errors))
    plaintext_controls_from_rows = np.array([float(row["u_plain"]) for row in rows])
    if not np.array_equal(plaintext_controls_from_rows, plaintext_controls):
        raise RuntimeError("Recorded control series differs from the executed run.")

    timing_means_ms = {
        "encrypt": float(np.mean(encrypt_ns) / 1e6),
        "evaluation": float(np.mean(evaluation_ns) / 1e6),
        "recovery": float(np.mean(recovery_ns) / 1e6),
        "total": float(np.mean(np.array(encrypt_ns) + evaluation_ns + recovery_ns) / 1e6),
    }
    ciphertext_bytes = int(np.mean(ciphertext_size_bytes))
    ciphertext_kb = ciphertext_bytes / 1024.0
    write_csv(rows)
    make_graphs(
        rows,
        timing_means_ms,
        ciphertext_kb,
        max_control_error,
        max_state_error,
        state_valid,
        control_valid,
    )
    write_summary(
        A,
        B,
        K,
        ciphertext_bytes,
        timing_means_ms,
        max_state_error,
        max_control_error,
        state_valid,
        control_valid,
        correctness_tests,
        min(all_noise),
        max(all_noise),
        float(np.std(all_noise)),
        max_control_tolerance,
        float(np.max(np.abs(lwe_controls))),
        max_abs_scaled_control,
        setup_ns / 1e6,
    )

    print(f"State dimension: {N_STATES}; control steps: {N_STEPS}")
    print(f"Plant/LQR configuration: m={MASS}, k={SPRING}, c={DAMPING}, dt={DT}")
    print(
        "Mean latency (ms/step): "
        f"encrypt={timing_means_ms['encrypt']:.9g}, "
        f"evaluation={timing_means_ms['evaluation']:.9g}, "
        f"recovery={timing_means_ms['recovery']:.9g}, "
        f"total={timing_means_ms['total']:.9g}"
    )
    print(f"Serialized state ciphertext: {ciphertext_bytes} bytes ({ciphertext_kb:.9g} KB)")
    print(f"Maximum state error (L2): {max_state_error:.12g}")
    print(f"Maximum control error: {max_control_error:.12g}")
    print(f"Maximum documented control tolerance: {max_control_tolerance:.12g}")
    print(
        "Correctness tests: "
        + ", ".join(f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in correctness_tests.items())
    )
    print(
        f"Recovery validation: state={'PASS' if state_valid else 'FAIL'}, "
        f"control={'PASS' if control_valid else 'FAIL'}"
    )
    print(f"Setup/key-generation time (excluded): {setup_ns / 1e6:.9g} ms")
    print(f"CSV: {OUTPUT_DIR / 'lwe_stream_results.csv'}")
    print(f"Summary: {OUTPUT_DIR / 'lwe_stream_summary.txt'}")
    print(f"PNG: {OUTPUT_DIR / 'lwe_stream_four_graphs.png'}")
    print(f"PDF: {OUTPUT_DIR / 'lwe_stream_four_graphs.pdf'}")


if __name__ == "__main__":
    main()
