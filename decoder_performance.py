import time

from ldpc.bp_decoder import BpDecoder
from ldpc.bposd_decoder import BpOsdDecoder

from basic_css_code import toric_code_matrices

import numpy as np

import math
from concurrent.futures import ProcessPoolExecutor, as_completed


def generate_bsc_error(n: int, error_rate: float) -> np.ndarray:
    """
    Generate a binary symmetric channel (BSC) errors.
    
    Parameters:
        n (int): The length of the array to generate.
        error_rate (float): The probability of a bit being flipped (0 <= error_rate <= 1)

    Returns:
        np.ndarray: An array of size n with 0s and 1s, where 1 indicates an error.
    
    Example:
        >>> generate_bsc_error(10, 0.1)
        array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    """
    return np.random.binomial(1, error_rate, size=n).astype(np.uint8)

def compute_logical_error_rate(
    H,
    L,
    error_rate,
    run_count,
    DECODER,
    run_label,
    DEBUG=False,
    failure_cap=None,          # stop if logical_error > failure_cap
    min_runs_before_stop=0,    # optional safety guard
):
    logical_error = 0
    start_time = time.time()

    failed_runs_not_bp = []
    outcomes = np.zeros(run_count, dtype=np.int8)

    completed_runs = 0
    early_stopped = False

    np.random.randint(low=1, high=2**32 - 1)

    for i in range(run_count):
        error = generate_bsc_error(H.shape[1], error_rate)
        syndrome = (H @ error) % 2
        decoding = DECODER.decode(syndrome)
        residual = (decoding + error) % 2

        failed = False

        if isinstance(DECODER, BpDecoder):
            if not DECODER.converge:
                failed = True

        if not failed and np.any((L @ residual) % 2):
            failed = True
            if isinstance(DECODER, BpOsdDecoder):
                failed_runs_not_bp.append(i)
            if DEBUG:
                print(f"Failed run: {i}")
                print(f"Syndrome: {np.nonzero(syndrome)[0].__repr__()}")

        if failed:
            logical_error += 1
            outcomes[i] = 1

        completed_runs += 1

        # EARLY STOP
        if (
            failure_cap is not None
            and completed_runs >= min_runs_before_stop
            and logical_error > failure_cap
        ):
            early_stopped = True
            break

    end_time = time.time()
    runtime = end_time - start_time

    logical_error_rate = logical_error / completed_runs
    stderr = (
        np.std(outcomes[:completed_runs], ddof=1) / np.sqrt(completed_runs)
        if completed_runs > 1 else 0.0
    )

    status = "EARLY-STOPPED" if early_stopped else "finished"
    print(
        f"Decoder {run_label} {status} in {runtime//60}m {runtime%60:.2f}s "
        f"with {logical_error} failures out of {completed_runs}/{run_count} runs."
    )
    print(
        f"Logical error rate for {run_label}: "
        f"{logical_error_rate} ± {stderr:.7f} (stderr)"
    )

    return logical_error_rate, stderr, runtime, logical_error, completed_runs, early_stopped

def _mc_worker(args):
    Hz, Lz, p, local_runs, bp_max_iter, seed = args

    rng = np.random.default_rng(seed)

    decoder = BpOsdDecoder(
        pcm=Hz,
        error_rate=float(p),
        max_iter=bp_max_iter,
        bp_method="minimum_sum",
        ms_scaling_factor=0.625,
        schedule="parallel",
        osd_method="OSD_CS",
        osd_order=2,
    )

    failures = 0
    outcomes = np.zeros(local_runs, dtype=np.int8)

    start = time.time()

    for i in range(local_runs):
        error = rng.binomial(1, p, size=Hz.shape[1]).astype(np.uint8)
        syndrome = (Hz @ error) % 2
        decoding = decoder.decode(syndrome)
        residual = (decoding + error) % 2

        failed = False

        if isinstance(decoder, BpDecoder):
            if not decoder.converge:
                failed = True

        if not failed and np.any((Lz @ residual) % 2):
            failed = True

        if failed:
            failures += 1
            outcomes[i] = 1

    runtime = time.time() - start
    completed_runs = local_runs
    return failures, completed_runs, outcomes, runtime

def compute_logical_error_rate_parallel(
    Hz,
    Lz,
    error_rate,
    run_count,
    run_label,
    workers=8,
):
    if run_count == 0:
        return 0.0, 0.0, 0.0, 0, 0, False

    start_time = time.time()
    bp_max_iter = int(Hz.shape[1] / 10)

    workers = min(workers, run_count)
    chunk_size = math.ceil(run_count / workers)

    jobs = []
    base_seed = np.random.SeedSequence().entropy

    for w in range(workers):
        local_runs = min(chunk_size, run_count - w * chunk_size)
        if local_runs <= 0:
            continue
        seed = int(base_seed) + w
        jobs.append((Hz, Lz, error_rate, local_runs, bp_max_iter, seed))

    total_failures = 0
    total_completed = 0
    all_outcomes = []
    worker_runtimes = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_mc_worker, job) for job in jobs]

        for fut in as_completed(futures):
            failures, completed_runs, outcomes, runtime = fut.result()
            total_failures += failures
            total_completed += completed_runs
            all_outcomes.append(outcomes)
            worker_runtimes.append(runtime)

    runtime = time.time() - start_time
    outcomes = np.concatenate(all_outcomes)

    ler = total_failures / total_completed
    stderr = (
        np.std(outcomes, ddof=1) / np.sqrt(total_completed)
        if total_completed > 1 else 0.0
    )

    print(
        f"Decoder {run_label} finished in {runtime//60}m {runtime%60:.2f}s "
        f"with {total_failures} failures out of {total_completed}/{run_count} runs."
    )
    print(f"Logical error rate for {run_label}: {ler} ± {stderr:.7f} (stderr)")

    return ler, stderr, runtime, total_failures, total_completed, False


def compute_logical_error_rate_parallel_batched(
    Hz,
    Lz,
    error_rate,
    run_count,
    run_label,
    workers=8,
    batch_size=5000,
    failure_cap=None,
    min_runs_before_stop=0,
):
    start_time = time.time()

    total_failures = 0
    total_completed = 0
    all_outcomes = []
    early_stopped = False

    remaining = run_count
    batch_idx = 0

    while remaining > 0:
        current_batch = min(batch_size, remaining)

        ler, stderr, _, failures, completed, _ = compute_logical_error_rate_parallel(
            Hz=Hz,
            Lz=Lz,
            error_rate=error_rate,
            run_count=current_batch,
            run_label=f"{run_label}_batch{batch_idx}",
            workers=workers,
        )

        total_failures += failures
        total_completed += completed

        batch_outcomes = np.zeros(completed, dtype=np.int8)
        batch_outcomes[:failures] = 1
        all_outcomes.append(batch_outcomes)

        if (
            failure_cap is not None
            and total_completed >= min_runs_before_stop
            and total_failures > failure_cap
        ):
            early_stopped = True
            break

        remaining -= current_batch
        batch_idx += 1

    runtime = time.time() - start_time
    outcomes = np.concatenate(all_outcomes) if all_outcomes else np.array([], dtype=np.int8)

    ler = total_failures / total_completed if total_completed > 0 else 0.0
    stderr = (
        np.std(outcomes, ddof=1) / np.sqrt(total_completed)
        if total_completed > 1 else 0.0
    )

    print(f"Decoder {run_label} finished in {runtime//60}m {runtime%60:.2f}s "
            f"with {total_failures} failures out of {total_completed}/{run_count} runs."
    )
    print(f"Logical error rate for {run_label}: {ler} ± {stderr:.7f} (stderr)")

    return ler, stderr, runtime, total_failures, total_completed, early_stopped

if __name__ == "__main__":
    from basic_css_code import toric_code_matrices

    import matplotlib.pyplot as plt

    # compare between BP and BP+OSD decoders with different distances of the toric code

    distances = [9, 11, 13, 15]
    error_rates = np.linspace(0.09, 0.18, 18)  # Error rates from 0.09 to 0.18
    max_iter = 100
    ms_scaling_factor = 0.625
    osd_order = 60  # Order of the OSD method
    run_count = 10000 # Number of runs for each error rate
    logical_error_rates_bp = []
    logical_error_rates_bp_osd = []

    for d in distances:
        print(f"Distance: {d}")
        logical_error_rates_bp_with_d = []
        logical_error_rates_bp_osd_with_d = []

        Hx, Hz, Lx, Lz = toric_code_matrices(d)
        # remove the last row of Hx and Hz
        Hx = Hx[:-1, :]
        Hz = Hz[:-1, :]
        # H = vstack([Hx, Hz])

        for p in error_rates:
            # Initialize the BP decoder
            # bp_decoder = BpDecoder(
            # pcm=Hz,
            # error_rate=float(p),
            # max_iter=max_iter,
            # ms_scaling_factor=ms_scaling_factor,
            # schedule='parallel',
            # )

            # logical_error_rate_bp = compute_logical_error_rate(
            #     Hz, Lx, p, run_count=run_count, DECODER=bp_decoder, run_label=f"BP (d={d}, p={p})"
            # )
            # logical_error_rates_bp_with_d.append(logical_error_rate_bp)

            bp_osd_decoder = BpOsdDecoder(
                pcm=Hz,
                error_rate=float(p),
                max_iter=max_iter,
                bp_method='minimum_sum',
                ms_scaling_factor=ms_scaling_factor,
                schedule='parallel',
                osd_method='OSD_CS',
                osd_order=osd_order,
            )
            ler, stderr, runtime, failures, completed_runs, early_stopped = compute_logical_error_rate(
                Hz, Lx, p, run_count=run_count, DECODER=bp_osd_decoder, run_label=f"BP+OSD (d={d}, p={p})"
            )
            logical_error_rates_bp_osd_with_d.append(ler)

        # logical_error_rates_bp.append(logical_error_rates_bp_with_d)
        logical_error_rates_bp_osd.append(logical_error_rates_bp_osd_with_d)


    colors = {
    9: 'tab:red',
    11: 'tab:blue',
    13: 'tab:green',
    15: 'tab:purple'
    }

    plt.figure(figsize=(5, 6))
    plt.title("Toric Code Decoders Comparison (BP vs BP+OSD)")
    plt.xlabel("Bit-error Rate")
    plt.ylabel(r"Logical-error rate, $p_L$")
    plt.xscale('log')
    plt.yscale('log')

    # Use different loop variables to avoid overwriting
    # for d, bp_vals in zip(distances, logical_error_rates_bp):
    #     plt.plot(
    #         error_rates, bp_vals,
    #         label=f"[[{2 * d**2}, 2, {d}]] BP",
    #         marker='o',
    #         linestyle='--',
    #         color=colors[d]
    #     )
    
    for d, bp_osd_vals in zip(distances, logical_error_rates_bp_osd):
        plt.plot(
            error_rates, bp_osd_vals,
            label=f"[[{2 * d**2}, 2, {d}]] BP+OSD",
            marker='o',
            linestyle='-',
            color=colors[d]
        )

    plt.legend(bbox_to_anchor=(1, 1.5), ncol=2)
    plt.tight_layout()
    plt.show()
    plt.savefig("toric_code_decoders_comparison.png", dpi=300, bbox_inches='tight')