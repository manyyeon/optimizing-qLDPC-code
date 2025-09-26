import time

from ldpc.bp_decoder import BpDecoder
from ldpc.bposd_decoder import BpOsdDecoder

from basic_css_code import toric_code_matrices

import numpy as np


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

def compute_logical_error_rate(H, L, error_rate, run_count, DECODER, run_label, DEBUG=False) -> float:
    """
    Calculate logical error rate for a given decoder.
    
    Parameters:
        H (np.ndarray): The parity check matrix. # bit-flip error channel -> Hz, phase-flip error channel -> Hx
        L (np.ndarray): The logical operator matrix. # bit-flip error channel -> Lz, phase-flip error channel -> Lx
        error_rate (float): The error rate for the BSC.
        run_count (int): The number of runs to perform.
        DECODER: The decoder instance to use (BpDecoder or BpOsdDecoder).
        run_label (str): A label for the run, used for logging.
        DEBUG (bool): If True, print debug information.

    Returns:
        logical_error_rate (float): The logical error rate calculated as the number of logical errors divided by the number of
        std (float): The standard deviation of the logical error rate.
        run_time (float): The time taken to perform the runs.
    """

    logical_error = 0

    start_time = time.time()

    failed_runs_not_bp = []
    outcomes = np.zeros(run_count, dtype=np.int8)

    np.random.randint(low=1,high=2**32-1)

    for i in range(run_count):
        error = generate_bsc_error(H.shape[1], error_rate)
        syndrome = (H @ error) % 2
        decoding = DECODER.decode(syndrome)
        residual = (decoding + error) % 2

        if isinstance(DECODER, BpDecoder):
            if not DECODER.converge:
                logical_error += 1
                outcomes[i] = 1
                continue
        
        if np.any((L @ residual) % 2):
            logical_error += 1
            outcomes[i] = 1
            if isinstance(DECODER, BpOsdDecoder):
                failed_runs_not_bp.append(i)
            if DEBUG:
                print(f"Failed run: {i}")
                print(f"Syndrome: {np.nonzero(syndrome)[0].__repr__()}")

    end_time = time.time()
    runtime = end_time - start_time
    logical_error_rate = logical_error / run_count
    # std = np.sqrt(logical_error_rate * (1 - logical_error_rate) / (run_count - 1))
    stderr = np.std(outcomes, ddof=1) / np.sqrt(run_count) 

    print(f"Decoder {run_label} finished in {runtime//60}m {runtime%60:.2f}s with {logical_error} failures out of {run_count} runs.")
    print(f"Logical error rate for {run_label}: {logical_error_rate} ± {stderr:.7f} (stderr)")

    return logical_error_rate, stderr, runtime

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
            logical_error_rate_bp_osd = compute_logical_error_rate(
                Hz, Lx, p, run_count=run_count, DECODER=bp_osd_decoder, run_label=f"BP+OSD (d={d}, p={p})"
            )
            logical_error_rates_bp_osd_with_d.append(logical_error_rate_bp_osd)

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