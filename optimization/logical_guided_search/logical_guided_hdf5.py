import numpy as np


def _ensure_ds(grp, name, shape_sample, dtype):
    if name in grp:
        return grp[name]
    shape = (0,) + shape_sample
    maxshape = (None,) + shape_sample
    return grp.create_dataset(
        name,
        shape=shape,
        maxshape=maxshape,
        dtype=dtype,
        chunks=True,
    )


def append_to_hdf5(
    grp,
    edge_list,
    params,
    ler=0.0,
    std=0.0,
    runtime=0.0,
    logical_weight=0.0,
    accepted=False,
    step=-1,
    trial=-1,
    parent_idx=-1,
    distance_before=-1.0,
    distance_after=-1.0,
    edges_to_add=None,
    edges_to_remove=None,
    screen_ler=None,
    screen_std=None,
    screen_runtime=None,
    prec_ler=None,
    prec_std=None,
    prec_runtime=None,
    score_info=None,
    selected_beam_rank=-1,
    precision_selected=False,
    final_best=False,
):
    ds_states = _ensure_ds(grp, "states", (edge_list.shape[0],), np.uint32)

    ds_ler = _ensure_ds(grp, "logical_error_rates", (), np.float64)
    ds_std = _ensure_ds(grp, "logical_error_rates_std", (), np.float64)
    ds_run = _ensure_ds(grp, "decoding_runtimes", (), np.float64)

    ds_screen_ler = _ensure_ds(
        grp, "screen_logical_error_rates", (), np.float64)
    ds_screen_std = _ensure_ds(
        grp, "screen_logical_error_rates_std", (), np.float64)
    ds_screen_run = _ensure_ds(grp, "screen_decoding_runtimes", (), np.float64)

    ds_prec_ler = _ensure_ds(grp, "prec_logical_error_rates", (), np.float64)
    ds_prec_std = _ensure_ds(
        grp, "prec_logical_error_rates_std", (), np.float64)
    ds_prec_run = _ensure_ds(grp, "prec_decoding_runtimes", (), np.float64)

    ds_dcl = _ensure_ds(grp, "distances_classical", (), np.float64)
    ds_dclT = _ensure_ds(grp, "distances_classical_T", (), np.float64)
    ds_dq = _ensure_ds(grp, "distances_quantum", (), np.float64)

    ds_logw = _ensure_ds(grp, "target_logical_weight", (), np.float64)
    ds_acc = _ensure_ds(grp, "accepted", (), np.uint8)
    ds_step = _ensure_ds(grp, "search_step", (), np.int32)
    ds_trial = _ensure_ds(grp, "search_trial", (), np.int32)

    ds_parent = _ensure_ds(grp, "parent_idx", (), np.int32)
    ds_dbefore = _ensure_ds(grp, "distance_before", (), np.float64)
    ds_dafter = _ensure_ds(grp, "distance_after", (), np.float64)

    ds_add = _ensure_ds(grp, "edges_to_add", (4,), np.int32)
    ds_remove = _ensure_ds(grp, "edges_to_remove", (4,), np.int32)

    score_data = _score_info_to_arrays(score_info)

    ds_score = _ensure_ds(grp, "low_weight_scores", (), np.float64)
    ds_score_mode = _ensure_ds(grp, "score_mode_code", (), np.int8)
    ds_score_dq = _ensure_ds(grp, "score_d_q", (), np.int32)
    ds_score_minw = _ensure_ds(grp, "score_min_weight", (), np.int32)
    ds_score_maxw = _ensure_ds(grp, "score_max_weight_used", (), np.int32)

    ds_score_counts = _ensure_expandable_2d(
        grp,
        "low_weight_counts_by_weight",
        width=len(score_data["counts_vec"]),
        dtype=np.int64,
        fillvalue=-1,
    )

    ds_score_coeffs = _ensure_expandable_2d(
        grp,
        "low_weight_coeffs_by_weight",
        width=len(score_data["coeffs_vec"]),
        dtype=np.float64,
        fillvalue=np.nan,
    )

    ds_beam_rank = _ensure_ds(grp, "selected_beam_rank", (), np.int32)
    ds_precision_selected = _ensure_ds(grp, "precision_selected", (), np.uint8)
    ds_final_best = _ensure_ds(grp, "final_best", (), np.uint8)

    idx = ds_ler.shape[0]

    screen_ler = np.nan if screen_ler is None else screen_ler
    screen_std = np.nan if screen_std is None else screen_std
    screen_runtime = np.nan if screen_runtime is None else screen_runtime

    prec_ler = np.nan if prec_ler is None else prec_ler
    prec_std = np.nan if prec_std is None else prec_std
    prec_runtime = np.nan if prec_runtime is None else prec_runtime

    ds_states.resize(idx + 1, axis=0)
    ds_states[idx] = edge_list

    ds_ler.resize(idx + 1, axis=0)
    ds_ler[idx] = ler

    ds_std.resize(idx + 1, axis=0)
    ds_std[idx] = std

    ds_run.resize(idx + 1, axis=0)
    ds_run[idx] = runtime

    ds_screen_ler.resize(idx + 1, axis=0)
    ds_screen_ler[idx] = screen_ler

    ds_screen_std.resize(idx + 1, axis=0)
    ds_screen_std[idx] = screen_std

    ds_screen_run.resize(idx + 1, axis=0)
    ds_screen_run[idx] = screen_runtime

    ds_prec_ler.resize(idx + 1, axis=0)
    ds_prec_ler[idx] = prec_ler

    ds_prec_std.resize(idx + 1, axis=0)
    ds_prec_std[idx] = prec_std

    ds_prec_run.resize(idx + 1, axis=0)
    ds_prec_run[idx] = prec_runtime

    ds_dcl.resize(idx + 1, axis=0)
    ds_dcl[idx] = params["d_classical"]

    ds_dclT.resize(idx + 1, axis=0)
    ds_dclT[idx] = params["d_T_classical"]

    ds_dq.resize(idx + 1, axis=0)
    ds_dq[idx] = params["d_quantum"]

    ds_logw.resize(idx + 1, axis=0)
    ds_logw[idx] = logical_weight

    ds_acc.resize(idx + 1, axis=0)
    ds_acc[idx] = 1 if accepted else 0

    ds_step.resize(idx + 1, axis=0)
    ds_step[idx] = step

    ds_trial.resize(idx + 1, axis=0)
    ds_trial[idx] = trial

    ds_parent.resize(idx + 1, axis=0)
    ds_parent[idx] = parent_idx

    ds_dbefore.resize(idx + 1, axis=0)
    ds_dbefore[idx] = distance_before

    ds_dafter.resize(idx + 1, axis=0)
    ds_dafter[idx] = distance_after

    ds_score.resize(idx + 1, axis=0)
    ds_score[idx] = score_data["score"]

    ds_score_mode.resize(idx + 1, axis=0)
    ds_score_mode[idx] = score_data["score_mode_code"]

    ds_score_dq.resize(idx + 1, axis=0)
    ds_score_dq[idx] = score_data["d_q"]

    ds_score_minw.resize(idx + 1, axis=0)
    ds_score_minw[idx] = score_data["min_weight"]

    ds_score_maxw.resize(idx + 1, axis=0)
    ds_score_maxw[idx] = score_data["max_weight"]

    ds_score_counts.resize(idx + 1, axis=0)
    ds_score_counts[idx, :] = -1
    ds_score_counts[idx, :len(score_data["counts_vec"])
                    ] = score_data["counts_vec"]

    ds_score_coeffs.resize(idx + 1, axis=0)
    ds_score_coeffs[idx, :] = np.nan
    ds_score_coeffs[idx, :len(score_data["coeffs_vec"])
                    ] = score_data["coeffs_vec"]

    ds_beam_rank.resize(idx + 1, axis=0)
    ds_beam_rank[idx] = selected_beam_rank

    ds_precision_selected.resize(idx + 1, axis=0)
    ds_precision_selected[idx] = 1 if precision_selected else 0

    ds_final_best.resize(idx + 1, axis=0)
    ds_final_best[idx] = 1 if final_best else 0

    add_flat = np.full(4, -1, dtype=np.int32)
    remove_flat = np.full(4, -1, dtype=np.int32)

    if edges_to_add is not None:
        add_flat[:] = np.asarray(edges_to_add, dtype=np.int32).reshape(-1)
    if edges_to_remove is not None:
        remove_flat[:] = np.asarray(
            edges_to_remove, dtype=np.int32).reshape(-1)

    ds_add.resize(idx + 1, axis=0)
    ds_add[idx] = add_flat

    ds_remove.resize(idx + 1, axis=0)
    ds_remove[idx] = remove_flat

    return idx


def update_hdf5_row(grp, idx, prec_ler=None, prec_std=None, prec_runtime=None):
    if prec_ler is not None:
        grp["prec_logical_error_rates"][idx] = prec_ler
        grp["logical_error_rates"][idx] = prec_ler

    if prec_std is not None:
        grp["prec_logical_error_rates_std"][idx] = prec_std
        grp["logical_error_rates_std"][idx] = prec_std

    if prec_runtime is not None:
        grp["prec_decoding_runtimes"][idx] = prec_runtime
        grp["decoding_runtimes"][idx] = prec_runtime


def _ensure_expandable_2d(grp, name, width, dtype, fillvalue):
    """
    Create a 2D dataset with expandable rows and expandable columns.

    Used for score components indexed by weight:
        low_weight_counts_by_weight[row, w] = A_w
    """
    if name in grp:
        ds = grp[name]

        if ds.ndim != 2:
            raise ValueError(f"{name} exists but is not 2D.")

        old_width = ds.shape[1]

        if old_width < width:
            ds.resize((ds.shape[0], width))
            if ds.shape[0] > 0:
                ds[:, old_width:width] = fillvalue

        return ds

    return grp.create_dataset(
        name,
        shape=(0, width),
        maxshape=(None, None),
        dtype=dtype,
        chunks=True,
        fillvalue=fillvalue,
    )


def _score_mode_to_code(score_mode):
    if score_mode == "relative":
        return 0
    if score_mode == "absolute":
        return 1
    return -1


def _score_info_to_arrays(score_info):
    """
    Convert score_info dict into scalar metadata + vectors indexed by weight.

    counts_vec[w] = A_w
    coeff_vec[w] = coefficient used for A_w
    """
    if score_info is None:
        return {
            "score": np.nan,
            "score_mode_code": -1,
            "d_q": -1,
            "min_weight": -1,
            "max_weight": -1,
            "counts_vec": np.full(1, -1, dtype=np.int64),
            "coeffs_vec": np.full(1, np.nan, dtype=np.float64),
        }

    components = {
        int(k): int(v)
        for k, v in score_info.get("components", {}).items()
    }

    weights = {
        int(k): float(v)
        for k, v in score_info.get("weights", {}).items()
    }

    max_weight = int(score_info.get("max_weight", 0))

    if components:
        max_weight = max(max_weight, max(components.keys()))
    if weights:
        max_weight = max(max_weight, max(weights.keys()))

    width = max_weight + 1

    counts_vec = np.full(width, -1, dtype=np.int64)
    coeffs_vec = np.full(width, np.nan, dtype=np.float64)

    for w, count in components.items():
        counts_vec[w] = count

    for w, coeff in weights.items():
        coeffs_vec[w] = coeff

    return {
        "score": float(score_info.get("score", np.nan)),
        "score_mode_code": _score_mode_to_code(score_info.get("score_mode")),
        "d_q": int(score_info.get("d_q", -1)),
        "min_weight": int(score_info.get("min_weight", -1)),
        "max_weight": int(score_info.get("max_weight", max_weight)),
        "counts_vec": counts_vec,
        "coeffs_vec": coeffs_vec,
    }


def mark_hdf5_row(
    grp,
    idx,
    selected_beam_rank=None,
    precision_selected=None,
    final_best=None,
):
    if selected_beam_rank is not None and "selected_beam_rank" in grp:
        grp["selected_beam_rank"][idx] = selected_beam_rank

    if precision_selected is not None and "precision_selected" in grp:
        grp["precision_selected"][idx] = 1 if precision_selected else 0

    if final_best is not None and "final_best" in grp:
        grp["final_best"][idx] = 1 if final_best else 0
