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
):
    ds_states = _ensure_ds(grp, "states", (edge_list.shape[0],), np.uint32)
    ds_ler = _ensure_ds(grp, "logical_error_rates", (), np.float64)
    ds_std = _ensure_ds(grp, "logical_error_rates_std", (), np.float64)
    ds_dcl = _ensure_ds(grp, "distances_classical", (), np.float64)
    ds_dclT = _ensure_ds(grp, "distances_classical_T", (), np.float64)
    ds_dq = _ensure_ds(grp, "distances_quantum", (), np.float64)
    ds_run = _ensure_ds(grp, "decoding_runtimes", (), np.float64)

    ds_logw = _ensure_ds(grp, "target_logical_weight", (), np.float64)
    ds_acc = _ensure_ds(grp, "accepted", (), np.uint8)
    ds_step = _ensure_ds(grp, "search_step", (), np.int32)
    ds_trial = _ensure_ds(grp, "search_trial", (), np.int32)

    ds_parent = _ensure_ds(grp, "parent_idx", (), np.int32)
    ds_dbefore = _ensure_ds(grp, "distance_before", (), np.float64)
    ds_dafter = _ensure_ds(grp, "distance_after", (), np.float64)

    ds_add = _ensure_ds(grp, "edges_to_add", (4,), np.int32)
    ds_remove = _ensure_ds(grp, "edges_to_remove", (4,), np.int32)

    idx = ds_ler.shape[0]

    ds_states.resize(idx + 1, axis=0)
    ds_states[idx] = edge_list

    ds_ler.resize(idx + 1, axis=0)
    ds_ler[idx] = ler

    ds_std.resize(idx + 1, axis=0)
    ds_std[idx] = std

    ds_dcl.resize(idx + 1, axis=0)
    ds_dcl[idx] = params["d_classical"]

    ds_dclT.resize(idx + 1, axis=0)
    ds_dclT[idx] = params["d_T_classical"]

    ds_dq.resize(idx + 1, axis=0)
    ds_dq[idx] = params["d_quantum"]

    ds_run.resize(idx + 1, axis=0)
    ds_run[idx] = runtime

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

    add_flat = np.full(4, -1, dtype=np.int32)
    remove_flat = np.full(4, -1, dtype=np.int32)

    if edges_to_add is not None:
        add_flat[:] = np.asarray(edges_to_add, dtype=np.int32).reshape(-1)
    if edges_to_remove is not None:
        remove_flat[:] = np.asarray(edges_to_remove, dtype=np.int32).reshape(-1)

    ds_add.resize(idx + 1, axis=0)
    ds_add[idx] = add_flat

    ds_remove.resize(idx + 1, axis=0)
    ds_remove[idx] = remove_flat

    return idx


def update_hdf5_row(grp, idx, ler, std, runtime):
    grp["logical_error_rates"][idx] = ler
    grp["logical_error_rates_std"][idx] = std
    grp["decoding_runtimes"][idx] += runtime