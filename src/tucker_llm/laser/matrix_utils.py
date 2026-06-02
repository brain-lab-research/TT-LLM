import torch
import numpy as np
import matplotlib.pyplot as plt


# Helper functions for abs weight pruning
def sorted_mat(matrix):
    temp = list(abs(matrix).flatten())
    temp.sort()
    return temp


def prune(matrix, mat_sort, to_prune):
    if to_prune != 0:
        alpha = mat_sort[int(to_prune * 0.1 * len(mat_sort))]
        matrix[abs(matrix) <= alpha] = 0
    return matrix


def rank(matrix):
    np_matrix = np.array(matrix)
    return np.linalg.matrix_rank(np_matrix) / min(list(np_matrix.shape))


def sparsity(matrix, alpha):
    abs_matrix = abs(matrix)
    filtered_matrix = abs_matrix[abs_matrix < alpha]
    return len(filtered_matrix) / matrix.size


def viz_rank_change(rank_list, name):
    fig = plt.figure()
    plt.plot(rank_list)
    plt.savefig(name)


def do_low_rank(weight, k, debug=False, niter=2):
    assert weight.ndim == 2

    original_device = weight.device
    original_dtype = weight.dtype

    weight_work = weight.detach().to(device=original_device, dtype=torch.float32)

    max_rank = min(weight_work.shape[0], weight_work.shape[1])
    desired_rank = int(max_rank * k)
    desired_rank = max(1, min(desired_rank, max_rank))

    if debug:
        print(
            f"Shape is {weight_work.shape}, dtype is {weight_work.dtype}, "
            f"device is {weight_work.device} => desired rank {desired_rank}"
        )

    results = torch.svd_lowrank(
        weight_work,
        q=desired_rank,
        niter=niter,
    )
    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0]
    assert weight_approx.shape[1] == weight.shape[1]

    return weight_approx.to(device=original_device, dtype=original_dtype)