import torch

def split_nd_rectangle_variable(state_lower_bound, state_upper_bound, M, pt_dtype):
    """
    Splits an N-dimensional rectangle into sub-rectangles with variable splits per dimension.

    Args:
        state_lower_bound (torch.Tensor): Lower bounds of the N dimensions, shape (N,)
        state_upper_bound (torch.Tensor): Upper bounds of the N dimensions, shape (N,)
        M (torch.Tensor): Number of splits per dimension, shape (N,), each M_i >=1
        config: Configuration object with attribute pt_dtype for tensor dtype

    Returns:
        state_lower_bound_all (torch.Tensor): Lower bounds of all sub-rectangles, shape (total_sub_rects, N)
        state_upper_bound_all (torch.Tensor): Upper bounds of all sub-rectangles, shape (total_sub_rects, N)
    """
    N = state_lower_bound.shape[0]

    if state_upper_bound.shape[0] != N or M.shape[0] != N:
        raise ValueError("state_lower_bound, state_upper_bound, and M must all have the same length N.")

    if torch.any(M < 1):
        raise ValueError("All elements of M must be at least 1.")

    # Compute step sizes for each dimension
    step = (state_upper_bound - state_lower_bound) / M.type_as(state_lower_bound)  # Shape: (N,)

    # Compute total number of sub-rectangles
    total_sub_rects = M.prod().item()

    # If total_sub_rects is zero, return empty tensors
    if total_sub_rects == 0:
        return (torch.empty((0, N), dtype=pt_dtype),
                torch.empty((0, N), dtype=pt_dtype))

    # Compute strides for each dimension to map 1D indices to multi-dimensional indices
    # strides[i] = product of M_j for j < i
    if N == 0:
        # Handle the edge case of N=0
        state_lower_bound_all = torch.empty((0, 0), dtype=pt_dtype)
        state_upper_bound_all = torch.empty((0, 0), dtype=pt_dtype)
        return state_lower_bound_all, state_upper_bound_all

    strides = torch.cat([torch.tensor([1], dtype=torch.long), torch.cumprod(M[:-1], dim=0)]).tolist()  # List of length N

    # Generate all 1D indices
    indices = torch.arange(total_sub_rects, dtype=torch.long, device=state_lower_bound.device)  # Shape: (total_sub_rects,)

    # Initialize a list to hold multi-dimensional indices for each dimension
    multi_idx = []
    for stride, m in zip(strides, M.tolist()):
        idx = (indices // stride) % m  # Shape: (total_sub_rects,)
        multi_idx.append(idx)

    # Stack to form a (total_sub_rects, N) tensor
    multi_idx = torch.stack(multi_idx, dim=1)  # Shape: (total_sub_rects, N)

    # Compute lower bounds for all sub-rectangles
    # state_lower_bound: (N,) -> (1, N)
    # multi_idx: (total_sub_rects, N)
    state_lower_bound_all = state_lower_bound.unsqueeze(0) + multi_idx.type_as(step) * step.unsqueeze(0)  # Shape: (total_sub_rects, N)

    # Compute upper bounds for all sub-rectangles
    state_upper_bound_all = state_lower_bound.unsqueeze(0) + (multi_idx.type_as(step) + 1) * step.unsqueeze(0)  # Shape: (total_sub_rects, N)

    return state_lower_bound_all, state_upper_bound_all

def split_nd_rectangle_at_index(
    state_lower_bound: torch.Tensor,
    state_upper_bound: torch.Tensor,
    M: torch.Tensor,
    index: int,
    pt_dtype=torch.float32
    ) -> (torch.Tensor, torch.Tensor):
    """
    Returns the sub-rectangle (lower and upper bounds) at the given 1D index
    when splitting an N-dimensional rectangle into sub-rectangles with variable
    splits per dimension.

    Args:
        state_lower_bound (torch.Tensor):
            Lower bounds of the N dimensions, shape (N,).
        state_upper_bound (torch.Tensor):
            Upper bounds of the N dimensions, shape (N,).
        M (torch.Tensor):
            Number of splits per dimension, shape (N,). Each M_i >= 1.
        index (int):
            The 0-based index of the sub-rectangle to return.
        pt_dtype (torch.dtype):
            The PyTorch dtype for the returned tensors.

    Returns:
        (torch.Tensor, torch.Tensor):
            A pair (lower_bound, upper_bound), each of shape (N,). 
    """
    # Basic checks
    N = state_lower_bound.shape[0]
    if state_upper_bound.shape[0] != N or M.shape[0] != N:
        raise ValueError("state_lower_bound, state_upper_bound, and M must all have the same length N.")

    if torch.any(M < 1):
        raise ValueError("All elements of M must be at least 1.")

    # Compute step sizes for each dimension: (N,)
    step = (state_upper_bound - state_lower_bound) / M.type_as(state_lower_bound)

    # Total number of sub-rectangles
    total_sub_rects = int(M.prod().item())
    if total_sub_rects == 0:
        raise ValueError("No sub-rectangles exist because M.prod()=0.")

    if not (0 <= index < total_sub_rects):
        raise IndexError(f"Index {index} is out of range (total sub-rects = {total_sub_rects}).")

    # Edge case: if N=0, return empty lower and upper
    if N == 0:
        # Usually N=0 is quite unusual, but we'll handle it gracefully
        return (
            torch.empty((0,), dtype=pt_dtype),
            torch.empty((0,), dtype=pt_dtype)
        )

    # strides[i] = product of M_j for j < i
    # used for decoding the 1D index into an N-D index
    strides = torch.cat([
        torch.tensor([1], dtype=torch.long, device=M.device),
        torch.cumprod(M[:-1], dim=0)
    ])

    # Decode `index` into an N-dimensional index: multi_idx[dim] = index in dimension "dim"
    multi_idx = []
    tmp_index = index
    for dim in range(N):
        stride = strides[dim].item()
        idx_dim = (tmp_index // stride) % M[dim].item()
        multi_idx.append(idx_dim)

    # Convert to tensor
    multi_idx_t = torch.tensor(multi_idx, dtype=step.dtype, device=step.device)

    # Compute lower and upper bound for this sub-rectangle
    sub_lower = state_lower_bound + multi_idx_t * step
    sub_upper = state_lower_bound + (multi_idx_t + 1) * step

    # Cast to requested dtype
    return sub_lower.to(pt_dtype), sub_upper.to(pt_dtype)