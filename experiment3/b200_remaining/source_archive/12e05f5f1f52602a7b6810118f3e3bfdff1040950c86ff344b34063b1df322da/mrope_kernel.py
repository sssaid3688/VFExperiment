import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import rsqrt

@triton.jit
def _triton_qwen2vl_mrope(
    q_ptr,
    k_ptr,
    cos,
    sin,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)

    # locate start address
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################

    # 1. program instances are laid out in a 1D vector of size bsz * seq_len, which
    # effectively represents a 2D grid of size [bsz, seq_len] with seq_len dimension
    # being the fastest changing dimension. Thus we can simply do pid // sl to get the batch index
    # and pid % sl to get the sequence index.
    # 2. We only need the left half of cos and sin matrix because the right half is just
    # a clone of the left half.
    t_end = mrope_section_t
    h_end = t_end + mrope_section_h

    t_cos = cos + pid * hd
    h_cos = t_cos + bs * sl * hd
    w_cos = h_cos + bs * sl * hd
    t_sin = sin + pid * hd
    h_sin = t_sin + bs * sl * hd
    w_sin = h_sin + bs * sl * hd

    cos_offsets = tl.arange(0, pad_hd // 2)
    t_mask = cos_offsets < t_end
    h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
    w_mask = (h_end <= cos_offsets) & (cos_offsets < hd // 2)
    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    if not BACKWARD_PASS:
        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        # with some math, we can get:
        # dy = [dx1, dx2] * [cos, cos] + [-dx2, dx1] * [-sin, -sin]
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
