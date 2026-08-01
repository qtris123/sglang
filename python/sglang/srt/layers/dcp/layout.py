# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pure index math for decode context parallel (DCP): per-rank lengths and
the owner-rule local-index filter."""

import torch

from sglang.srt.runtime_context import get_parallel


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-rank visible KV length under the owner rule pos % dcp_size == dcp_rank.

    Superset implementation (PR #25090): supports both start=None and a per-request
    `start` offset. update_local_kv_lens_for_dcp is the start=None special case.
    """
    if dcp_size == 1:
        return lens
    if start is None:
        return lens // dcp_size + (dcp_rank < lens % dcp_size)

    first = start + torch.remainder(dcp_rank - start, dcp_size)
    remaining = start + lens - first
    return torch.clamp((remaining + dcp_size - 1) // dcp_size, min=0)


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kv_indices = (
            kv_indices[kv_indices % parallel.dcp_size == parallel.dcp_rank]
            // parallel.dcp_size
        )
    return kv_indices


def update_local_kv_lens_for_dcp(kv_len_arr):
    """In-place per-rank KV length: the start=0 case of get_dcp_lens.

    floor((len - rank - 1) / N) + 1  ==  len // N + (rank < len % N)  for len >= 0
    (bit-identical; see test/registered/cp/test_dcp_layout_unit.py). Kept as an
    in-place mutation because callers (plan_dcp_decode_metadata, the FlashInfer-MLA
    cuda-graph replay path) rely on it.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return
    kv_len_arr.copy_(get_dcp_lens(kv_len_arr, parallel.dcp_size, parallel.dcp_rank))


def dcp_paged_stride(page_size: int, dcp_size: int) -> int:
    """Token-space stride of one page in the DCP-widened virtual address space.

    Under DCP the allocator hands out VIRTUAL token ids and widens its page to
    ``page_size * dcp_size``, while the physical row of a virtual id is
    ``v // dcp_size``. Collapses to ``page_size`` when DCP is off.
    """
    return page_size * dcp_size


def build_dcp_page_table(
    loc_table: torch.Tensor,
    page_size: int,
    dcp_size: int,
    strided_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a paged block table from virtual token ids, DCP-aware.

    ``loc_table`` is ``[B, max_seq]`` of virtual token ids as stored in
    ``req_to_token``. The result is ``[B, ceil(max_seq / stride)]`` of PHYSICAL page
    indices, where ``stride = page_size * dcp_size``.

    The table is IDENTICAL on every rank; only the per-rank sequence lengths differ
    (see ``dcp_local_cache_seqlens``). That is the whole reason DCP needs no gather
    or scatter here, and it follows from the owner rule plus allocator alignment:

        widened page ``k`` spans virtual ids ``[k*P*D, (k+1)*P*D)``; rank ``r`` owns
        those ``≡ r (mod D)``, i.e. ``v = k*P*D + r + j*D`` for ``j in [0, P)``;
        their physical rows are ``v // D = k*P + j``, which is exactly physical page
        ``k``, fully covered, for every rank.

    Note the reduction is by ``stride``, not by ``page_size``: with ``page_size == 1``
    the non-DCP code leaves token ids untouched, but under DCP the ``// dcp_size``
    virtual-to-physical step is still required. Callers must therefore key the
    "needs reduction" branch on ``stride > 1``, not on ``page_size > 1``.

    ``strided_indices`` may be passed to reuse a cached ``arange``; ``out`` writes
    into a fixed-address buffer for CUDA-graph replay.
    """
    stride = dcp_paged_stride(page_size, dcp_size)
    if strided_indices is None:
        strided_indices = torch.arange(
            0, loc_table.shape[1], stride, device=loc_table.device
        )
    table = loc_table[:, strided_indices] // stride
    if out is not None:
        out[:, : table.shape[1]].copy_(table)
        return out
    return table


def dcp_local_cache_seqlens(
    seq_lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-rank owned KV count to hand the attention kernel as ``cache_seqlens``.

    Thin wrapper over ``get_dcp_lens`` adding an optional fixed-address destination
    for CUDA-graph replay; the arithmetic is not reimplemented here.
    """
    lens = get_dcp_lens(seq_lens, dcp_size, dcp_rank)
    if out is not None:
        out[: lens.shape[0]].copy_(lens)
        return out
    return lens
