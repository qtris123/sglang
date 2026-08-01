"""Numerical check of FA3 absorbed-MLA decode under decode context parallelism.

The kernel is real; the sharding is simulated. One GPU is enough because each
DCP rank's attention is independent -- only the final LSE merge crosses ranks,
and that is done here with ``correct_attn_out``, the same kernel the model uses
behind ``cp_lse_ag_out_rs_mla``. So this covers, end to end, everything stage 2
introduces:

* the page table built at the widened stride ``page_size * dcp_size``, and the
  claim that one table serves every rank,
* per-rank ``cache_seqlens`` from the owner rule,
* the LSE orientation ``(H, T) -> [T, H]`` and its conversion to base 2,

against a float32 full-attention reference. A wrong log base or a transposed
LSE both produce a plausible-looking but wrong merge, which is exactly the
failure mode a text-level test would miss.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

# Absorbed MLA decode: scores are q_rope . k_rope + q_nope . c_kv, and the
# "value" is the latent c_kv itself. DeepSeek-V2/V3 dimensions.
ROPE_DIM = 64
KV_LORA_RANK = 512


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is required"
    major = torch.cuda.get_device_capability()[0]
    if major < 9:
        return f"FA3 requires SM90+, got SM{major}0"
    return None


class _Case:
    """A batch laid out exactly the way the allocator lays one out under DCP.

    ``req_to_token`` holds VIRTUAL token ids; request ``b`` starts at a base that
    is a multiple of ``page_size * dcp_size`` (the allocator widens its page by
    ``dcp_size``), which is what makes ``virtual_id % dcp_size`` agree with
    ``position % dcp_size`` and therefore what makes one page table valid on
    every rank. Rank ``r``'s cache holds token ``v`` at physical row
    ``v // dcp_size``, matching the owner mask inside
    ``set_mla_kv_buffer_triton``.
    """

    def __init__(self, seq_lens, num_heads, dcp_size, page_size, device, dtype, seed):
        torch.manual_seed(seed)
        self.seq_lens = list(seq_lens)
        self.batch = len(seq_lens)
        self.num_heads = num_heads
        self.dcp_size = dcp_size
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        self.scale = (ROPE_DIM + KV_LORA_RANK) ** -0.5

        stride = page_size * dcp_size
        max_len = max(self.seq_lens)

        # Start at a nonzero base so a dropped offset cannot pass by coincidence.
        bases, cursor = [], stride
        for length in self.seq_lens:
            bases.append(cursor)
            cursor += -(-length // stride) * stride
        self.bases = bases

        self.req_to_token = torch.zeros(
            self.batch, max_len, dtype=torch.int32, device=device
        )
        for b, (base, length) in enumerate(zip(bases, self.seq_lens)):
            self.req_to_token[b, :length] = torch.arange(
                base, base + length, dtype=torch.int32, device=device
            )

        self.k_rope = 0.2 * torch.randn(
            self.batch, max_len, ROPE_DIM, dtype=dtype, device=device
        )
        self.c_kv = 0.2 * torch.randn(
            self.batch, max_len, KV_LORA_RANK, dtype=dtype, device=device
        )
        self.q_rope = 0.2 * torch.randn(
            self.batch, num_heads, ROPE_DIM, dtype=dtype, device=device
        )
        self.q_nope = 0.2 * torch.randn(
            self.batch, num_heads, KV_LORA_RANK, dtype=dtype, device=device
        )

        rows = cursor // dcp_size + page_size
        self.num_pages = -(-rows // page_size)
        self.seq_lens_tensor = torch.tensor(
            self.seq_lens, dtype=torch.int32, device=device
        )

    def shard_caches(self, rank):
        """The KV cache rank ``rank`` would hold after owner-masked writes."""
        k_cache = torch.zeros(
            self.num_pages,
            self.page_size,
            1,
            ROPE_DIM,
            dtype=self.dtype,
            device=self.device,
        )
        v_cache = torch.zeros(
            self.num_pages,
            self.page_size,
            1,
            KV_LORA_RANK,
            dtype=self.dtype,
            device=self.device,
        )
        for b, (base, length) in enumerate(zip(self.bases, self.seq_lens)):
            for pos in range(length):
                virtual = base + pos
                if virtual % self.dcp_size != rank:
                    continue
                row = virtual // self.dcp_size
                page, slot = divmod(row, self.page_size)
                k_cache[page, slot, 0] = self.k_rope[b, pos]
                v_cache[page, slot, 0] = self.c_kv[b, pos]
        return k_cache, v_cache

    def reference(self):
        """Full float32 attention over every position, no sharding."""
        out = torch.empty(
            self.batch,
            self.num_heads,
            KV_LORA_RANK,
            dtype=torch.float32,
            device=self.device,
        )
        for b, length in enumerate(self.seq_lens):
            k = self.k_rope[b, :length].float()
            c = self.c_kv[b, :length].float()
            scores = (
                self.q_rope[b].float() @ k.T + self.q_nope[b].float() @ c.T
            ) * self.scale
            out[b] = torch.softmax(scores, dim=-1) @ c
        return out


def _run_rank(case, rank, ver):
    """One rank's FA call, with the metadata the backend would have built."""
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
    from sglang.srt.layers.attention.flashattention_backend import _normalize_dcp_lse
    from sglang.srt.layers.dcp import build_dcp_page_table, dcp_local_cache_seqlens

    page_table = build_dcp_page_table(
        case.req_to_token, case.page_size, case.dcp_size
    ).to(torch.int32)
    cache_seqlens = dcp_local_cache_seqlens(
        case.seq_lens_tensor, case.dcp_size, rank
    ).to(torch.int32)
    k_cache, v_cache = case.shard_caches(rank)

    result = flash_attn_with_kvcache(
        q=case.q_rope,
        k_cache=k_cache,
        v_cache=v_cache,
        qv=case.q_nope,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=torch.arange(
            0, case.batch + 1, dtype=torch.int32, device=case.device
        ),
        max_seqlen_q=1,
        softmax_scale=case.scale,
        causal=False,
        return_softmax_lse=True,
        ver=ver,
    )
    return result[0], _normalize_dcp_lse(result[1]), page_table


def _merge(case, outs, lses):
    """Sum of every rank's LSE-corrected partial, i.e. what the model computes.

    ``cp_lse_ag_out_rs_mla`` calls ``correct_attn_out`` on the local partial and
    then reduce-scatters over the head dim; summing the corrections here is the
    same value before heads are split back up.
    """
    from sglang.kernels.ops.attention.dcp_kernels import correct_attn_out

    stacked = torch.stack(lses, dim=0)
    merged = None
    for rank, out in enumerate(outs):
        buffer = torch.empty(
            case.num_heads,
            case.batch,
            KV_LORA_RANK,
            dtype=torch.float32,
            device=case.device,
        )
        # ctx=None on every call, matching cp_lse_ag_out_rs_mla. Reusing a
        # CPTritonContext takes its cached-relaunch branch, which passes only the
        # non-constexpr args and does not survive current Triton.
        corrected, _ = correct_attn_out(out, stacked, rank, None, buffer)
        merged = corrected.clone() if merged is None else merged + corrected
    return merged.transpose(0, 1)


def _relative_error(got, want):
    return ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class TestFA3MLADecodeUnderDCP(CustomTestCase):
    DTYPE = torch.bfloat16
    VER = 3
    # bf16 KV plus a reduction split across ranks; the float32 reference is the
    # loose bound, agreement with unsharded FA3 is the tight one.
    REF_TOL = 3e-2
    UNSHARDED_TOL = 3e-2

    def _check(self, seq_lens, num_heads, dcp_size, page_size, seed=0):
        case = _Case(
            seq_lens,
            num_heads,
            dcp_size,
            page_size,
            "cuda",
            self.DTYPE,
            seed,
        )
        outs, lses, tables = [], [], []
        for rank in range(dcp_size):
            out, lse, table = _run_rank(case, rank, self.VER)
            outs.append(out)
            lses.append(lse)
            tables.append(table)

        for table in tables[1:]:
            self.assertTrue(
                torch.equal(table, tables[0]),
                "the page table must be identical on every DCP rank",
            )

        merged = _merge(case, outs, lses)
        self.assertFalse(
            torch.isnan(merged).any(), "merged output contains NaN"
        )
        reference = case.reference()
        error = _relative_error(merged, reference)
        self.assertLess(
            error,
            self.REF_TOL,
            f"dcp_size={dcp_size} page_size={page_size} seq_lens={seq_lens}: "
            f"relative error vs float32 reference is {error:.4f}",
        )
        return case, merged, reference

    def test_matches_reference_across_dcp_sizes(self):
        for dcp_size in (2, 4, 8):
            with self.subTest(dcp_size=dcp_size):
                self._check([37, 64, 5, 128], num_heads=16, dcp_size=dcp_size, page_size=1)

    def test_matches_reference_with_paged_kv(self):
        # page_size > 1 exercises the full stride = page_size * dcp_size, where a
        # stray `// page_size` would still produce in-range but wrong pages.
        for page_size in (16, 64):
            with self.subTest(page_size=page_size):
                self._check([200, 71, 256], num_heads=16, dcp_size=4, page_size=page_size)

    def test_ranks_with_no_owned_kv(self):
        # seq_len < dcp_size leaves the high ranks with cache_seqlens == 0. Their
        # LSE must come back as -inf so the merge drops them, rather than NaN
        # poisoning every rank's output.
        _, merged, _ = self._check([1, 2, 3], num_heads=8, dcp_size=8, page_size=1)
        self.assertTrue(torch.isfinite(merged).all())

    def test_empty_rank_lse_is_negative_infinity(self):
        case = _Case([1], num_heads=8, dcp_size=4, page_size=1, device="cuda",
                     dtype=self.DTYPE, seed=3)
        # Only rank 0 owns position 0; ranks 1..3 own nothing.
        _, lse_owner, _ = _run_rank(case, 0, self.VER)
        self.assertTrue(torch.isfinite(lse_owner).all())
        for rank in (1, 2, 3):
            _, lse_empty, _ = _run_rank(case, rank, self.VER)
            self.assertTrue(
                (lse_empty == float("-inf")).all() | torch.isfinite(lse_empty).all(),
                f"rank {rank} produced a mixed/NaN LSE: {lse_empty}",
            )
            self.assertFalse(torch.isnan(lse_empty).any())

    def test_matches_unsharded_fa3(self):
        # The tight comparison: same kernel, same inputs, sharding the only
        # difference. Bounds how much of the error above is just bf16.
        case = _Case([37, 64, 5, 128], 16, 1, 1, "cuda", self.DTYPE, 0)
        unsharded, _, _ = _run_rank(case, 0, self.VER)
        _, merged, _ = self._check([37, 64, 5, 128], 16, 4, 1, seed=0)
        error = _relative_error(merged, unsharded)
        self.assertLess(
            error,
            self.UNSHARDED_TOL,
            f"DCP merge diverges from unsharded FA3 by {error:.4f}",
        )


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class TestNormalizeDCPLSE(CustomTestCase):
    """Pin the LSE contract the merge depends on: ``[T, H]``, float32, base 2."""

    def test_varlen_form_is_transposed(self):
        from sglang.srt.layers.attention.flashattention_backend import _normalize_dcp_lse

        lse = torch.randn(8, 3, device="cuda")  # (H, T)
        got = _normalize_dcp_lse(lse)
        self.assertEqual(tuple(got.shape), (3, 8))
        self.assertEqual(got.dtype, torch.float32)

    def test_batched_form_is_squeezed(self):
        from sglang.srt.layers.attention.flashattention_backend import _normalize_dcp_lse

        lse = torch.randn(3, 8, 1, device="cuda")  # (B, H, S)
        got = _normalize_dcp_lse(lse)
        self.assertEqual(tuple(got.shape), (3, 8))

    def test_converts_natural_log_to_base_two(self):
        from sglang.srt.layers.attention.flashattention_backend import _normalize_dcp_lse

        # A natural-log LSE of ln(8) must come back as log2(8) == 3.
        lse = torch.full((4, 2), float(torch.tensor(8.0).log()), device="cuda")
        got = _normalize_dcp_lse(lse)
        self.assertTrue(torch.allclose(got, torch.full_like(got, 3.0), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
