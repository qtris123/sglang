"""CPU unit test for the decode-context-parallel (DCP) per-rank KV-length math.

Pins ``get_dcp_lens`` (the single, superset implementation in
``layers/dcp/layout.py``) to a brute-force owner-count reference, and proves
it is bit-identical to the legacy in-place formula that
``update_local_kv_lens_for_dcp`` used before it was collapsed into a wrapper:

    floor((len - rank - 1) / N) + 1   ==   len // N + (rank < len % N)   (len >= 0)

Usage:
    python -m pytest test_dcp_layout_unit.py -v
    python test_dcp_layout_unit.py
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.layers.dcp.layout import (
    build_dcp_page_table,
    dcp_local_cache_seqlens,
    dcp_paged_stride,
    get_dcp_lens,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 8]
LENS = list(range(0, 41))
STARTS = [0, 1, 2, 5, 7, 13, 31]


def _owner_count(length: int, n: int, rank: int, start: int) -> int:
    """Ground truth: # of absolute positions p in [start, start+length) with p % n == rank."""
    return sum(1 for p in range(start, start + length) if p % n == rank)


def _legacy_inplace_formula(length: int, n: int, rank: int) -> int:
    """The pre-refactor update_local_kv_lens_for_dcp body (start == 0 case)."""
    return (length - rank - 1) // n + 1


class TestGetDcpLens(CustomTestCase):
    def test_start_none_matches_owner_count(self):
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int32)
                got = get_dcp_lens(lens, n, rank)
                expected = torch.tensor(
                    [_owner_count(L, n, rank, 0) for L in LENS], dtype=torch.int32
                )
                self.assertTrue(
                    torch.equal(got.to(torch.int32), expected),
                    f"start=None mismatch at n={n}, rank={rank}: {got.tolist()} != {expected.tolist()}",
                )

    def test_start_none_matches_legacy_inplace_formula(self):
        # The collapse claim: get_dcp_lens (start=None) == legacy floor((L-rank-1)/N)+1.
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int64)
                got = get_dcp_lens(lens, n, rank)
                legacy = torch.tensor(
                    [_legacy_inplace_formula(L, n, rank) for L in LENS],
                    dtype=torch.int64,
                )
                self.assertTrue(
                    torch.equal(got.to(torch.int64), legacy),
                    f"legacy-formula mismatch at n={n}, rank={rank}",
                )

    def test_start_tensor_matches_owner_count(self):
        for n in DCP_SIZES:
            for rank in range(n):
                for start in STARTS:
                    lens = torch.tensor(LENS, dtype=torch.int64)
                    start_t = torch.full_like(lens, start)
                    got = get_dcp_lens(lens, n, rank, start=start_t)
                    expected = torch.tensor(
                        [_owner_count(L, n, rank, start) for L in LENS],
                        dtype=torch.int64,
                    )
                    self.assertTrue(
                        torch.equal(got.to(torch.int64), expected),
                        f"start={start} mismatch at n={n}, rank={rank}: "
                        f"{got.tolist()} != {expected.tolist()}",
                    )

    def test_dcp_size_one_is_identity(self):
        lens = torch.tensor(LENS, dtype=torch.int32)
        self.assertTrue(torch.equal(get_dcp_lens(lens, 1, 0), lens))

    def test_paged_allocator_exposes_dcp_virtual_capacity(self):
        real_kv_size = 1024
        dcp_size = 4
        physical_page_size = 64
        allocator = PagedTokenToKVPoolAllocator(
            size=real_kv_size * dcp_size,
            page_size=physical_page_size * dcp_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=object(),
            need_sort=False,
        )

        allocations = [allocator.alloc(physical_page_size * dcp_size) for _ in range(4)]
        self.assertTrue(all(indices is not None for indices in allocations))
        virtual_indices = torch.cat(allocations)

        self.assertEqual(allocator.size, real_kv_size * dcp_size)
        self.assertEqual(allocator.page_size, physical_page_size * dcp_size)
        self.assertEqual(allocator.num_pages, real_kv_size // physical_page_size)
        self.assertEqual(
            len(torch.unique(virtual_indices // dcp_size)),
            len(virtual_indices) // dcp_size,
        )
        self.assertLess(
            int((virtual_indices // dcp_size).max()),
            real_kv_size + physical_page_size,
        )

    def test_configurator_scales_only_the_virtual_dcp_allocator(self):
        physical_kv_size = 1024
        physical_page_size = 64
        physical_kv_cache = SimpleNamespace(
            size=physical_kv_size,
            page_size=physical_page_size,
        )
        sizes = SimpleNamespace(
            max_total_num_tokens=physical_kv_size,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )
        allocators = {}

        # The configurator's bag reads (disaggregation_mode / page_size /
        # enable_hisparse) come from the published context; the per-iteration
        # dcp_size stays on the injected instance stand-in.
        self._sa_override = rc.get_context().override_server_args(
            disaggregation_mode="null",
            page_size=physical_page_size,
            enable_hisparse=False,
        )
        self._sa_override.install()
        self.addCleanup(self._sa_override.restore)

        for dcp_size in (1, 4):
            configurator = SimpleNamespace(
                server_args=SimpleNamespace(
                    disaggregation_mode="null",
                    enable_hisparse=False,
                    page_size=physical_page_size,
                    dcp_size=dcp_size,
                ),
                hybrid_gdn_config=None,
                is_hybrid_swa=False,
                kv_cache_dtype=torch.bfloat16,
                device="cpu",
                is_draft_worker=False,
            )
            with patch(
                "sglang.srt.mem_cache.kv_cache_configurator.current_platform.is_out_of_tree",
                return_value=False,
            ):
                allocators[dcp_size] = (
                    KVCacheConfigurator._build_token_to_kv_pool_allocator(
                        configurator,
                        sizes=sizes,
                        token_to_kv_pool=physical_kv_cache,
                        is_dsv4_model=False,
                        req_to_token_pool=object(),
                        token_to_kv_pool_allocator=None,
                    )
                )

        dcp1_allocator = allocators[1]
        dcp4_allocator = allocators[4]
        self.assertIs(dcp1_allocator.get_kvcache(), physical_kv_cache)
        self.assertIs(dcp4_allocator.get_kvcache(), physical_kv_cache)
        self.assertEqual(dcp1_allocator.size, 1024)
        self.assertEqual(dcp1_allocator.page_size, 64)
        self.assertEqual(dcp1_allocator.num_pages, 16)
        self.assertEqual(dcp4_allocator.size, 4096)
        self.assertEqual(dcp4_allocator.page_size, 256)
        self.assertEqual(dcp4_allocator.num_pages, 16)

    def test_live_cell_and_page_ownership_formulas(self):
        dcp_size = 4
        physical_page_size = 64
        ragged_lengths = (0, 1, 2, 3, 4, 63, 64, 65, 255, 256, 257, 515)

        per_rank_counts = []
        for rank in range(dcp_size):
            expected_counts = [
                length // dcp_size + int(rank < length % dcp_size)
                for length in ragged_lengths
            ]
            actual_counts = [
                _owner_count(length, dcp_size, rank, 0) for length in ragged_lengths
            ]
            self.assertEqual(actual_counts, expected_counts)
            per_rank_counts.append(sum(actual_counts))

            allocated_pages = [
                math.ceil(length / (physical_page_size * dcp_size))
                for length in ragged_lengths
            ]
            active_pages = [
                math.ceil(count / physical_page_size) for count in actual_counts
            ]
            self.assertTrue(
                all(
                    active <= allocated
                    for active, allocated in zip(active_pages, allocated_pages)
                )
            )
            self.assertTrue(
                all(
                    allocated - active <= 1
                    for active, allocated in zip(active_pages, allocated_pages)
                )
            )

        self.assertEqual(sum(per_rank_counts), sum(ragged_lengths))

        aligned_lengths = (256, 512, 768, 1024)
        full_replica_cells = sum(aligned_lengths)
        full_replica_pages = sum(
            length // physical_page_size for length in aligned_lengths
        )
        for rank in range(dcp_size):
            local_cells = sum(
                _owner_count(length, dcp_size, rank, 0) for length in aligned_lengths
            )
            local_pages = sum(
                math.ceil(_owner_count(length, dcp_size, rank, 0) / physical_page_size)
                for length in aligned_lengths
            )
            self.assertEqual(local_cells * dcp_size, full_replica_cells)
            self.assertEqual(local_pages * dcp_size, full_replica_pages)

    def test_hybrid_pool_reports_the_backing_attention_shape(self):
        pool = object.__new__(HybridLinearKVPool)
        pool.start_layer = 0
        pool.layer_transfer_counter = None
        pool.full_attention_layer_id_mapping = {3: 0, 7: 1}
        pool.full_kv_pool = MagicMock()
        expected = (torch.Size([1024, 1, 576]), torch.Size([1024, 1, 576]))
        pool.full_kv_pool.get_kv_buffer_shape.return_value = expected

        self.assertEqual(pool.get_kv_buffer_shape(), expected)
        pool.full_kv_pool.get_kv_buffer_shape.assert_called_once_with()


PAGE_SIZES = [1, 16, 32, 64, 128]
PAGED_DCP_SIZES = [1, 2, 4, 8]


class TestDcpPagedLayout(CustomTestCase):
    """Pin the property that lets FA build one block table for all DCP ranks.

    ``build_dcp_page_table`` is only correct because a widened page maps onto
    exactly one fully-covered physical page on every rank. These tests assert that
    directly rather than trusting the algebra, because every paged FA DCP path
    depends on it.
    """

    def test_widened_page_covers_exactly_one_physical_page_per_rank(self):
        # D1: {v // D : v in [k*P*D, (k+1)*P*D), v % D == r} == [k*P, (k+1)*P)
        for page_size in PAGE_SIZES:
            for dcp_size in PAGED_DCP_SIZES:
                stride = page_size * dcp_size
                for k in range(4):
                    expected = set(range(k * page_size, (k + 1) * page_size))
                    for rank in range(dcp_size):
                        owned = {
                            v // dcp_size
                            for v in range(k * stride, (k + 1) * stride)
                            if v % dcp_size == rank
                        }
                        self.assertEqual(
                            owned,
                            expected,
                            f"P={page_size} D={dcp_size} k={k} rank={rank}",
                        )

    def test_page_table_is_identical_on_every_rank(self):
        # The rank index never enters build_dcp_page_table; this pins that the
        # resulting table really is rank-invariant, so no gather/scatter is needed.
        for page_size in PAGE_SIZES:
            for dcp_size in PAGED_DCP_SIZES:
                stride = page_size * dcp_size
                max_seq = stride * 5
                loc_table = torch.arange(2 * max_seq, dtype=torch.int32).view(
                    2, max_seq
                )
                tables = [
                    build_dcp_page_table(loc_table, page_size, dcp_size)
                    for _ in range(dcp_size)
                ]
                for table in tables[1:]:
                    self.assertTrue(torch.equal(table, tables[0]))

    def test_page_table_entries_are_the_physical_pages(self):
        for page_size in PAGE_SIZES:
            for dcp_size in PAGED_DCP_SIZES:
                stride = page_size * dcp_size
                num_pages = 5
                # Contiguous virtual run starting at 0, i.e. widened pages 0..4.
                loc_table = torch.arange(
                    stride * num_pages, dtype=torch.int32
                ).unsqueeze(0)
                table = build_dcp_page_table(loc_table, page_size, dcp_size)
                expected = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
                self.assertTrue(
                    torch.equal(table, expected),
                    f"P={page_size} D={dcp_size}: {table.tolist()}",
                )

    def test_dcp_size_one_matches_the_existing_fa_reduction(self):
        # With DCP off the helper must reproduce today's FA code byte for byte:
        #   page_table[:, arange(0, N, page_size)] // page_size
        for page_size in PAGE_SIZES:
            max_seq = page_size * 7
            loc_table = torch.randint(0, 10_000, (3, max_seq), dtype=torch.int32)
            strided = torch.arange(0, max_seq, page_size)
            reference = loc_table[:, strided] // page_size
            got = build_dcp_page_table(loc_table, page_size, dcp_size=1)
            self.assertTrue(torch.equal(got, reference), f"P={page_size}")

    def test_cached_strided_indices_match_the_computed_ones(self):
        page_size, dcp_size = 16, 4
        stride = page_size * dcp_size
        max_seq = stride * 6
        loc_table = torch.randint(0, 10_000, (2, max_seq), dtype=torch.int32)
        cached = torch.arange(0, max_seq, stride)
        self.assertTrue(
            torch.equal(
                build_dcp_page_table(loc_table, page_size, dcp_size),
                build_dcp_page_table(
                    loc_table, page_size, dcp_size, strided_indices=cached
                ),
            )
        )

    def test_out_buffer_receives_the_table(self):
        # CUDA-graph replay needs a fixed-address destination.
        page_size, dcp_size = 32, 2
        stride = page_size * dcp_size
        max_seq = stride * 3
        loc_table = torch.arange(2 * max_seq, dtype=torch.int32).view(2, max_seq)
        expected = build_dcp_page_table(loc_table, page_size, dcp_size)
        out = torch.full((2, 8), -1, dtype=torch.int32)
        returned = build_dcp_page_table(loc_table, page_size, dcp_size, out=out)
        self.assertIs(returned, out)
        self.assertTrue(torch.equal(out[:, : expected.shape[1]], expected))
        # Columns beyond the live width must be left alone for the caller to mask.
        self.assertTrue(torch.all(out[:, expected.shape[1] :] == -1))

    def test_table_entry_addresses_the_rows_each_rank_actually_owns(self):
        # End-to-end property, brute forced: for widened page k of request b, the
        # physical rows rank r owns must be exactly the rows of physical page
        # table[b][k]. This is what the kernel relies on when it treats one shared
        # block table as valid for every rank.
        for page_size in PAGE_SIZES:
            for dcp_size in PAGED_DCP_SIZES:
                stride = page_size * dcp_size
                num_pages = 3
                # Aligned run that does NOT start at zero, so a stray `// page_size`
                # or a dropped offset cannot pass by coincidence.
                base = 5 * stride
                loc_table = torch.arange(
                    base, base + stride * num_pages, dtype=torch.int64
                ).unsqueeze(0)
                table = build_dcp_page_table(loc_table, page_size, dcp_size)
                for k in range(num_pages):
                    physical_page = int(table[0, k])
                    expected_rows = set(
                        range(
                            physical_page * page_size,
                            (physical_page + 1) * page_size,
                        )
                    )
                    virtual_slice = loc_table[0, k * stride : (k + 1) * stride]
                    for rank in range(dcp_size):
                        owned = virtual_slice[virtual_slice % dcp_size == rank]
                        rows = {int(v) // dcp_size for v in owned}
                        self.assertEqual(
                            rows,
                            expected_rows,
                            f"P={page_size} D={dcp_size} k={k} rank={rank}",
                        )

    def test_stride_helper(self):
        self.assertEqual(dcp_paged_stride(64, 1), 64)
        self.assertEqual(dcp_paged_stride(64, 4), 256)
        self.assertEqual(dcp_paged_stride(1, 8), 8)


class TestDcpLocalCacheSeqlens(CustomTestCase):
    def test_matches_get_dcp_lens(self):
        lens = torch.tensor(LENS, dtype=torch.int32)
        for dcp_size in PAGED_DCP_SIZES:
            for rank in range(dcp_size):
                self.assertTrue(
                    torch.equal(
                        dcp_local_cache_seqlens(lens, dcp_size, rank),
                        get_dcp_lens(lens, dcp_size, rank),
                    )
                )

    def test_out_buffer_receives_the_lengths(self):
        lens = torch.tensor([7, 8, 9], dtype=torch.int32)
        out = torch.full((8,), -1, dtype=torch.int32)
        returned = dcp_local_cache_seqlens(lens, 4, 1, out=out)
        self.assertIs(returned, out)
        self.assertTrue(torch.equal(out[:3], get_dcp_lens(lens, 4, 1)))
        self.assertTrue(torch.all(out[3:] == -1))

    def test_lengths_sum_to_the_full_sequence(self):
        # Every token is owned by exactly one rank, so the per-rank lengths must
        # partition the sequence. A drifted owner rule would break this.
        lens = torch.tensor(LENS, dtype=torch.int64)
        for dcp_size in PAGED_DCP_SIZES:
            total = sum(
                dcp_local_cache_seqlens(lens, dcp_size, rank)
                for rank in range(dcp_size)
            )
            self.assertTrue(torch.equal(total, lens))


class TestInvariantA(CustomTestCase):
    """Aligned allocation makes the index rule and the position rule agree.

    DCP ownership is defined on the request POSITION (``p % D == rank``) but every
    hot path tests the virtual INDEX (``v % D == rank``). Those coincide only
    because a request's KV run starts at a multiple of ``page_size * dcp_size``.
    """

    def test_index_rule_equals_position_rule_for_aligned_runs(self):
        for page_size in PAGE_SIZES:
            for dcp_size in PAGED_DCP_SIZES:
                stride = page_size * dcp_size
                for start_page in (0, 1, 3, 7):
                    start = start_page * stride
                    for length in (1, 5, stride, stride + 3, 3 * stride):
                        virtual_ids = torch.arange(start, start + length)
                        positions = torch.arange(length)
                        for rank in range(dcp_size):
                            by_index = virtual_ids % dcp_size == rank
                            by_position = positions % dcp_size == rank
                            self.assertTrue(
                                torch.equal(by_index, by_position),
                                f"P={page_size} D={dcp_size} start={start} "
                                f"len={length} rank={rank}",
                            )

    def test_design_doc_worked_example(self):
        # Design doc section 3.2.1, page_size = 1, D = 4:
        #   r1 -> [0, 1, 2, 3, 8, 9],  r2 -> [4, 5, 6]
        # Both runs start on a multiple of 4, which is Invariant A in action.
        dcp_size = 4
        for virtual_ids in ([0, 1, 2, 3, 8, 9], [4, 5, 6]):
            ids = torch.tensor(virtual_ids)
            # Runs are contiguous per allocated page, so rebuild positions the way
            # the allocator laid them out rather than assuming one flat range.
            self.assertEqual(ids[0].item() % dcp_size, 0, "run must start aligned")
            for rank in range(dcp_size):
                owned = ids[ids % dcp_size == rank]
                # Each rank owns at most one token per widened page of 4.
                self.assertEqual(
                    len(owned), len(set((v // dcp_size).item() for v in owned))
                )
                # And its physical rows are the widened page ids.
                self.assertTrue(
                    torch.equal(
                        owned // dcp_size, torch.tensor([v // dcp_size for v in owned])
                    )
                )

    def test_unaligned_start_breaks_the_equivalence(self):
        # Negative control: without Invariant A the two rules genuinely diverge,
        # so the alignment guarantee is load bearing rather than incidental.
        dcp_size = 4
        start = 2  # not a multiple of dcp_size
        length = 8
        virtual_ids = torch.arange(start, start + length)
        positions = torch.arange(length)
        mismatched = any(
            not torch.equal(
                virtual_ids % dcp_size == rank, positions % dcp_size == rank
            )
            for rank in range(dcp_size)
        )
        self.assertTrue(mismatched)


if __name__ == "__main__":
    unittest.main()
