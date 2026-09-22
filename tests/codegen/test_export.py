"""
Export helpers: densify, generate C, hash, describe a layout, merge a manifest.

``dense`` gets the most attention because the bug it prevents is silent. A
structural zero -- a ``ca.diag`` matrix, an unproduced ``SUM`` signal --
makes CasADi's generated C write fewer doubles than the output's published
length, and every later entry lands in the wrong slot with nothing erroring.
"""

import hashlib
import json
import os

import casadi as ca
import pytest

from machina.codegen import (
    canonical_json,
    dense,
    generate_c,
    git_commit,
    layout_block,
    merge,
    schema_hash,
    sha256_of,
)
from machina.model import SignalRegistry

pytestmark = pytest.mark.requires_casadi


def sparse_output():
    x = ca.SX.sym("x")
    return ca.Function("inertia", [x], [ca.diag(ca.vertcat(x, 0.0, 2.0 * x))], ["x"], ["I"])


class TestDense:

    def test_the_problem_is_real(self):
        """The structural zero is what the generated C would skip."""
        f = sparse_output()
        assert f.sparsity_out(0).nnz() < f.numel_out(0)

    def test_dense_makes_every_output_element_structural(self):
        f = dense(sparse_output())
        assert f.sparsity_out(0).nnz() == f.numel_out(0) == 9

    def test_dense_keeps_names_values_and_shapes(self):
        original, densified = sparse_output(), dense(sparse_output(), "inertia_dense")
        assert densified.name() == "inertia_dense"
        assert densified.name_in() == original.name_in()
        assert densified.name_out() == original.name_out()
        assert densified.size_out(0) == (3, 3)
        assert (ca.DM(densified(1.5)) == ca.DM(original(1.5))).is_one()

    def test_the_output_sparsity_the_c_is_generated_from_is_dense(self):
        """CasADi's generated C writes one double per structural nonzero of the
        output sparsity, so a dense sparsity is what makes it write all nine."""
        assert dense(sparse_output()).sparsity_out(0).is_dense()
        assert not sparse_output().sparsity_out(0).is_dense()

    def test_densifying_changes_the_generated_c(self):
        assert generate_c(dense(sparse_output(), "inertia")) != generate_c(sparse_output())


class TestGenerateC:

    def test_it_returns_c_source_for_the_function(self):
        source = generate_c(dense(sparse_output(), "inertia"))
        assert "inertia" in source and "casadi_real" in source

    def test_it_leaves_no_file_in_the_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        generate_c(dense(sparse_output(), "inertia"))
        assert os.listdir(tmp_path) == []

    def test_it_is_byte_stable(self):
        f = dense(sparse_output(), "inertia")
        assert generate_c(f) == generate_c(f)


class TestHashesAndLayouts:

    def test_sha256_of_a_file_hashes_its_bytes(self, tmp_path):
        path = tmp_path / "f.txt"
        payload = b"machina" + bytes([13, 10])   # CRLF, to prove bytes are hashed as-is
        path.write_bytes(payload)
        assert sha256_of(path) == hashlib.sha256(payload).hexdigest()

    def test_layout_block_counts_elements_column_major(self):
        registry = SignalRegistry()
        registry.declare_frame("world")
        registry.declare("pos", 3, "m", frame="world")
        registry.declare("inertia", (3, 3), "kg*m^2")
        block = layout_block(["pos", "inertia"], registry)
        assert block["pos"] == {"offset": 0, "shape": [3, 1], "unit": "m", "frame": "world"}
        assert block["inertia"]["offset"] == 3 and block["inertia"]["shape"] == [3, 3]

    def test_layout_block_reads_scoped_paths_by_their_signal_name(self):
        registry = SignalRegistry()
        registry.declare("mass", 1, "kg")
        block = layout_block(["a/mass", "b/mass"], registry)
        assert [block["a/mass"]["offset"], block["b/mass"]["offset"]] == [0, 1]

    def test_schema_hash_is_a_stable_u64(self):
        block = {"pos": {"offset": 0, "shape": [3, 1]}}
        value = schema_hash(block)
        assert 0 <= value < 2 ** 64
        assert value == schema_hash({"pos": {"shape": [3, 1], "offset": 0}})

    def test_schema_hash_changes_when_the_layout_does(self):
        assert schema_hash({"pos": {"offset": 0}}) != schema_hash({"pos": {"offset": 1}})


class TestManifest:

    def test_merge_adds_rather_than_replaces_in_either_order(self, tmp_path):
        """Stages write their own blocks. Order must not matter; icarus found
        out it did when one stage replaced the file."""
        first, second = tmp_path / "a" / "MANIFEST.json", tmp_path / "b" / "MANIFEST.json"
        merge(first, {"plant": {"n": 1}})
        merge(first, {"params": {"n": 2}})
        merge(second, {"params": {"n": 2}})
        merge(second, {"plant": {"n": 1}})
        assert first.read_bytes() == second.read_bytes()
        assert json.loads(first.read_text()) == {"plant": {"n": 1}, "params": {"n": 2}}

    def test_a_block_of_the_same_name_replaces_the_old_one(self, tmp_path):
        path = tmp_path / "MANIFEST.json"
        merge(path, {"plant": {"n": 1}})
        merge(path, {"plant": {"n": 3}})
        assert json.loads(path.read_text()) == {"plant": {"n": 3}}

    def test_the_manifest_is_canonical_lf_json(self, tmp_path):
        path = tmp_path / "MANIFEST.json"
        merge(path, {"b": 1, "a": 2})
        assert path.read_bytes() == canonical_json({"a": 2, "b": 1}).encode()
        assert b"\r\n" not in path.read_bytes() and path.read_bytes().endswith(b"}\n")

    def test_merging_into_something_that_is_not_an_object_is_refused(self, tmp_path):
        path = tmp_path / "MANIFEST.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="not a JSON object"):
            merge(path, {"plant": {}})

    def test_git_commit_is_none_outside_a_repository(self, tmp_path):
        assert git_commit(tmp_path) is None
        assert git_commit(None) is None
