"""Where the descriptors live now; Phase 3b moves the rest of the April descriptor tests here."""

import casadi as ca
import pytest

import machina.blocks.descriptor as shim
import machina.model.descriptor as canonical
from machina.model import FunctionDescriptor, SymbolDescriptor

pytestmark = pytest.mark.requires_casadi


def symbol(rows: int = 1, cols: int = 1, name: str = "x") -> ca.MX:
    return ca.MX.sym(name, rows, cols)


class TestTheCanonicalHomeIsMachinaModel:

    def test_the_package_exports_the_names_from_machina_model_descriptor(self):
        assert FunctionDescriptor is canonical.FunctionDescriptor
        assert SymbolDescriptor is canonical.SymbolDescriptor

    def test_the_blocks_shim_re_exports_the_same_objects(self):
        assert shim.FunctionDescriptor is canonical.FunctionDescriptor
        assert shim.SymbolDescriptor is canonical.SymbolDescriptor

    def test_the_shim_re_exports_the_private_valid_semantic_types(self):
        assert shim._VALID_SEMANTIC_TYPES is canonical._VALID_SEMANTIC_TYPES


class TestSemanticTypes:

    def test_matrix_is_a_valid_semantic_type_for_a_two_dimensional_symbol(self):
        descriptor = SymbolDescriptor(
            symbol=symbol(2, 3, "gain"), name="gain", shape=(2, 3), semantic_type="matrix",
        )
        assert descriptor.semantic_type == "matrix"
        assert descriptor.shape == (2, 3)

    def test_an_unknown_semantic_type_is_refused_and_the_message_lists_the_valid_ones(self):
        with pytest.raises(ValueError) as excinfo:
            SymbolDescriptor(
                symbol=symbol(), name="gain", shape=(1, 1), semantic_type="tensor",
            )
        message = str(excinfo.value)
        assert "tensor" in message
        for valid in sorted(canonical._VALID_SEMANTIC_TYPES):
            assert valid in message
