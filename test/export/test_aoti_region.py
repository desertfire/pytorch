# Owner(s): ["oncall: export"]

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch._export._aoti_region import (
    _AOTI_REGION_SPEC_ATTR,
    _discover_aoti_regions,
    AOTIRegionSpec,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestAOTIRegion(TestCase):
    def test_discovers_nested_regions_in_module_order(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return x, x

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.second = torch.nn.Sequential(torch.nn.Identity(), Region())
                self.first = Region()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.first(self.second(x)[0])[0]

        model = Model()
        regions = _discover_aoti_regions(model)

        self.assertEqual(
            [region.module_fqn for region in regions], ["second.1", "first"]
        )
        self.assertIs(regions[0].module, model.second[1])
        self.assertEqual(
            regions[0].signature.return_annotation,
            tuple[torch.Tensor, torch.Tensor],
        )
        self.assertEqual(regions[0].spec, AOTIRegionSpec())

    def test_decorator_supports_parentheses_and_preserves_function(self) -> None:
        def forward(self: Any, x: torch.Tensor) -> torch.Tensor:
            return x

        marked = torch._export.aoti_region()(forward)

        self.assertIs(marked, forward)
        self.assertEqual(getattr(marked, _AOTI_REGION_SPEC_ATTR), AOTIRegionSpec())
        with self.assertRaisesRegex(dataclasses.FrozenInstanceError, "cannot assign"):
            getattr(marked, _AOTI_REGION_SPEC_ATTR).schema_version = 2

    def test_receiver_name_is_not_significant(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(module, x: torch.Tensor) -> torch.Tensor:
                return x

        regions = _discover_aoti_regions(torch.nn.Sequential(Region()))

        self.assertEqual([region.module_fqn for region in regions], ["0"])

    def test_rejects_duplicate_and_non_function_decoration(self) -> None:
        @torch._export.aoti_region
        def forward(self: Any, x: torch.Tensor) -> torch.Tensor:
            return x

        with self.assertRaisesRegex(ValueError, "already marked"):
            torch._export.aoti_region(forward)
        with self.assertRaisesRegex(TypeError, "must decorate a Python method"):
            torch._export.aoti_region(torch.nn.Identity())

    def test_rejects_root_and_non_forward_regions(self) -> None:
        class RootRegion(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        with self.assertRaisesRegex(ValueError, "root module cannot"):
            _discover_aoti_regions(RootRegion())

        class HelperRegion(torch.nn.Module):
            @torch._export.aoti_region
            def helper(self, x: torch.Tensor) -> torch.Tensor:
                return x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.helper(x)

        model = torch.nn.Sequential(HelperRegion())
        with self.assertRaisesRegex(ValueError, "must be named 'forward'.*'helper'"):
            _discover_aoti_regions(model)

    @parametrize("missing", ("parameter", "return"))
    def test_rejects_missing_annotations(self, missing: str) -> None:
        def forward(receiver, x):
            return x

        if missing == "return":
            forward.__annotations__["x"] = torch.Tensor
            error = "missing a return"
        else:
            forward.__annotations__["return"] = torch.Tensor
            error = "parameter 'x' is missing"
        torch._export.aoti_region(forward)
        region_type = type("Region", (torch.nn.Module,), {"forward": forward})

        with self.assertRaisesRegex(TypeError, error):
            _discover_aoti_regions(torch.nn.Sequential(region_type()))

    @parametrize("annotation", ("torch.DoesNotExist", "tuple["))
    def test_normalizes_annotation_resolution_errors(self, annotation: str) -> None:
        def forward(receiver, x):
            return x

        forward.__annotations__ = {
            "x": annotation,
            "return": torch.Tensor,
        }
        torch._export.aoti_region(forward)
        region_type = type("Region", (torch.nn.Module,), {"forward": forward})

        with self.assertRaisesRegex(
            TypeError, "Could not resolve annotations for AOTI region '0.forward'"
        ):
            _discover_aoti_regions(torch.nn.Sequential(region_type()))

    def test_rejects_unsupported_abi_type(self) -> None:
        class BadType(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
                return x[0]

        with self.assertRaisesRegex(TypeError, "fixed tuple.*list"):
            _discover_aoti_regions(torch.nn.Sequential(BadType()))

    def test_rejects_empty_tuple(self) -> None:
        class EmptyTuple(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> tuple[()]:
                return ()

        with self.assertRaisesRegex(TypeError, "nonempty fixed tuple"):
            _discover_aoti_regions(torch.nn.Sequential(EmptyTuple()))

    def test_rejects_variadic_parameters(self) -> None:
        class Variadic(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, *args: torch.Tensor) -> torch.Tensor:
                return args[0]

        with self.assertRaisesRegex(TypeError, "variadic"):
            _discover_aoti_regions(torch.nn.Sequential(Variadic()))

    def test_rejects_default_parameter_values(self) -> None:
        default = torch.empty(0)

        class DefaultValue(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor = default) -> torch.Tensor:
                return x

        with self.assertRaisesRegex(TypeError, "default parameter"):
            _discover_aoti_regions(torch.nn.Sequential(DefaultValue()))

    def test_rejects_required_keyword_only_parameters(self) -> None:
        class KeywordOnly(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, *, x: torch.Tensor) -> torch.Tensor:
                return x

        with self.assertRaisesRegex(TypeError, "required keyword-only parameter 'x'"):
            _discover_aoti_regions(torch.nn.Sequential(KeywordOnly()))

    @parametrize("descriptor", (classmethod, staticmethod))
    def test_rejects_descriptor_decorator_orderings(self, descriptor: type) -> None:
        def marked_inside(receiver, x: torch.Tensor) -> torch.Tensor:
            return x

        marked_inside = torch._export.aoti_region(marked_inside)
        region_type = type(
            "Region",
            (torch.nn.Module,),
            {"forward": descriptor(marked_inside)},
        )
        with self.assertRaisesRegex(TypeError, "must be an instance method"):
            _discover_aoti_regions(torch.nn.Sequential(region_type()))

        def marked_outside(receiver, x: torch.Tensor) -> torch.Tensor:
            return x

        with self.assertRaisesRegex(TypeError, "must decorate a Python method"):
            torch._export.aoti_region(descriptor(marked_outside))

    def test_rejects_aliased_and_overlapping_regions(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        region = Region()
        aliased = torch.nn.ModuleDict({"first": region, "second": region})
        with self.assertRaisesRegex(ValueError, "aliased by both 'first' and 'second'"):
            _discover_aoti_regions(aliased)

        class ParentRegion(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = Region()

            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.child(x)

        with self.assertRaisesRegex(ValueError, "regions '0' and '0.child' overlap"):
            _discover_aoti_regions(torch.nn.Sequential(ParentRegion()))


instantiate_parametrized_tests(TestAOTIRegion)


if __name__ == "__main__":
    run_tests()
