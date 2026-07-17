# Owner(s): ["oncall: export"]

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import Mock, patch

import torch
from torch._export._aoti_region import (
    _AOTI_REGION_SPEC_ATTR,
    _AOTIRegionStub,
    _capture_aoti_regions,
    _create_aoti_region_stub,
    _discover_aoti_regions,
    _export_aoti_regions,
    _lower_aoti_region_stub,
    _render_aoti_region_stub_source,
    AOTIRegionSpec,
)
from torch.export.graph_signature import InputKind, OutputKind
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

    def test_captures_nested_regions_and_kwargs(self) -> None:
        class AddRegion(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        class MulRegion(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = torch.nn.ModuleDict({"add": AddRegion()})
                self.mul = MulRegion()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return self.mul(self.nested["add"](y=y, x=x))

        model = Model()
        x = torch.randn(4)
        y = torch.randn(4)

        captures = _capture_aoti_regions(model, (x,), {"y": y})

        self.assertEqual(
            [capture.region.module_fqn for capture in captures],
            ["nested.add", "mul"],
        )
        self.assertEqual(captures[0].invocations[0].args, ())
        self.assertEqual(captures[0].invocations[0].kwargs, (("x", x), ("y", y)))
        self.assertEqual(captures[0].invocations[0].output, x + y)
        self.assertEqual(captures[1].invocations[0].args, (x + y,))
        self.assertEqual(captures[1].invocations[0].kwargs, ())

    def test_captures_reentrant_invocations_in_start_order(self) -> None:
        class RecursiveRegion(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.numel() == 1:
                    return x
                return self(x[:-1]) + x[-1]

        region = RecursiveRegion()
        model = torch.nn.Sequential(region)
        x = torch.arange(1, 4)

        captures = _capture_aoti_regions(model, (x,))

        invocations = captures[0].invocations
        self.assertEqual([call.args[0].numel() for call in invocations], [3, 2, 1])
        self.assertEqual(
            [call.output for call in invocations], [x.sum(), x[:2].sum(), x[:1]]
        )

    def test_discards_failed_reentrant_invocation(self) -> None:
        class RecursiveRegion(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.numel() == 1:
                    raise RuntimeError("inner failure")
                try:
                    self(x[:1])
                except RuntimeError:
                    pass
                return x + 1

        region = RecursiveRegion()
        x = torch.arange(3)

        captures = _capture_aoti_regions(torch.nn.Sequential(region), (x,))

        self.assertEqual(len(captures[0].invocations), 1)
        self.assertIs(captures[0].invocations[0].args[0], x)
        self.assertEqual(captures[0].invocations[0].output, x + 1)
        self.assertEqual(region._forward_pre_hooks, {})
        self.assertEqual(region._forward_hooks, {})

    def test_rejects_unexercised_conditional_region(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used = Region()
                self.skipped = Region()

            def forward(self, x: torch.Tensor, use_skipped: bool) -> torch.Tensor:
                if use_skipped:
                    return self.skipped(x)
                return self.used(x)

        model = Model()

        with self.assertRaisesRegex(
            ValueError, "AOTI regions 'skipped' did not complete an invocation"
        ):
            _capture_aoti_regions(model, (torch.randn(2), False))

        self.assertEqual(model.used._forward_pre_hooks, {})
        self.assertEqual(model.used._forward_hooks, {})
        self.assertEqual(model.skipped._forward_pre_hooks, {})
        self.assertEqual(model.skipped._forward_hooks, {})

    @parametrize("hook_kind", ("pre", "forward"))
    def test_rejects_existing_region_hooks(self, hook_kind: str) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        region = Region()
        if hook_kind == "pre":
            handle = region.register_forward_pre_hook(lambda *args: None)
            error = "existing forward pre-hooks"
        else:
            handle = region.register_forward_hook(lambda *args: None)
            error = "existing forward hooks"

        try:
            with self.assertRaisesRegex(ValueError, error):
                _capture_aoti_regions(torch.nn.Sequential(region), (torch.randn(2),))
            self.assertEqual(len(region._forward_pre_hooks), hook_kind == "pre")
            self.assertEqual(len(region._forward_hooks), hook_kind == "forward")
        finally:
            handle.remove()

    @parametrize("hook_kind", ("pre", "forward"))
    def test_rejects_existing_global_hooks(self, hook_kind: str) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        module_hooks = torch.nn.modules.module
        if hook_kind == "pre":
            handle = module_hooks.register_module_forward_pre_hook(lambda *args: None)
            registry = module_hooks._global_forward_pre_hooks
            error = "Global nn.Module forward pre-hooks are unsupported"
        else:
            handle = module_hooks.register_module_forward_hook(lambda *args: None)
            registry = module_hooks._global_forward_hooks
            error = "Global nn.Module forward hooks are unsupported"

        try:
            with self.assertRaisesRegex(ValueError, error):
                _capture_aoti_regions(torch.nn.Sequential(Region()), (torch.randn(2),))
            self.assertIn(handle.id, registry)
        finally:
            handle.remove()

    def test_removes_capture_hooks_when_calibration_raises(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("calibration failed")

        region = Region()

        with self.assertRaisesRegex(RuntimeError, "calibration failed"):
            _capture_aoti_regions(torch.nn.Sequential(region), (torch.randn(2),))

        self.assertEqual(region._forward_pre_hooks, {})
        self.assertEqual(region._forward_hooks, {})

    def test_capture_preserves_mode_and_runs_without_grad(self) -> None:
        class Region(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2))
                self.register_buffer("offset", torch.arange(2))
                self.grad_enabled = True
                self.observed_training = False

            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.grad_enabled = torch.is_grad_enabled()
                self.observed_training = self.training
                return x * self.weight + self.offset

        region = Region()
        model = torch.nn.Sequential(region)
        model.train()
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        buffer_ids = tuple(id(buffer) for buffer in model.buffers())
        state = {name: value.clone() for name, value in model.state_dict().items()}

        captures = _capture_aoti_regions(model, (torch.randn(2, requires_grad=True),))

        self.assertTrue(model.training)
        self.assertTrue(region.observed_training)
        self.assertFalse(region.grad_enabled)
        self.assertFalse(captures[0].invocations[0].output.requires_grad)
        self.assertEqual(
            tuple(id(parameter) for parameter in model.parameters()), parameter_ids
        )
        self.assertEqual(tuple(id(buffer) for buffer in model.buffers()), buffer_ids)
        self.assertEqual(model.state_dict(), state)
        self.assertEqual(region._forward_pre_hooks, {})
        self.assertEqual(region._forward_hooks, {})
        with self.assertRaisesRegex(dataclasses.FrozenInstanceError, "cannot assign"):
            captures[0].invocations = ()
        with self.assertRaisesRegex(dataclasses.FrozenInstanceError, "cannot assign"):
            captures[0].invocations[0].args = ()

    def test_exports_with_normalized_positional_arguments(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = Region()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return self.region(y=y, x=x)

        x = torch.randn(3)
        y = torch.randn(3)
        exports = _export_aoti_regions(_capture_aoti_regions(Model(), (x, y)))

        self.assertEqual(len(exports), 1)
        self.assertEqual(exports[0].example_args, (x, y))
        self.assertEqual(exports[0].exported_program.module()(x, y), x + y)
        with self.assertRaisesRegex(dataclasses.FrozenInstanceError, "cannot assign"):
            exports[0].example_args = ()

    @parametrize("mismatch", ("arity", "type"))
    def test_rejects_input_tuple_value_mismatch(self, mismatch: str) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(
                self, values: tuple[torch.Tensor, torch.Tensor]
            ) -> torch.Tensor:
                return values[0]

        x = torch.randn(2)
        values = (x,) if mismatch == "arity" else [x, x]
        captures = _capture_aoti_regions(torch.nn.Sequential(Region()), (values,))

        with self.assertRaisesRegex(
            TypeError, "'0.forward' invocation 1 parameter 'values'.*tuple with 2"
        ):
            _export_aoti_regions(captures)

    def test_rejects_output_value_mismatch(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return x

        captures = _capture_aoti_regions(
            torch.nn.Sequential(Region()), (torch.randn(2),)
        )

        with self.assertRaisesRegex(
            TypeError, "'0.forward' invocation 1 return.*tuple with 2"
        ):
            _export_aoti_regions(captures)

    @parametrize("alias", ("direct", "view"))
    def test_rejects_output_aliasing_input(self, alias: str) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x if alias == "direct" else x.view_as(x)

        captures = _capture_aoti_regions(
            torch.nn.Sequential(Region()), (torch.randn(2),)
        )

        with self.assertRaisesRegex(
            ValueError, "'0.forward' invocation 1 return aliases user input"
        ):
            _export_aoti_regions(captures)

    def test_validates_multiple_compatible_invocations(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x * y

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = Region()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                first = self.region(x, y=y)
                second = self.region(y=x, x=y)
                return first + second

        x = torch.randn(4)
        y = torch.randn(4)
        exports = _export_aoti_regions(_capture_aoti_regions(Model(), (x, y)))

        self.assertEqual(exports[0].example_args, (x, y))
        self.assertEqual(exports[0].exported_program.module()(y, x), y * x)

    def test_rejects_incompatible_additional_shapes(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = Region()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                first = self.region(x)
                self.region(x[:2])
                return first

        captures = _capture_aoti_regions(Model(), (torch.randn(4),))

        with self.assertRaisesRegex(
            ValueError, "'region.forward' invocation 2 parameter 'x' has shape"
        ):
            _export_aoti_regions(captures)

    def test_rejects_additional_invocation_dtype_mismatch(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = Region()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                first = self.region(x)
                self.region(x.to(torch.float64))
                return first

        captures = _capture_aoti_regions(Model(), (torch.randn(4),))

        with self.assertRaisesRegex(
            ValueError,
            "'region.forward' invocation 2 parameter 'x' has dtype torch.float64",
        ):
            _export_aoti_regions(captures)

    def test_rejects_additional_invocation_requires_grad_mismatch(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.requires_grad:
                    return x + 1
                return x - 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = Region()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                first = self.region(x)
                self.region(x.detach())
                return first

        x = torch.randn(4, requires_grad=True)
        captures = _capture_aoti_regions(Model(), (x,))

        with self.assertRaisesRegex(
            ValueError,
            "'region.forward' invocation 2 parameter 'x' has requires_grad False",
        ):
            _export_aoti_regions(captures)

    @parametrize("mismatch", ("stride", "storage_offset"))
    def test_rejects_additional_invocation_view_metadata_mismatch(
        self, mismatch: str
    ) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = Region()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if mismatch == "stride":
                    first_input = x
                    second_input = x.as_strided(x.shape, (1, 2))
                else:
                    first_input = x[:2]
                    second_input = x[1:3]
                first = self.region(first_input)
                self.region(second_input)
                return first

        shape = (2, 3) if mismatch == "stride" else (3, 3)
        captures = _capture_aoti_regions(Model(), (torch.randn(shape),))

        with self.assertRaisesRegex(
            ValueError,
            f"'region.forward' invocation 2 parameter 'x' has {mismatch}",
        ):
            _export_aoti_regions(captures)

    def test_exports_under_calibration_grad_mode(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if torch.is_grad_enabled():
                    return x.sin()
                return x.cos()

        x = torch.randn(3, requires_grad=True)
        captures = _capture_aoti_regions(torch.nn.Sequential(Region()), (x,))
        exports = _export_aoti_regions(captures)

        self.assertEqual(captures[0].invocations[0].output, x.cos())
        self.assertEqual(exports[0].exported_program.module()(x), x.cos())

    def test_rejects_exported_output_structure_divergence(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if torch.compiler.is_compiling():
                    return (x + 1,)
                return x + 1

        captures = _capture_aoti_regions(
            torch.nn.Sequential(Region()), (torch.randn(2),)
        )

        with self.assertRaisesRegex(
            TypeError, "exported invocation 1 return must be a torch.Tensor"
        ):
            _export_aoti_regions(captures)

    def test_rejects_exported_output_alias_divergence(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if torch.compiler.is_compiling():
                    return x.view_as(x)
                return x.clone()

        captures = _capture_aoti_regions(
            torch.nn.Sequential(Region()), (torch.randn(2),)
        )

        with self.assertRaisesRegex(
            ValueError, "exported invocation 1 return aliases user input"
        ):
            _export_aoti_regions(captures)

    @parametrize("mutation", ("input", "buffer"))
    def test_rejects_exported_mutations(self, mutation: str) -> None:
        class InputMutation(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x.add_(1)
                return x + 0

        class BufferMutation(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("value", torch.ones(2))

            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.value.add_(1)
                return x + self.value

        region = InputMutation() if mutation == "input" else BufferMutation()
        captures = _capture_aoti_regions(torch.nn.Sequential(region), (torch.randn(2),))

        with self.assertRaisesRegex(
            ValueError, "unsupported boundary effects.*MUTATION"
        ):
            _export_aoti_regions(captures)

    @parametrize(
        "effect_kind",
        (InputKind.TOKEN, OutputKind.PARAMETER_MUTATION),
    )
    def test_rejects_other_export_boundary_effects(self, effect_kind: Any) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        captures = _capture_aoti_regions(
            torch.nn.Sequential(Region()), (torch.randn(2),)
        )
        exported_program = Mock()
        if isinstance(effect_kind, InputKind):
            exported_program.graph_signature.input_specs = (Mock(kind=effect_kind),)
            exported_program.graph_signature.output_specs = ()
        else:
            exported_program.graph_signature.input_specs = ()
            exported_program.graph_signature.output_specs = (Mock(kind=effect_kind),)

        with patch("torch.export.export", return_value=exported_program):
            with self.assertRaisesRegex(
                ValueError, f"unsupported boundary effects.*{effect_kind.name}"
            ):
                _export_aoti_regions(captures)

    def test_allows_lifted_custom_objects_and_calls_strict_export(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = Region()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return self.region(y=y, x=x)

        model = Model()
        x = torch.randn(2)
        y = torch.randn(2)
        captures = _capture_aoti_regions(model, (x, y))
        exported_program = Mock()
        exported_program.graph_signature.input_specs = (
            Mock(kind=InputKind.CUSTOM_OBJ),
        )
        exported_program.graph_signature.output_specs = (
            Mock(kind=OutputKind.USER_OUTPUT),
        )
        exported_program.module.return_value = lambda x, y: x + y

        with patch("torch.export.export", return_value=exported_program) as export:
            exports = _export_aoti_regions(captures)

        export.assert_called_once_with(model.region, (x, y), strict=True)
        self.assertIs(exports[0].exported_program, exported_program)

    def test_preserves_strict_export_failure_context(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.sum() > 0:
                    return x + 1
                return x - 1

        captures = _capture_aoti_regions(
            torch.nn.Sequential(Region()), (torch.ones(2),)
        )

        with self.assertRaisesRegex(
            RuntimeError, "Failed to export AOTI region '0.forward'"
        ) as error:
            _export_aoti_regions(captures)
        self.assertIsNotNone(error.exception.__cause__)

    def test_creates_typed_schema_stub(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(
                self,
                /,
                input: torch.Tensor,
                state: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
            ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                return (input + state[0], state[1][0] + state[1][1]), input + 1

        region = Region()
        model = torch.nn.Sequential(region)
        input = torch.randn(2)
        state = (torch.randn(2), (torch.randn(2), torch.randn(2)))
        region_export = _export_aoti_regions(
            _capture_aoti_regions(model, (input, state))
        )[0]

        stub = _create_aoti_region_stub(region_export)

        self.assertIs(model[0], region)
        self.assertEqual(
            stub.source,
            'def forward(self, input: Tensor, state: Tuple[Tensor, Tuple[Tensor, Tensor]]) -> Tuple[Tuple[Tensor, Tensor], Tensor]:\n    assert False, "AOTI region schema stub cannot execute"\n',  # noqa: B950, RUF100
        )
        schema = stub.module.forward.schema
        arguments = schema.arguments[1:]
        self.assertEqual([argument.name for argument in arguments], ["input", "state"])
        self.assertEqual(
            [str(argument.type) for argument in arguments],
            ["Tensor", "Tuple[Tensor, Tuple[Tensor, Tensor]]"],
        )
        self.assertEqual(
            [str(result.type) for result in schema.returns],
            ["Tuple[Tuple[Tensor, Tensor], Tensor]"],
        )
        with self.assertRaisesRegex(
            torch.jit.Error, "AOTI region schema stub cannot execute"
        ):
            stub.module(input, state)
        with self.assertRaisesRegex(dataclasses.FrozenInstanceError, "cannot assign"):
            stub.source = ""

    def test_stub_parameter_can_shadow_runtime_error(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, RuntimeError: torch.Tensor) -> torch.Tensor:
                return RuntimeError + 1

        value = torch.randn(2)
        region_export = _export_aoti_regions(
            _capture_aoti_regions(torch.nn.Sequential(Region()), (value,))
        )[0]

        stub = _create_aoti_region_stub(region_export)

        self.assertEqual(stub.module.forward.schema.arguments[1].name, "RuntimeError")
        with self.assertRaisesRegex(
            torch.jit.Error, "AOTI region schema stub cannot execute"
        ):
            stub.module(value)

    def test_schema_stubs_have_distinct_jit_types(self) -> None:
        class TensorRegion(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value + 1

        class TupleRegion(torch.nn.Module):
            @torch._export.aoti_region
            def forward(
                self, values: tuple[torch.Tensor, torch.Tensor]
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return values[0] + 1, values[1] + 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tensor = TensorRegion()
                self.tuple = TupleRegion()

            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self.tensor(x), self.tuple((x, y))[0]

        exports = _export_aoti_regions(
            _capture_aoti_regions(Model(), (torch.randn(2), torch.randn(2)))
        )
        tensor_stub = _create_aoti_region_stub(exports[0])
        tuple_stub = _create_aoti_region_stub(exports[1])

        self.assertNotEqual(tensor_stub.module._c._type(), tuple_stub.module._c._type())
        tensor_type = tensor_stub.module.forward.schema.arguments[1].type
        tuple_type = tuple_stub.module.forward.schema.arguments[1].type
        self.assertEqual(str(tensor_type), "Tensor")
        self.assertEqual(str(tuple_type), "Tuple[Tensor, Tensor]")

    def test_lowers_stub_with_aoti_package_spec(self) -> None:
        module = Mock()
        stub = _AOTIRegionStub(module, "stub source")
        lowered = Mock()

        with patch("torch._C._jit_to_backend", return_value=lowered) as to_backend:
            result = _lower_aoti_region_stub(stub, "/tmp/region.pt2")

        self.assertIs(result, lowered)
        to_backend.assert_called_once_with(
            "aoti",
            module,
            {
                "forward": {
                    "package_path": "/tmp/region.pt2",
                    "model_name": "model",
                    "device_index": -1,
                }
            },
        )

    @parametrize("package_path", (None, 1, False))
    def test_rejects_non_string_aoti_package_path(self, package_path: Any) -> None:
        stub = _AOTIRegionStub(Mock(), "stub source")

        with patch("torch._C._jit_to_backend") as to_backend:
            with self.assertRaisesRegex(TypeError, "package_path must be a string"):
                _lower_aoti_region_stub(stub, package_path)

        to_backend.assert_not_called()

    def test_rejects_empty_aoti_package_path(self) -> None:
        stub = _AOTIRegionStub(Mock(), "stub source")

        with patch("torch._C._jit_to_backend") as to_backend:
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                _lower_aoti_region_stub(stub, "")

        to_backend.assert_not_called()

    def test_rejects_non_stub_aoti_region_lowering(self) -> None:
        with patch("torch._C._jit_to_backend") as to_backend:
            with self.assertRaisesRegex(TypeError, "Expected an _AOTIRegionStub"):
                _lower_aoti_region_stub(Mock(), "/tmp/region.pt2")

        to_backend.assert_not_called()

    @parametrize("unsupported", ("positional_only", "self_name"))
    def test_rejects_unrepresentable_stub_signature(self, unsupported: str) -> None:
        if unsupported == "positional_only":

            class Region(torch.nn.Module):
                @torch._export.aoti_region
                def forward(self, value: torch.Tensor, /) -> torch.Tensor:
                    return value + 1

            error = "positional-only parameter 'value' cannot be represented"
        else:

            class Region(torch.nn.Module):
                @torch._export.aoti_region
                def forward(module, self: torch.Tensor) -> torch.Tensor:
                    return self + 1

            error = "parameter 'self' cannot be represented"

        region_export = _export_aoti_regions(
            _capture_aoti_regions(torch.nn.Sequential(Region()), (torch.randn(2),))
        )[0]

        with self.assertRaisesRegex(TypeError, error):
            _create_aoti_region_stub(region_export)

    def test_rejects_non_ascii_stub_parameter_name(self) -> None:
        parameter_name = "value_\u03b4"
        function_globals = {"torch": torch}
        function_locals: dict[str, Any] = {}
        exec(
            f"def forward(self, {parameter_name}: torch.Tensor) -> torch.Tensor:\n"
            f"    return {parameter_name} + 1\n",
            function_globals,
            function_locals,
        )
        forward = torch._export.aoti_region(function_locals["forward"])
        region_type = type("Region", (torch.nn.Module,), {"forward": forward})
        region = _discover_aoti_regions(torch.nn.Sequential(region_type()))[0]

        with self.assertRaisesRegex(
            TypeError,
            f"AOTI region '0.forward' parameter '{parameter_name}'.*ASCII",
        ):
            _render_aoti_region_stub_source(region)

    @parametrize("parameter_name", ("Ellipsis", "NoneType"))
    def test_rejects_torchscript_reserved_parameter_name(
        self, parameter_name: str
    ) -> None:
        function_globals = {"torch": torch}
        function_locals: dict[str, Any] = {}
        exec(
            f"def forward(self, {parameter_name}: torch.Tensor) -> torch.Tensor:\n"
            f"    return {parameter_name} + 1\n",
            function_globals,
            function_locals,
        )
        forward = torch._export.aoti_region(function_locals["forward"])
        region_type = type("Region", (torch.nn.Module,), {"forward": forward})
        region = _discover_aoti_regions(torch.nn.Sequential(region_type()))[0]

        with self.assertRaisesRegex(
            TypeError,
            f"AOTI region '0.forward' parameter '{parameter_name}' cannot be represented.*reserved",
        ):
            _render_aoti_region_stub_source(region)


instantiate_parametrized_tests(TestAOTIRegion)


if __name__ == "__main__":
    run_tests()
