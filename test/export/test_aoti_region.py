# Owner(s): ["oncall: export"]

from __future__ import annotations

import dataclasses
import io
from typing import Any
from unittest.mock import call, Mock, patch

import torch
from torch._export._aoti_region import (
    _AOTI_REGION_SPEC_ATTR,
    _AOTIRegionExport,
    _AOTIRegionStub,
    _capture_aoti_regions,
    _compile_aoti_region,
    _CompiledAOTIRegion,
    _create_aoti_region_stub,
    _discover_aoti_regions,
    _export_aoti_regions,
    _lower_aoti_region_stub,
    _render_aoti_region_stub_source,
    _substitute_compiled_aoti_regions,
    AOTIRegionSpec,
    compile_aoti_regions,
)
from torch.export.graph_signature import InputKind, OutputKind
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _schema_stub(source: str) -> _AOTIRegionStub:
    class SchemaModule(torch.jit.ScriptModule):
        def __init__(self) -> None:
            super().__init__()
            self.define(source)

    return _AOTIRegionStub(SchemaModule(), source)


class TestAOTIRegion(TestCase):
    def test_compile_aoti_regions_is_public(self) -> None:
        self.assertIs(torch._export.compile_aoti_regions, compile_aoti_regions)

    @parametrize("parameter", ("root", "args", "kwargs"))
    def test_compile_aoti_regions_validates_inputs_before_pipeline(
        self, parameter: str
    ) -> None:
        root: Any = torch.nn.Identity()
        args: Any = ()
        kwargs: Any = None
        if parameter == "root":
            root = object()
            error = "Expected an nn.Module"
        elif parameter == "args":
            args = []
            error = "args must be a tuple"
        else:
            kwargs = []
            error = "kwargs must be a dict or None"

        with (
            patch("torch._export._aoti_region._capture_aoti_regions") as capture,
            patch("torch.jit.script") as script,
            self.assertRaisesRegex(TypeError, error),
        ):
            compile_aoti_regions(root, args, kwargs)

        capture.assert_not_called()
        script.assert_not_called()

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

    @parametrize(
        "source",
        (
            "def forward(self) -> Tensor:\n    assert False\n",
            "def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:\n    assert False\n",
        ),
    )
    def test_lowers_supported_stub_schema_with_aoti_package_spec(
        self, source: str
    ) -> None:
        stub = _schema_stub(source)
        lowered = Mock()

        with patch("torch._C._jit_to_backend", return_value=lowered) as to_backend:
            result = _lower_aoti_region_stub(stub, "/tmp/region.pt2")

        self.assertIs(result, lowered)
        to_backend.assert_called_once_with(
            "aoti",
            stub.module,
            {
                "forward": {
                    "package_path": "/tmp/region.pt2",
                    "model_name": "model",
                    "device_index": -1,
                }
            },
        )

    @parametrize(
        "input_type",
        (
            "Tuple[Tensor, Tensor]",
            "Tuple[Tensor, Tuple[Tensor, Tensor]]",
            "List[Tensor]",
            "int",
        ),
    )
    def test_rejects_unsupported_aoti_backend_input_schema(
        self, input_type: str
    ) -> None:
        stub = _schema_stub(
            f"def forward(self, value: {input_type}) -> Tensor:\n    assert False\n"
        )

        with patch("torch._C._jit_to_backend") as to_backend:
            with self.assertRaisesRegex(
                TypeError, "forward arguments must be Tensor.*argument 'value'"
            ):
                _lower_aoti_region_stub(stub, "/tmp/region.pt2")

        to_backend.assert_not_called()

    @parametrize(
        "output_type",
        (
            "Tuple[()]",
            "Tuple[Tensor, Tuple[Tensor, Tensor]]",
            "Tuple[Tensor, int]",
            "List[Tensor]",
            "int",
        ),
    )
    def test_rejects_unsupported_aoti_backend_output_schema(
        self, output_type: str
    ) -> None:
        stub = _schema_stub(
            f"def forward(self, value: Tensor) -> {output_type}:\n    assert False\n"
        )

        with patch("torch._C._jit_to_backend") as to_backend:
            with self.assertRaisesRegex(
                TypeError, "return must be Tensor or a nonempty flat tuple of Tensor"
            ):
                _lower_aoti_region_stub(stub, "/tmp/region.pt2")

        to_backend.assert_not_called()

    @parametrize("return_count", (0, 2))
    def test_rejects_malformed_aoti_backend_return_arity(
        self, return_count: int
    ) -> None:
        schema = Mock(arguments=[Mock()], returns=[Mock()] * return_count)
        stub = _AOTIRegionStub(Mock(forward=Mock(schema=schema)), "stub source")

        with patch("torch._C._jit_to_backend") as to_backend:
            with self.assertRaisesRegex(
                TypeError, f"exactly one return, but has {return_count}"
            ):
                _lower_aoti_region_stub(stub, "/tmp/region.pt2")

        to_backend.assert_not_called()

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

    def test_compiles_and_lowers_one_aoti_region(self) -> None:
        region = Mock()
        exported_program = Mock()
        region_export = _AOTIRegionExport(region, (), exported_program)
        stub = _schema_stub(
            "def forward(self, value: Tensor) -> Tensor:\n    assert False\n"
        )
        lowered = Mock()
        calls = []

        def validate(schema: torch._C.FunctionSchema) -> None:
            self.assertIs(schema, stub.module.forward.schema)
            calls.append("validate")

        def compile(program: Any) -> str:
            self.assertIs(program, exported_program)
            calls.append("compile")
            return "/tmp/compiled-region.pt2"

        def lower(received_stub: _AOTIRegionStub, package_path: str) -> Any:
            self.assertIs(received_stub, stub)
            self.assertEqual(package_path, "/tmp/compiled-region.pt2")
            calls.append("lower")
            return lowered

        with (
            patch(
                "torch._export._aoti_region._create_aoti_region_stub",
                return_value=stub,
            ),
            patch(
                "torch._export._aoti_region._validate_aoti_region_stub_schema",
                side_effect=validate,
            ),
            patch("torch._inductor.aoti_compile_and_package", side_effect=compile),
            patch(
                "torch._export._aoti_region._lower_aoti_region_stub",
                side_effect=lower,
            ),
        ):
            result = _compile_aoti_region(region_export)

        self.assertEqual(calls, ["validate", "compile", "lower"])
        self.assertIs(result.region, region)
        self.assertEqual(result.package_path, "/tmp/compiled-region.pt2")
        self.assertIs(result.module, lowered)
        with self.assertRaisesRegex(dataclasses.FrozenInstanceError, "cannot assign"):
            result.package_path = ""

    def test_validates_schema_before_aoti_compilation(self) -> None:
        region_export = _AOTIRegionExport(Mock(), (), Mock())
        stub = _schema_stub(
            "def forward(self, values: Tuple[Tensor, Tensor]) -> Tensor:\n"
            "    assert False\n"
        )

        with (
            patch(
                "torch._export._aoti_region._create_aoti_region_stub",
                return_value=stub,
            ),
            patch("torch._inductor.aoti_compile_and_package") as compile,
            patch("torch._export._aoti_region._lower_aoti_region_stub") as lower,
        ):
            with self.assertRaisesRegex(TypeError, "forward arguments must be Tensor"):
                _compile_aoti_region(region_export)

        compile.assert_not_called()
        lower.assert_not_called()

    def test_aoti_compiler_failure_propagates(self) -> None:
        error = RuntimeError("AOTI compilation failed")
        region_export = _AOTIRegionExport(Mock(), (), Mock())
        stub = _schema_stub("def forward(self) -> Tensor:\n    assert False\n")

        with (
            patch(
                "torch._export._aoti_region._create_aoti_region_stub",
                return_value=stub,
            ),
            patch("torch._inductor.aoti_compile_and_package", side_effect=error),
            patch("torch._export._aoti_region._lower_aoti_region_stub") as lower,
        ):
            with self.assertRaisesRegex(RuntimeError, "AOTI compilation failed") as cm:
                _compile_aoti_region(region_export)

        self.assertIs(cm.exception, error)
        lower.assert_not_called()

    @parametrize("package_path", (None, 1, False, ""))
    def test_rejects_invalid_aoti_compiler_package_path(
        self, package_path: Any
    ) -> None:
        region_export = _AOTIRegionExport(Mock(), (), Mock())
        stub = _schema_stub("def forward(self) -> Tensor:\n    assert False\n")
        error = "package path string" if package_path != "" else "empty package path"

        with (
            patch(
                "torch._export._aoti_region._create_aoti_region_stub",
                return_value=stub,
            ),
            patch(
                "torch._inductor.aoti_compile_and_package",
                return_value=package_path,
            ),
            patch("torch._export._aoti_region._lower_aoti_region_stub") as lower,
        ):
            with self.assertRaisesRegex((TypeError, ValueError), error):
                _compile_aoti_region(region_export)

        lower.assert_not_called()

    def test_rejects_non_aoti_region_export_compilation(self) -> None:
        with patch("torch._inductor.aoti_compile_and_package") as compile:
            with self.assertRaisesRegex(TypeError, "Expected an _AOTIRegionExport"):
                _compile_aoti_region(Mock())

        compile.assert_not_called()

    def test_substitutes_compiled_regions_in_a_copy(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = torch.nn.Sequential(Region())
                self.second = Region()
                self.other = torch.nn.Linear(2, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.other(self.second(self.nested(x)))

        model = Model()
        regions = _discover_aoti_regions(model)
        lowered = tuple(torch.jit.script(torch.nn.Identity()) for _ in regions)
        compiled = tuple(
            _CompiledAOTIRegion(region, f"/tmp/region-{index}.pt2", module)
            for index, (region, module) in enumerate(zip(regions, lowered))
        )

        result = _substitute_compiled_aoti_regions(model, compiled)

        self.assertIsNot(result, model)
        for region, module in zip(regions, lowered):
            self.assertIs(model.get_submodule(region.module_fqn), region.module)
            self.assertIs(result.get_submodule(region.module_fqn), module)
        self.assertIsNot(result.nested, model.nested)
        self.assertIsNot(result.other, model.other)
        self.assertEqual(result.other.state_dict(), model.other.state_dict())
        x = torch.randn(2)
        scripted = torch.jit.script(result)
        self.assertEqual(scripted(x), result(x))

    @parametrize("invalid", ("empty", "missing", "stale", "duplicate"))
    def test_rejects_invalid_compiled_region_paths(self, invalid: str) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        model = torch.nn.Sequential(Region())
        region = _discover_aoti_regions(model)[0]
        if invalid == "empty":
            region = dataclasses.replace(region, module_fqn="")
            error = "nonempty module path"
        elif invalid == "missing":
            region = dataclasses.replace(region, module_fqn="missing")
            error = "does not identify a submodule"
        elif invalid == "stale":
            region = dataclasses.replace(region, module=torch.nn.Identity())
            error = "does not match the root submodule"
        else:
            error = "Duplicate compiled AOTI region path '0'"
        compiled = _CompiledAOTIRegion(
            region, "/tmp/region.pt2", torch.jit.script(torch.nn.Identity())
        )
        records = (compiled, compiled) if invalid == "duplicate" else (compiled,)

        with self.assertRaisesRegex(ValueError, error):
            _substitute_compiled_aoti_regions(model, records)

        self.assertIs(model[0], _discover_aoti_regions(model)[0].module)

    def test_rejects_overlapping_compiled_region_paths(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        model = torch.nn.Sequential(torch.nn.Sequential(Region()))
        child = _discover_aoti_regions(model)[0]
        parent = dataclasses.replace(child, module_fqn="0", module=model[0])
        lowered = torch.jit.script(torch.nn.Identity())
        compiled = (
            _CompiledAOTIRegion(parent, "/tmp/parent.pt2", lowered),
            _CompiledAOTIRegion(child, "/tmp/child.pt2", lowered),
        )

        with self.assertRaisesRegex(
            ValueError, "Compiled AOTI region paths '0' and '0.0' overlap"
        ):
            _substitute_compiled_aoti_regions(model, compiled)

        self.assertIs(model.get_submodule("0.0"), child.module)

    def test_reports_module_copy_failure(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        class UncopyableModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = Region()

            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                raise RuntimeError("copy disabled")

        model = UncopyableModel()
        region = _discover_aoti_regions(model)[0]
        compiled = _CompiledAOTIRegion(
            region, "/tmp/region.pt2", torch.jit.script(torch.nn.Identity())
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Failed to copy UncopyableModel.*copy disabled",
        ) as error:
            _substitute_compiled_aoti_regions(model, (compiled,))

        self.assertIsInstance(error.exception.__cause__, RuntimeError)

    def test_rejects_copy_that_shares_region_parent(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        class SharedParentModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = torch.nn.Sequential(Region())

            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                result = type(self)()
                result.nested = self.nested
                return result

        model = SharedParentModel()
        region = _discover_aoti_regions(model)[0]
        compiled = _CompiledAOTIRegion(
            region, "/tmp/region.pt2", torch.jit.script(torch.nn.Identity())
        )

        with self.assertRaisesRegex(
            RuntimeError, "copied parent 'nested' is shared with the original module"
        ):
            _substitute_compiled_aoti_regions(model, (compiled,))

        self.assertIs(model.get_submodule("nested.0"), region.module)

    def test_compiles_aoti_regions_and_scripts_parent_in_order(self) -> None:
        root = torch.nn.Identity()
        args = (Mock(), Mock())
        kwargs = {"scale": Mock()}
        captures = (Mock(), Mock())
        region_exports = (Mock(), Mock())
        compiled_regions = (Mock(), Mock())
        substituted = Mock()
        scripted = Mock()
        events = []

        def capture(
            received_root: Any, received_args: Any, received_kwargs: Any
        ) -> Any:
            events.append("capture")
            return captures

        def export(received_captures: Any) -> Any:
            events.append("export")
            return region_exports

        def compile(region_export: Any) -> Any:
            index = region_exports.index(region_export)
            events.append(f"compile:{index}")
            return compiled_regions[index]

        def substitute(received_root: Any, received_compiled: Any) -> Any:
            events.append("substitute")
            return substituted

        def script(received_module: Any) -> Any:
            events.append("script")
            return scripted

        with (
            patch(
                "torch._export._aoti_region._capture_aoti_regions",
                side_effect=capture,
            ) as capture_mock,
            patch(
                "torch._export._aoti_region._export_aoti_regions",
                side_effect=export,
            ) as export_mock,
            patch(
                "torch._export._aoti_region._compile_aoti_region",
                side_effect=compile,
            ) as compile_mock,
            patch(
                "torch._export._aoti_region._substitute_compiled_aoti_regions",
                side_effect=substitute,
            ) as substitute_mock,
            patch("torch.jit.script", side_effect=script) as script_mock,
        ):
            result = compile_aoti_regions(root, args, kwargs)

        self.assertEqual(
            events,
            ["capture", "export", "compile:0", "compile:1", "substitute", "script"],
        )
        capture_mock.assert_called_once_with(root, args, kwargs)
        export_mock.assert_called_once_with(captures)
        self.assertEqual(
            compile_mock.call_args_list,
            [call(region_exports[0]), call(region_exports[1])],
        )
        substitute_mock.assert_called_once_with(root, compiled_regions)
        script_mock.assert_called_once_with(substituted)
        self.assertIs(result, scripted)

    def test_compiles_exported_region_inside_control_flow_parent(self) -> None:
        class Region(torch.nn.Module):
            @torch._export.aoti_region
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2 + 1

        class CompiledRegion(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2 + 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = Region()
                self.register_buffer("bias", torch.tensor(3.0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.sum() > 0:
                    value = self.region(x)
                else:
                    value = x - 4
                return value + self.bias

        model = Model()
        original_region = model.region
        original_state = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        replacement = torch.jit.script(CompiledRegion())
        received_programs: list[torch.export.ExportedProgram] = []
        received_stubs: list[_AOTIRegionStub] = []

        def compile_package(program: torch.export.ExportedProgram) -> str:
            received_programs.append(program)
            return "/tmp/region.pt2"

        def lower(stub: _AOTIRegionStub, package_path: str) -> Any:
            received_stubs.append(stub)
            self.assertEqual(package_path, "/tmp/region.pt2")
            schema = stub.module.forward.schema
            self.assertEqual(
                [str(argument.type) for argument in schema.arguments[1:]], ["Tensor"]
            )
            self.assertEqual(
                [str(result.type) for result in schema.returns], ["Tensor"]
            )
            return replacement

        x = torch.tensor([1.0, 2.0])
        with (
            patch(
                "torch._inductor.aoti_compile_and_package",
                side_effect=compile_package,
            ) as compile_mock,
            patch(
                "torch._export._aoti_region._lower_aoti_region_stub",
                side_effect=lower,
            ) as lower_mock,
        ):
            result = compile_aoti_regions(model, (x,))

        compile_mock.assert_called_once()
        lower_mock.assert_called_once()
        self.assertEqual(len(received_programs), 1)
        self.assertEqual(len(received_stubs), 1)
        exported_program = received_programs[0]
        self.assertIsInstance(exported_program, torch.export.ExportedProgram)
        self.assertEqual(exported_program.module()(x), x * 2 + 1)
        self.assertIsInstance(result, torch.jit.RecursiveScriptModule)
        alternate = torch.tensor([-1.0, -2.0])
        self.assertEqual(result(x), x * 2 + 4)
        self.assertEqual(result(alternate), alternate - 1)

        self.assertIs(model.region, original_region)
        self.assertEqual(model.state_dict(), original_state)
        self.assertEqual(model(x), x * 2 + 4)
        self.assertEqual(model(alternate), alternate - 1)

        buffer = io.BytesIO()
        torch.jit.save(result, buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        self.assertEqual(loaded(x), x * 2 + 4)
        self.assertEqual(loaded(alternate), alternate - 1)

    def test_scripts_parent_when_there_are_no_aoti_regions(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("calls", torch.zeros((), dtype=torch.int64))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.calls.add_(1)
                return x + self.calls

        model = Model()
        x = torch.randn(2)

        result = compile_aoti_regions(model, (x,))

        self.assertIsInstance(result, torch.jit.RecursiveScriptModule)
        self.assertEqual(model.calls, torch.zeros_like(model.calls))
        self.assertEqual(result(x), x + 1)
        self.assertEqual(model.calls, torch.zeros_like(model.calls))
        self.assertEqual(result.calls, torch.ones_like(result.calls))

    def test_region_compile_failure_stops_parent_compilation(self) -> None:
        root = torch.nn.Identity()
        region_exports = (Mock(), Mock(), Mock())
        first_compiled = Mock()
        error = RuntimeError("region compilation failed")

        with (
            patch(
                "torch._export._aoti_region._capture_aoti_regions",
                return_value=(Mock(),),
            ),
            patch(
                "torch._export._aoti_region._export_aoti_regions",
                return_value=region_exports,
            ),
            patch(
                "torch._export._aoti_region._compile_aoti_region",
                side_effect=(first_compiled, error),
            ) as compile,
            patch(
                "torch._export._aoti_region._substitute_compiled_aoti_regions"
            ) as substitute,
            patch("torch.jit.script") as script,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "region compilation failed"
            ) as cm:
                compile_aoti_regions(root, ())

        self.assertIs(cm.exception, error)
        self.assertEqual(
            compile.call_args_list,
            [call(region_exports[0]), call(region_exports[1])],
        )
        substitute.assert_not_called()
        script.assert_not_called()

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
