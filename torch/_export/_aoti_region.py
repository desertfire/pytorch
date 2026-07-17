import dataclasses
import inspect
import typing
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import torch
from torch.export.graph_signature import InputKind, OutputKind


_P = ParamSpec("_P")
_R = TypeVar("_R")
_AOTI_REGION_SPEC_ATTR = "_torch_aoti_region_spec"
_TORCHSCRIPT_RESERVED_PARAMETER_NAMES = {"Ellipsis", "NoneType"}


@dataclasses.dataclass(frozen=True)
class AOTIRegionSpec:
    """Experimental metadata attached to an AOTI region method."""

    schema_version: int = dataclasses.field(default=1, init=False)


@dataclasses.dataclass(frozen=True)
class _AOTIRegion:
    module_fqn: str
    module: torch.nn.Module
    spec: AOTIRegionSpec
    signature: inspect.Signature


@dataclasses.dataclass(frozen=True)
class _AOTIRegionInvocation:
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]
    output: Any


@dataclasses.dataclass(frozen=True)
class _AOTIRegionCapture:
    region: _AOTIRegion
    invocations: tuple[_AOTIRegionInvocation, ...]


@dataclasses.dataclass(frozen=True)
class _AOTIRegionExport:
    region: _AOTIRegion
    example_args: tuple[Any, ...]
    exported_program: "torch.export.ExportedProgram"


@dataclasses.dataclass(frozen=True)
class _AOTIRegionStub:
    module: "torch.jit.ScriptModule"
    source: str


@typing.overload
def aoti_region(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...


@typing.overload
def aoti_region(
    fn: None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


def aoti_region(
    fn: Callable[_P, _R] | None = None,
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Mark an ``nn.Module.forward`` method as an experimental AOTI region.

    This decorator only records metadata. A later hybrid compilation API will
    discover and compile the marked submodule.
    """

    def mark(method: Callable[_P, _R]) -> Callable[_P, _R]:
        if not inspect.isfunction(method):
            raise TypeError("@aoti_region must decorate a Python method")
        if hasattr(method, _AOTI_REGION_SPEC_ATTR):
            raise ValueError(
                f"{method.__qualname__} is already marked as an AOTI region"
            )
        setattr(method, _AOTI_REGION_SPEC_ATTR, AOTIRegionSpec())
        return method

    if fn is None:
        return mark
    return mark(fn)


def _marked_methods(module: torch.nn.Module) -> tuple[tuple[str, Any, Any], ...]:
    method_names = {name for cls in type(module).__mro__ for name in cls.__dict__}
    marked = []
    for name in sorted(method_names):
        descriptor = inspect.getattr_static(type(module), name)
        method = (
            descriptor.__func__
            if isinstance(descriptor, (classmethod, staticmethod))
            else descriptor
        )
        spec = getattr(method, _AOTI_REGION_SPEC_ATTR, None)
        if spec is not None:
            marked.append((name, descriptor, spec))
    return tuple(marked)


def _validate_abi_type(annotation: Any, location: str) -> None:
    if annotation is torch.Tensor:
        return

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is tuple and not args:
        raise TypeError(
            f"AOTI region {location} must use a nonempty fixed tuple with "
            "explicit element types"
        )
    if origin is tuple and Ellipsis not in args:
        for index, element_type in enumerate(args):
            _validate_abi_type(element_type, f"{location}[{index}]")
        return

    raise TypeError(
        f"AOTI region {location} must be annotated as torch.Tensor or a fixed "
        "tuple of supported types, "
        f"but got {annotation!r}"
    )


def _validated_signature(
    method: Callable[..., Any], module_fqn: str
) -> inspect.Signature:
    try:
        signature = inspect.signature(method)
        type_hints = typing.get_type_hints(method)
    except (AttributeError, NameError, SyntaxError, TypeError) as exc:
        raise TypeError(
            f"Could not resolve annotations for AOTI region "
            f"'{module_fqn}.forward': {exc}"
        ) from exc

    parameters = tuple(signature.parameters.values())
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError(
            f"AOTI region '{module_fqn}.forward' must be an instance method"
        )

    resolved_parameters = [parameters[0]]
    for parameter in parameters[1:]:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"AOTI region '{module_fqn}.forward' does not support "
                "variadic parameters"
            )
        if (
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
        ):
            raise TypeError(
                f"AOTI region '{module_fqn}.forward' does not support required "
                f"keyword-only parameter '{parameter.name}'"
            )
        if parameter.default is not inspect.Parameter.empty:
            raise TypeError(
                f"AOTI region '{module_fqn}.forward' does not support default "
                "parameter values"
            )
        if parameter.name not in type_hints:
            raise TypeError(
                f"AOTI region '{module_fqn}.forward' parameter "
                f"'{parameter.name}' is missing a type annotation"
            )
        annotation = type_hints[parameter.name]
        location = f"'{module_fqn}.forward' parameter '{parameter.name}'"
        _validate_abi_type(annotation, location)
        resolved_parameters.append(parameter.replace(annotation=annotation))

    if "return" not in type_hints:
        raise TypeError(
            f"AOTI region '{module_fqn}.forward' is missing a return type annotation"
        )
    return_type = type_hints["return"]
    _validate_abi_type(return_type, f"'{module_fqn}.forward' return")
    return signature.replace(
        parameters=resolved_parameters, return_annotation=return_type
    )


def _discover_aoti_regions(root: torch.nn.Module) -> tuple[_AOTIRegion, ...]:
    """Discover and validate marked submodule ``forward`` methods."""
    if not isinstance(root, torch.nn.Module):
        raise TypeError(f"Expected an nn.Module, but got {type(root)!r}")

    regions = []
    paths_by_module_id: dict[int, str] = {}
    for module_fqn, module in root.named_modules(remove_duplicate=False):
        marked_methods = _marked_methods(module)
        if not marked_methods:
            continue

        unsupported = [name for name, _, _ in marked_methods if name != "forward"]
        if unsupported:
            path = module_fqn or "<root>"
            names = ", ".join(repr(name) for name in unsupported)
            raise ValueError(
                f"AOTI region methods on '{path}' must be named 'forward'; "
                f"found {names}"
            )
        if module_fqn == "":
            raise ValueError(
                "The root module cannot be an AOTI region; mark a submodule "
                "forward method instead"
            )

        _, descriptor, spec = marked_methods[0]
        if isinstance(descriptor, (classmethod, staticmethod)):
            raise TypeError(
                f"AOTI region '{module_fqn}.forward' must be an instance method"
            )
        if not isinstance(spec, AOTIRegionSpec):
            raise TypeError(
                f"AOTI region '{module_fqn}.forward' has invalid marker metadata"
            )

        module_id = id(module)
        if module_id in paths_by_module_id:
            first_path = paths_by_module_id[module_id]
            raise ValueError(
                f"AOTI region submodule is aliased by both '{first_path}' and "
                f"'{module_fqn}'; shared regions are unsupported"
            )
        paths_by_module_id[module_id] = module_fqn

        for region in regions:
            if module_fqn.startswith(region.module_fqn + "."):
                raise ValueError(
                    f"AOTI regions '{region.module_fqn}' and '{module_fqn}' "
                    "overlap; nested regions are unsupported"
                )

        signature = _validated_signature(descriptor, module_fqn)
        regions.append(_AOTIRegion(module_fqn, module, spec, signature))

    return tuple(regions)


def _capture_aoti_regions(
    root: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> tuple[_AOTIRegionCapture, ...]:
    """Run eager calibration and capture every completed region invocation."""
    regions = _discover_aoti_regions(root)
    kwargs = {} if kwargs is None else kwargs

    global_hook_kinds = []
    if torch.nn.modules.module._global_forward_pre_hooks:
        global_hook_kinds.append("forward pre-hooks")
    if torch.nn.modules.module._global_forward_hooks:
        global_hook_kinds.append("forward hooks")
    if global_hook_kinds:
        hooks = " and ".join(global_hook_kinds)
        raise ValueError(
            f"Global nn.Module {hooks} are unsupported during AOTI region calibration"
        )

    for region in regions:
        hook_kinds = []
        if region.module._forward_pre_hooks:
            hook_kinds.append("forward pre-hooks")
        if region.module._forward_hooks:
            hook_kinds.append("forward hooks")
        if hook_kinds:
            hooks = " and ".join(hook_kinds)
            raise ValueError(
                f"AOTI region '{region.module_fqn}.forward' has existing "
                f"{hooks}; region hooks are unsupported"
            )

    pending: dict[
        str, list[tuple[int, tuple[Any, ...], tuple[tuple[str, Any], ...]]]
    ] = {region.module_fqn: [] for region in regions}
    completed: dict[str, list[tuple[int, _AOTIRegionInvocation]]] = {
        region.module_fqn: [] for region in regions
    }
    next_sequence = {region.module_fqn: 0 for region in regions}
    handles = []

    def make_pre_hook(module_fqn: str) -> Callable[..., None]:
        def pre_hook(
            module: torch.nn.Module,
            call_args: tuple[Any, ...],
            call_kwargs: dict[str, Any],
        ) -> None:
            sequence = next_sequence[module_fqn]
            next_sequence[module_fqn] += 1
            pending[module_fqn].append(
                (sequence, tuple(call_args), tuple(sorted(call_kwargs.items())))
            )

        return pre_hook

    def make_post_hook(module_fqn: str) -> Callable[..., None]:
        def post_hook(
            module: torch.nn.Module,
            call_args: tuple[Any, ...],
            call_kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            sequence, captured_args, captured_kwargs = pending[module_fqn].pop()
            if output is None:
                return
            invocation = _AOTIRegionInvocation(captured_args, captured_kwargs, output)
            completed[module_fqn].append((sequence, invocation))

        return post_hook

    try:
        for region in regions:
            handles.append(
                region.module.register_forward_pre_hook(
                    make_pre_hook(region.module_fqn), with_kwargs=True
                )
            )
            handles.append(
                region.module.register_forward_hook(
                    make_post_hook(region.module_fqn),
                    with_kwargs=True,
                    always_call=True,
                )
            )
        with torch.no_grad():
            root(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    unexercised = [
        region.module_fqn for region in regions if not completed[region.module_fqn]
    ]
    if unexercised:
        names = ", ".join(repr(name) for name in unexercised)
        raise ValueError(
            f"AOTI regions {names} did not complete an invocation during calibration"
        )

    return tuple(
        _AOTIRegionCapture(
            region,
            tuple(invocation for _, invocation in sorted(completed[region.module_fqn])),
        )
        for region in regions
    )


def _validate_abi_value(value: Any, annotation: Any, location: str) -> None:
    if annotation is torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"AOTI region {location} must be a torch.Tensor, "
                f"but got {type(value)!r}"
            )
        return

    element_types = typing.get_args(annotation)
    if not isinstance(value, tuple):
        raise TypeError(
            f"AOTI region {location} must be a tuple with "
            f"{len(element_types)} elements, but got {type(value)!r}"
        )
    if len(value) != len(element_types):
        raise TypeError(
            f"AOTI region {location} must be a tuple with "
            f"{len(element_types)} elements, but got {len(value)}"
        )
    for index, (element, element_type) in enumerate(zip(value, element_types)):
        _validate_abi_value(element, element_type, f"{location}[{index}]")


def _abi_tensors(value: Any, path: str) -> tuple[tuple[str, torch.Tensor], ...]:
    if isinstance(value, torch.Tensor):
        return ((path, value),)
    return tuple(
        tensor
        for index, element in enumerate(value)
        for tensor in _abi_tensors(element, f"{path}[{index}]")
    )


def _invocation_input_tensors(
    region: _AOTIRegion, normalized_args: tuple[Any, ...]
) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        tensor
        for parameter, value in zip(
            tuple(region.signature.parameters.values())[1:], normalized_args
        )
        for tensor in _abi_tensors(value, f"parameter '{parameter.name}'")
    )


def _validate_invocation_output(
    region: _AOTIRegion,
    normalized_args: tuple[Any, ...],
    output: Any,
    location: str,
) -> None:
    _validate_abi_value(
        output,
        region.signature.return_annotation,
        f"{location} return",
    )
    input_tensors = _invocation_input_tensors(region, normalized_args)
    for output_path, output_tensor in _abi_tensors(output, "return"):
        for input_path, input_tensor in input_tensors:
            if torch._C._overlaps(output_tensor, input_tensor):
                raise ValueError(
                    f"AOTI region {location} {output_path} aliases user input "
                    f"{input_path}; outputs must have independent storage"
                )


def _tensor_static_metadata(tensor: torch.Tensor) -> tuple[tuple[str, Any], ...]:
    metadata: list[tuple[str, Any]] = [
        ("type", type(tensor)),
        ("dtype", tensor.dtype),
        ("device", tensor.device),
        ("layout", tensor.layout),
        ("shape", tuple(tensor.shape)),
        ("requires_grad", tensor.requires_grad),
    ]
    if tensor.layout is torch.strided:
        metadata.extend(
            (
                ("stride", tuple(tensor.stride())),
                ("storage_offset", tensor.storage_offset()),
            )
        )
    return tuple(metadata)


def _validate_static_input_metadata(
    region: _AOTIRegion,
    reference_args: tuple[Any, ...],
    normalized_args: tuple[Any, ...],
    invocation_index: int,
) -> None:
    reference_tensors = _invocation_input_tensors(region, reference_args)
    invocation_tensors = _invocation_input_tensors(region, normalized_args)
    for (reference_path, reference), (path, tensor) in zip(
        reference_tensors, invocation_tensors
    ):
        if path != reference_path:
            raise AssertionError(
                f"Input tensor paths differ: {reference_path} and {path}"
            )
        reference_metadata = dict(_tensor_static_metadata(reference))
        for name, value in _tensor_static_metadata(tensor):
            if value != reference_metadata[name]:
                raise ValueError(
                    f"AOTI region '{region.module_fqn}.forward' invocation "
                    f"{invocation_index} {path} has {name} {value}, but the "
                    f"first invocation uses {reference_metadata[name]}"
                )


def _normalize_aoti_region_invocation(
    capture: _AOTIRegionCapture,
    invocation: _AOTIRegionInvocation,
    invocation_index: int,
) -> tuple[Any, ...]:
    region = capture.region
    location = f"'{region.module_fqn}.forward' invocation {invocation_index}"
    try:
        bound = region.signature.bind(
            region.module, *invocation.args, **dict(invocation.kwargs)
        )
    except TypeError as exc:
        raise TypeError(f"Could not bind AOTI region {location}: {exc}") from exc

    normalized_args = tuple(
        bound.arguments[parameter.name]
        for parameter in tuple(region.signature.parameters.values())[1:]
    )
    for parameter, value in zip(
        tuple(region.signature.parameters.values())[1:], normalized_args
    ):
        _validate_abi_value(
            value,
            parameter.annotation,
            f"{location} parameter '{parameter.name}'",
        )
    _validate_invocation_output(region, normalized_args, invocation.output, location)
    return normalized_args


def _validate_export_graph_signature(
    region: _AOTIRegion, exported_program: "torch.export.ExportedProgram"
) -> None:
    graph_signature = exported_program.graph_signature
    allowed_input_kinds = {
        InputKind.USER_INPUT,
        InputKind.PARAMETER,
        InputKind.BUFFER,
        InputKind.CONSTANT_TENSOR,
        InputKind.CUSTOM_OBJ,
    }
    effects = [
        f"input {spec.kind.name}"
        for spec in graph_signature.input_specs
        if spec.kind not in allowed_input_kinds
    ]
    effects.extend(
        f"output {spec.kind.name}"
        for spec in graph_signature.output_specs
        if spec.kind is not OutputKind.USER_OUTPUT
    )
    if effects:
        raise ValueError(
            f"AOTI region '{region.module_fqn}.forward' export has unsupported "
            f"boundary effects: {', '.join(effects)}"
        )


def _export_aoti_regions(
    captures: tuple[_AOTIRegionCapture, ...],
) -> tuple[_AOTIRegionExport, ...]:
    """Strict-export captured regions using their first completed invocation."""
    exports = []
    for capture in captures:
        region = capture.region
        if not capture.invocations:
            raise ValueError(
                f"AOTI region '{region.module_fqn}.forward' has no completed "
                "invocations to export"
            )

        normalized_invocations = tuple(
            _normalize_aoti_region_invocation(capture, invocation, index)
            for index, invocation in enumerate(capture.invocations, 1)
        )
        example_args = normalized_invocations[0]
        try:
            with torch.no_grad():
                exported_program = torch.export.export(
                    region.module, example_args, strict=True
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to export AOTI region '{region.module_fqn}.forward': {exc}"
            ) from exc

        _validate_export_graph_signature(region, exported_program)
        try:
            exported_module = exported_program.module()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to prepare exported AOTI region "
                f"'{region.module_fqn}.forward': {exc}"
            ) from exc
        for index, invocation_args in enumerate(normalized_invocations, 1):
            if index > 1:
                _validate_static_input_metadata(
                    region, example_args, invocation_args, index
                )
            try:
                with torch.no_grad():
                    output = exported_module(*invocation_args)
            except Exception as exc:
                raise RuntimeError(
                    f"AOTI region '{region.module_fqn}.forward' invocation "
                    f"{index} is incompatible with the exported program: {exc}"
                ) from exc
            _validate_invocation_output(
                region,
                invocation_args,
                output,
                f"'{region.module_fqn}.forward' exported invocation {index}",
            )

        exports.append(_AOTIRegionExport(region, example_args, exported_program))

    return tuple(exports)


def _render_aoti_region_stub_type(annotation: Any) -> str:
    if annotation is torch.Tensor:
        return "Tensor"
    elements = ", ".join(
        _render_aoti_region_stub_type(element)
        for element in typing.get_args(annotation)
    )
    return f"Tuple[{elements}]"


def _render_aoti_region_stub_source(region: _AOTIRegion) -> str:
    parameters = tuple(region.signature.parameters.values())[1:]
    for parameter in parameters:
        if not parameter.name.isascii():
            raise TypeError(
                f"AOTI region '{region.module_fqn}.forward' parameter "
                f"'{parameter.name}' cannot be represented in TorchScript; "
                "parameter names must be ASCII"
            )
        if parameter.name in _TORCHSCRIPT_RESERVED_PARAMETER_NAMES:
            raise TypeError(
                f"AOTI region '{region.module_fqn}.forward' parameter "
                f"'{parameter.name}' cannot be represented in TorchScript; "
                "parameter name is reserved"
            )
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"AOTI region '{region.module_fqn}.forward' positional-only "
                f"parameter '{parameter.name}' cannot be represented in TorchScript"
            )
        if parameter.name == "self":
            raise TypeError(
                f"AOTI region '{region.module_fqn}.forward' parameter 'self' "
                "cannot be represented in a TorchScript method"
            )

    arguments = ["self"]
    arguments.extend(
        f"{parameter.name}: {_render_aoti_region_stub_type(parameter.annotation)}"
        for parameter in parameters
    )
    return_type = _render_aoti_region_stub_type(region.signature.return_annotation)
    return (
        f"def forward({', '.join(arguments)}) -> {return_type}:\n"
        '    assert False, "AOTI region schema stub cannot execute"\n'
    )


def _create_aoti_region_stub(region_export: _AOTIRegionExport) -> _AOTIRegionStub:
    source = _render_aoti_region_stub_source(region_export.region)

    class SchemaModule(torch.jit.ScriptModule):
        def __init__(self) -> None:
            super().__init__()
            self.define(source)

    return _AOTIRegionStub(SchemaModule(), source)


def _validate_aoti_region_stub_schema(schema: torch._C.FunctionSchema) -> None:
    for argument in schema.arguments[1:]:
        if argument.type.kind() != "TensorType":
            raise TypeError(
                "AOTI backend forward arguments must be Tensor; "
                f"argument '{argument.name}' has type {argument.type}"
            )

    if len(schema.returns) != 1:
        raise TypeError(
            "AOTI backend forward schema must have exactly one return, "
            f"but has {len(schema.returns)}"
        )

    output_type = schema.returns[0].type
    if output_type.kind() == "TensorType":
        return
    if output_type.kind() == "TupleType":
        elements = typing.cast(torch._C.TupleType, output_type).elements()
        if elements and all(element.kind() == "TensorType" for element in elements):
            return
    raise TypeError(
        "AOTI backend forward return must be Tensor or a nonempty flat tuple "
        f"of Tensor; got {output_type}"
    )


def _lower_aoti_region_stub(
    stub: _AOTIRegionStub, package_path: str
) -> "torch.jit.RecursiveScriptModule":
    if not isinstance(stub, _AOTIRegionStub):
        raise TypeError(f"Expected an _AOTIRegionStub, but got {type(stub)!r}")
    if not isinstance(package_path, str):
        raise TypeError(
            f"package_path must be a string, but got {type(package_path)!r}"
        )
    if not package_path:
        raise ValueError("package_path must be a non-empty string")

    schema = stub.module.forward.schema  # pyrefly: ignore[missing-attribute]
    _validate_aoti_region_stub_schema(schema)
    method_compile_spec = {
        "forward": {
            "package_path": package_path,
            "model_name": "model",
            "device_index": -1,
        }
    }
    return torch._C._jit_to_backend(  # pyrefly: ignore[missing-attribute]
        "aoti", stub.module, method_compile_spec
    )
