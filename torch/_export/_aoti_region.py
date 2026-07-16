import dataclasses
import inspect
import typing
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import torch


_P = ParamSpec("_P")
_R = TypeVar("_R")
_AOTI_REGION_SPEC_ATTR = "_torch_aoti_region_spec"


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
