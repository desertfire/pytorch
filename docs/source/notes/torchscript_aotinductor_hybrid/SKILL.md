---
name: resume-torchscript-aoti-hybrid
description: Resume the in-progress PyTorch hybrid TorchScript, torch.export, and AOTInductor implementation. Use when restoring or reviewing the AOTI-region branch, continuing direct annotated-function support, validating the single-thread/current-stream runtime, or updating the implementation design. This file is a self-contained handoff and requires no sibling skill files.
---

# Resume TorchScript AOTI Hybrid

Continue a PyTorch prototype in small, independently reviewed commits. TorchScript owns parent control flow. Strict `torch.export` and AOTInductor own explicitly annotated compute regions. The AOTI backend owns one single-threaded runner and executes CUDA work on the caller's current stream.

## Restore the Reviewed Branch

The reviewed implementation consists of 18 commits at:

```text
8da58b476f11d27d3ed703a0fca9fbfbf06079d4
```

It was copied without rewriting hashes into the personal fork checkout whose remote was `git@github.com:desertfire/pytorch.git`. The destination checkout has both `regional_aoti` and the legacy transfer name `aoti_ewfactor_1` pointing at this commit. After one is uploaded, restore the preferred `regional_aoti` name with:

```bash
git fetch origin regional_aoti
git switch --create regional_aoti --track origin/regional_aoti
```

If only `origin/aoti_ewfactor_1` exists, substitute that name. If the branch already exists locally, inspect it rather than recreating it. Confirm:

```bash
git rev-parse HEAD
git rev-list --count 3556b95a8b2f1fc95b6cad78ea13f0500b2794e9..HEAD
git status --short
```

Expected count: `18`. Base prerequisite: `3556b95a8b2f1fc95b6cad78ea13f0500b2794e9`.

## Read Repository Instructions First

Read the destination repository's `AGENTS.md` before modifying, building, testing, linting, or committing. The source repository required:

- Preserve unrelated dirty worktree files.
- Never touch `.ci/docker/` unless intentionally rebuilding CI images.
- Build only with `pip install -e . -v --no-build-isolation`, after discovering the local incremental-build configuration.
- Lint only through `spin`.
- Run `lintrunner -a` immediately before every requested commit.
- Use PyTorch `TestCase`, `run_tests`, `parametrize`, and device-generic test patterns.
- Include a fenced-command Test Plan and `Authored with an AI assistant.` in commit messages.

Obey the destination instructions if they differ.

## Non-Negotiable Design

- Do not rewrite TorchScript graph text after scripting.
- Resolve direct annotated calls during TorchScript static analysis to ordinary `prim::GetAttr` plus `prim::CallMethod` nodes.
- Keep compiled runners owned by serialized module state. Do not create a process-global string-to-runner registry.
- Keep the MVP runtime single-threaded: one runner, no overlapping host calls, and one ordered caller stream unless the caller externally synchronizes.
- Pass a null stream handle to AOTI so CUDA uses the caller's current stream.
- Keep the boundary ABI flat: Tensor inputs and Tensor or nonempty flat fixed tuple of Tensor outputs.
- Strict-export each region. Keep tensor-dependent parent control flow in TorchScript.
- Reject hidden or aliased state rather than silently freezing ambiguous state.
- Keep training, cross-boundary autograd, dynamic shapes, multi-runner execution, and multi-thread inference out of this phase.

## Implemented Compile Pipeline

Public experimental API:

```python
@torch._export.aoti_region
def forward_or_function(...):
    ...

torch._export.bind_aoti_region(function)
torch._export.compile_aoti_regions(model, args, kwargs=None)
```

Annotated submodules work directly:

```python
class Region(torch.nn.Module):
    @torch._export.aoti_region
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
```

Stateless functions work through explicit binding:

```python
@torch._export.aoti_region
def fast(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fast = torch._export.bind_aoti_region(fast)
```

Direct `fast(x)` calls from a parent method are not yet integrated.

`torch/_export/_aoti_region.py` currently:

1. Validates the root, annotations, placement, ABI, and state ownership.
2. Discovers annotated submodule `forward` methods.
3. Rejects aliased modules, nested regions, boundary state aliases, hooks, and unsupported schemas.
4. Runs eager calibration once with supplied inputs and records completed region invocations through module hooks.
5. Normalizes arguments and verifies compatible static Tensor metadata across observed calls.
6. Strict-exports each region using its first invocation.
7. Replays all captured invocations through the exported program and checks output ABI and aliasing.
8. Generates a typed schema-only TorchScript stub.
9. Calls `torch._inductor.aoti_compile_and_package` once per region.
10. Lowers each stub with `torch._C._jit_to_backend("aoti", ...)`.
11. Deep-copies the parent, substitutes compiled region submodules, and scripts the copied parent.

The original eager model remains unchanged.

## Runtime and Serialization

The AOTI JIT backend is implemented under `torch/csrc/jit/backends/aoti/` and registered in the JIT/backend build.

It:

- Embeds each `.pt2` package as a contiguous CPU `uint8` Tensor during preprocessing.
- Serializes those bytes inside lowered TorchScript state.
- Materializes bytes into a `c10::TempFile` during backend compilation/load.
- Keeps the temporary package alive for the loader lifetime.
- Creates exactly one `AOTIModelPackageLoader` with `run_single_threaded=true` and `num_runners=1`.
- Boxes only Tensor inputs and outputs.
- Calls `loader_->boxed_run(inputs, nullptr)` so CUDA uses the caller's current stream.

The contract requires calls on one backend instance to come from one host thread without overlap. Calls must remain on one ordered current stream unless externally synchronized before switching streams.

C++ tests in `test/cpp/aoti_inference/test.cpp` cover CPU/CUDA execution, validation, preprocessing, embedded bytes, deletion of the source package, save/load, and execution under a non-default CUDA `CUDAStreamGuard`.

## Bound Function Safety

`bind_aoti_region` creates a fresh `nn.Module` whose generated `forward` has real fixed-arity bytecode parameters. It does not forge `__signature__` over `*args/**kwargs`.

It accepts only positional-or-keyword Tensor parameters without defaults. Returns must be Tensor or a nonempty flat fixed tuple of Tensor.

It recursively inspects captures with cycle protection. It rejects Tensor, Parameter, Module, arbitrary object, custom class, and custom module state reachable through containers, helper closures, defaults, keyword defaults, function attributes, partials, slices, and bound receivers. Trusted built-in and PyTorch modules/classes/functions are accepted by object identity rather than mutable metadata strings.

## Reviewed Commit History

Apply or compare in this exact order:

```text
0f57b0e4d5e6613b086464b5faafa5eccb9190a2 [Export] Add experimental AOTI region contract
3ef7d24f3f3127e251f6c382b98479d4cb3def4c [Export] Capture AOTI region calibration calls
eed746a17820df75f4ef269b58db1726349d94b8 [Export] Strict-export annotated AOTI regions
e32b7cc27fd5a1f536586132b2eae2833c98762f [Export] Generate typed TorchScript region stubs
34a50038966b0783d1a097984aba33f7107c6e06 [JIT] Add single-threaded AOTI backend runtime
703e95b6ef5efdff1d0e48bd4e8d976e20d4b145 [JIT] Register AOTI backend preprocessing
058acaca19a51c60414355c6f3bb9a9253d66702 [Export] Lower AOTI region stubs through JIT
2b3974f872d0d1e3b9dfb45d86893386f81f77d9 [Export] Validate flat AOTI region schemas
64b9b2182a3d5d82be16c78487cfe953b785ef0e [Export] Compile individual AOTI regions
0a915ec051cd17cad79b168d7bdd84244073fb8a [Export] Substitute compiled AOTI regions
5d59bc86b30d5e5f24a6ecff9817ed68d5e31c5b [Export] Orchestrate AOTI region compilation
81913b2dbb813a46b905644fe9d6a89dfe8d9538 [JIT] Embed AOTI packages in processed state
dd204a490838be7fb8e027e311b8b55421d336a4 [JIT] Load AOTI packages from embedded state
7717a7325ce100545ecabf247fc339fc3c43a329 [Export] Expose AOTI region compilation
ddaae8ba7e5935350ce9cfdbbbb0711122c14e7d [Export] Test hybrid control flow composition
581d20c17ae1b180317f638bdfd40ce866549139 [Export] Reject AOTI region state aliases
3db06245f0b9a3d7d10428c2e4cce95731aaea36 [Export] Bind annotated functions as AOTI regions
8da58b476f11d27d3ed703a0fca9fbfbf06079d4 [Export] Test bound AOTI region composition
```

## Verification Status

Static checks passed repeatedly on reviewed commits:

```bash
python -m py_compile torch/_export/_aoti_region.py test/export/test_aoti_region.py
git diff --check
spin quicklint <changed files>
lintrunner -a
```

Runtime Python tests could not start in the source environment because the existing local extension lacked `torch.bcomplex32` during `import torch`. No C++ rebuild was run because the required local incremental-build configuration was unavailable.

Do not claim runtime success until the destination builds correctly and runs focused Python plus CPU/CUDA AOTI tests.

The source worktree also had an unrelated modification to `torch/csrc/inductor/aoti_runtime/model_container.h` and many untracked repro/build files. None belong to this implementation. Never stage them.

## Commit and Review Protocol

For every new commit:

1. Give a fresh implementation agent one bounded commit with minimal necessary context.
2. Inspect the resulting diff locally.
3. Give a different fresh reviewer the raw diff and acceptance criteria.
4. Send valid findings back to the implementer. Push back on findings that contradict repository behavior or the agreed architecture.
5. Require explicit approval after revisions.
6. Run focused tests and required lint.
7. Stage only reviewed files and preserve unrelated changes.
8. Commit with a fenced-command Test Plan and AI disclosure.

Do not combine frontend resolution, direct-function discovery, calibration, substitution, and end-to-end integration in one commit.

## Next Commit: Review the WIP Module Reference

An implementation agent produced the three-file patch embedded at the end of this skill. It was never approved: its fresh reviewer spawned C++ and test-design checks, then failed before returning findings.

Apply the embedded patch only after confirming `HEAD=8da58b476f1`. Give it to a new reviewer with these questions:

- Does `py::type::handle_of(obj).is(module_reference_type)` compile and enforce exact type identity?
- Is importing `torch._jit_internal` inside every `toSugaredValue` call acceptable, or should recognition be cheaper?
- Is graph input 0 guaranteed to be module `self` for supported use, with clear rejection elsewhere?
- Does each `ClassType` check prove every path segment is a registered submodule before `GetAttr` emission?
- Do nested paths work, including numeric Sequential/ModuleList names?
- Does `SimpleValue(...).attr("forward")` plus `MethodValue::call` preserve positional/keyword schema matching and return binders?
- Should Python reject dots inside individual path segments?
- Add negative tests for missing/non-module paths and use from a scripted free function.
- Add a nested-path positive test and save/load assertion if inexpensive.
- Confirm serialization contains only existing `prim::GetAttr` and `prim::CallMethod`, with no custom runtime node or global registry.

Apply valid findings, re-review, build/test/lint, then commit only those three files.

## Following Small Commits

### Discover Direct Annotated Functions

Discover exact marked Python function identities referenced directly as global/nonlocal names by scriptable parent methods. Reuse `bind_aoti_region` signature and capture validation. Assign deterministic, collision-checked private delegate paths per compiled root.

Keep the first version to direct name calls. Reject local aliases, containers, arbitrary object indirection, and nested helper indirection unless separately implemented and tested. This commit should only discover and map; do not alter calibration or scripting.

### Capture Direct Function Calls

Module hooks cannot observe a plain global function. Add a scoped calibration mechanism without permanently replacing shared globals.

A candidate is a temporary `sys.setprofile` callback that matches exact annotated code objects, captures fixed-signature arguments on `call`, captures outputs on `return`, tracks frames for recursion/exceptions, and restores an existing profiler in `finally`. Compare this against scoped wrapper/resolver alternatives before choosing. Explicitly reject unsupported concurrent/threaded calibration.

### Export and Compile Direct Functions

Use a synthetic stateless bound module for strict export and AOTInductor compilation. Reuse ABI, static metadata, output alias, graph-signature, and capture-state validation. Test repeated calls with compatible metadata and rejection of incompatible calls.

### Register Delegates and Script Through the Resolver

On the copied parent only, register compiled direct-function modules at private paths. Wrap only relevant method resolution callbacks. Map exact annotated function identities to `_TorchScriptModuleReference(path)` and delegate every other name to the original callback.

Do not mutate the original model, Python function, or function globals. Prove mappings cannot leak between separately compiled model instances.

### Add Direct Function End-to-End Coverage

Use a parent with tensor-dependent TorchScript control flow and a direct annotated global call. Mock only native AOTInductor packaging/lowering. Exercise calibration, strict export replay, delegate registration, `GetAttr + CallMethod`, both control-flow branches, distinct eager/compiled formulas, original model preservation, and TorchScript save/load without Python.

### Update the HTML Design

The prior HTML design is not embedded here and is not authoritative. Recreate or update it from this status. Remove stale runner-pool, content-addressed-cache, and memory-backed package claims. Describe the implemented one-runner/current-stream contract and `c10::TempFile` package materialization. Mark bound functions complete and direct calls in progress.

### Build and Run

After discovering the destination build configuration, run the repository-approved build. Then run focused tests such as:

```bash
python test/export/test_aoti_region.py
python test/jit/test_recursive_script.py -k module_reference
```

Run the applicable `test_aoti_inference` CPU and CUDA targets through the repository's build/test system. Record exact commands and results.

## WIP Patch to Review, Not Yet Approved

Save the following block as a patch and apply it with `git apply --check` followed by `git apply`. Do not commit it before fresh review.

```diff
diff --git a/test/jit/test_recursive_script.py b/test/jit/test_recursive_script.py
index 4399c260499..73e44cc6e89 100644
--- a/test/jit/test_recursive_script.py
+++ b/test/jit/test_recursive_script.py
@@ -77,6 +77,53 @@ class TestRecursiveScript(JitTestCase):

         self.checkModule(mod, (torch.randn(2, 2),))

+    def test_resolution_callback_module_reference(self):
+        def global_fn(x: torch.Tensor) -> torch.Tensor:
+            return x - 100
+
+        class Replacement(torch.nn.Module):
+            def forward(self, x: torch.Tensor) -> torch.Tensor:
+                return x * 2
+
+        replacement = torch.jit.script(Replacement())
+
+        class M(torch.nn.Module):
+            def __init__(self) -> None:
+                super().__init__()
+                self.replacement = replacement
+
+            def forward(self, x: torch.Tensor) -> torch.Tensor:
+                return global_fn(x)
+
+        model = M()
+
+        def make_stubs(module):
+            stubs = torch.jit._recursive.infer_methods_to_compile(module)
+            if module is not model:
+                return stubs
+            result = []
+            for stub in stubs:
+                callback = stub.resolution_callback
+
+                def resolve(name, callback=callback):
+                    value = callback(name)
+                    if value is global_fn:
+                        return torch._jit_internal._TorchScriptModuleReference(
+                            ("replacement",)
+                        )
+                    return value
+
+                result.append(stub._replace(resolution_callback=resolve))
+            return result
+
+        scripted = torch.jit._recursive.create_script_module(model, make_stubs)
+        x = torch.randn(2, 2)
+
+        self.assertEqual(scripted(x), replacement(x))
+        FileCheck().check('prim::GetAttr[name="replacement"]').check(
+            'prim::CallMethod[name="forward"]'
+        ).run(scripted.graph)
+
     def test_failed_function_compilation(self):
         def fn(x):
             return i_dont_exist  # noqa: F821
diff --git a/torch/_jit_internal.py b/torch/_jit_internal.py
index 9c2c9509a81..1475d8a77b1 100644
--- a/torch/_jit_internal.py
+++ b/torch/_jit_internal.py
@@ -55,6 +55,21 @@ class HasGetattr(Protocol):
     def __getattr__(self, key: str) -> Any: ...


+class _TorchScriptModuleReference:
+    __slots__ = ("_path",)
+
+    def __init__(self, path: tuple[str, ...]) -> None:
+        if not isinstance(path, tuple) or not path or any(
+            not isinstance(name, str) or not name for name in path
+        ):
+            raise ValueError("TorchScript module reference path must be nonempty")
+        self._path = path
+
+    @property
+    def path(self) -> tuple[str, ...]:
+        return self._path
+
+
 _P = ParamSpec("_P")
 _R = TypeVar("_R")

diff --git a/torch/csrc/jit/python/python_sugared_value.cpp b/torch/csrc/jit/python/python_sugared_value.cpp
index 6bf206e2614..2db05154624 100644
--- a/torch/csrc/jit/python/python_sugared_value.cpp
+++ b/torch/csrc/jit/python/python_sugared_value.cpp
@@ -27,6 +27,52 @@ std::optional<StrongFunctionPtr> as_function(const py::object& obj) {
   return std::nullopt;
 }

+struct VISIBILITY_HIDDEN PythonModuleReferenceValue : public SugaredValue {
+  explicit PythonModuleReferenceValue(std::vector<std::string> path)
+      : path_(std::move(path)) {}
+
+  std::string kind() const override {
+    return "module reference";
+  }
+
+  std::shared_ptr<SugaredValue> call(
+      const SourceRange& loc,
+      GraphFunction& m,
+      at::ArrayRef<NamedValue> args,
+      at::ArrayRef<NamedValue> kwargs,
+      size_t n_binders) override {
+    auto graph = m.graph();
+    if (graph->inputs().empty() ||
+        !graph->inputs().at(0)->type()->is_module()) {
+      throw(
+          ErrorReport(loc)
+          << "A module reference can only be called from a module method");
+    }
+    if (path_.empty()) {
+      throw(ErrorReport(loc) << "A module reference path cannot be empty");
+    }
+
+    Value* module = graph->inputs().at(0);
+    for (const auto& field : path_) {
+      auto class_type = module->type()->cast<ClassType>();
+      if (field.empty() || !class_type || !class_type->hasAttribute(field) ||
+          !class_type->getAttribute(field)->is_module()) {
+        throw(
+            ErrorReport(loc)
+            << "Module reference attribute '" << field
+            << "' is not a registered submodule of the caller");
+      }
+      module = graph->insertGetAttr(module, field);
+    }
+    return SimpleValue(module)
+        .attr(loc, m, "forward")
+        ->call(loc, m, args, kwargs, n_binders);
+  }
+
+ private:
+  std::vector<std::string> path_;
+};
+
 FunctionSchema PythonValue::getSchema(
     const size_t n_args,
     const size_t n_binders,
@@ -1191,6 +1237,13 @@ std::shared_ptr<SugaredValue> toSugaredValue(
     obj = py::getattr(obj, "op");
   }

+  auto module_reference_type = py::module::import("torch._jit_internal")
+                                   .attr("_TorchScriptModuleReference");
+  if (py::type::handle_of(obj).is(module_reference_type)) {
+    return std::make_shared<PythonModuleReferenceValue>(
+        py::cast<std::vector<std::string>>(obj.attr("path")));
+  }
+
 #ifdef USE_RPC
   bool isRpcAvailable = py::cast<bool>(
       py::module::import("torch.distributed.rpc").attr("is_available")());
```
