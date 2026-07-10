# SDX C# Binding

Files:

- `SdxRuntime.cs` — the complete SDK wrapper (zero external dependencies)

## Public surface

| Type | Purpose |
|---|---|
| `SdxRuntime` | Create/destroy the native runtime; load model bundles |
| `SdxModel` | Loaded model handle; create execution contexts |
| `SdxContext` | Execute, query and manage the DSP plan lifecycle |
| `SdxTensorViewLease` | Manage the unmanaged shape allocation for `SdxTensorView` |
| `SdxTensorView` | Non-owning tensor view passed to `SdxContext.Run` |
| `SdxExecutionReport` | Telemetry struct returned by `SdxContext.ExecutionReport` |
| `SdxRuntimeOptions` | Options for `SdxRuntime.Create` |
| `SdxModelOptions` | Options for `SdxRuntime.LoadModel` |
| `SdxRunOptions` | Per-call options for `SdxContext.Run` |
| `SdxConstants` | ABI integer constants (`SDX_BACKEND_*`, `SDX_PHASE_*`, `SDX_STATUS_*`, …) |

## Requirements

- .NET 6+ runtime (`NativeLibrary` P/Invoke resolver, `Marshal.PtrToStringUTF8`)
- `#nullable enable` is set inside the file; callers on nullable-disabled projects are unaffected

## Usage

```csharp
// Create the runtime (probes well-known library names automatically).
using var runtime = SdxRuntime.Create();                 // or Create("libsdx_cpu.so")

// Load a compiled model bundle.
using var model = runtime.LoadModel("model.sdz");

// Create an execution context requesting specific outputs.
using var ctx = model.CreateContext(new[] { "probs" });

// Discover the positional input contract from the loaded plan.
string?[] names = ctx.InputNames();  // e.g. ["w1","b1","w2","b2","x"]

// Mark runtime-variable inputs before the first Run.
ctx.MarkInputPlaceholder(4);  // "x" changes shape/value between calls

// Warmup — let the DSP plan observe and stabilise tensor shapes.
ctx.Run(inputViews, outputViews);

// Freeze shapes to enable CUDA graph capture and the argTable stable fast path.
ctx.FreezeShapes();

// Steady-state inference.
ctx.Run(inputViews, outputViews);

// Query telemetry after execution.
SdxExecutionReport report = ctx.ExecutionReport();
// report.plan_phase == SdxConstants.SDX_PHASE_REPLAYING (2) when graph replay is active.
```

## Native library resolution

`SdxRuntime.Create(libraryNameOrPath?)` calls `NativeLibrary.SetDllImportResolver` to
probe candidates in this order:

1. The explicit path/name passed to `Create` (if non-null).
2. `../../lib/` relative to the assembly — the standard SDK package layout.
3. Default candidates on the system path:
   `sdx_cpu`, `sdx_cuda`, `libsdx_cpu.so/.dylib`, `sdx_cpu.dll`,
   then the monolithic backend libraries `nd4jcpu`, `nd4jcuda`, `nd4jamd`
   and their platform-suffixed variants.

## P/Invoke struct layout notes

All structs use `[StructLayout(LayoutKind.Sequential)]` and match the
`sdx_runtime_c.h` layout exactly.  Fields use the same integer widths as the C
counterparts (`uint` for `uint32_t`, `ulong` for `uint64_t`, `UIntPtr` for
`size_t`).  The `struct_size` field in each options/report struct must be
pre-filled via the `Default()` factory — this is the ABI versioning mechanism.

## Extending the wrapper

The wrapper is intentionally dependency-free so it can be embedded in SDK
packages without pulling in NuGet dependencies.  Ergonomic conveniences that
require NuGet packages (e.g. `DenseTensor<float>` from `System.Numerics.Tensors`,
`OrtValue`-style named-input helpers) belong in the example or application layer,
not here.

When adding new `sdxGet*` / `sdxSet*` functions from `dsp_runtime_c.h`:

1. Add a `[DllImport]` entry to `NativeMethods`.
2. Add a typed public method to `SdxContext` (or `SdxRuntime`/`SdxModel` as appropriate)
   with `<summary>` XML doc and `<param>` tags.
3. Throw `ObjectDisposedException` when `_handle == IntPtr.Zero`.
4. Delegate error handling through `ThrowOnError`.
5. Update this README and the example layer README.
