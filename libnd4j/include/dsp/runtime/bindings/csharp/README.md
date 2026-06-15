# SDX C# Binding

File:

- `SdxRuntime.cs`

This wrapper provides:

- `SdxRuntime` -> runtime lifecycle
- `SdxModel` -> model handle
- `SdxContext` -> execution context
- `SdxTensorViewLease` -> helper for unmanaged shape buffers

Notes:

- Requires .NET runtime with `NativeLibrary` support (for dynamic library resolution).
- `SdxRuntime.Create()` auto-resolves `nd4jcpu`, `nd4jcuda`, then `nd4jamd`.
- Use `SdxRuntime.Create("...")` to force a specific library path/name.
- `SdxModel.CreateContext(...)` accepts optional requested output names.
