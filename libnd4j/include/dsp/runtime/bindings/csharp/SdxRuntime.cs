using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

#nullable enable

namespace Nd4j.Dsp.Runtime
{

/// <summary>Integer constants mirroring the SDX C ABI enumerations.</summary>
public static class SdxConstants
{
    // sdx_status_t
    public const int SDX_STATUS_OK = 0;
    public const int SDX_STATUS_INVALID_ARGUMENT = 1;
    public const int SDX_STATUS_INCOMPATIBLE_ABI = 2;
    public const int SDX_STATUS_MODEL_LOAD_FAILED = 3;
    public const int SDX_STATUS_EXECUTION_FAILED = 4;
    public const int SDX_STATUS_BACKEND_UNAVAILABLE = 5;
    public const int SDX_STATUS_IO_ERROR = 6;
    public const int SDX_STATUS_UNSUPPORTED = 7;

    // sdx_backend_t
    public const int SDX_BACKEND_AUTO = 0;
    public const int SDX_BACKEND_SLOT_BY_SLOT = 1;
    public const int SDX_BACKEND_CUDA_GRAPHS = 2;
    public const int SDX_BACKEND_NVRTC = 3;
    public const int SDX_BACKEND_PTX = 4;
    public const int SDX_BACKEND_TRITON = 5;
    public const int SDX_BACKEND_MLX = 6;
    public const int SDX_BACKEND_ARM_HYBRID = 7;
    public const int SDX_BACKEND_NNAPI = 8;
    public const int SDX_BACKEND_HIP_GRAPHS = 9;
    public const int SDX_BACKEND_LEVEL_ZERO = 10;
    public const int SDX_BACKEND_VULKAN = 11;
    public const int SDX_BACKEND_METAL = 12;
    public const int SDX_BACKEND_TPU = 13;
    public const int SDX_BACKEND_HEXAGON = 14;

    // sdx_device_type_t
    public const int SDX_DEVICE_HOST = 0;
    public const int SDX_DEVICE_CUDA = 1;
    public const int SDX_DEVICE_AMD = 2;

    // sdx_gpu_target_t
    public const int SDX_GPU_TARGET_AUTO = 0;
    public const int SDX_GPU_TARGET_CUDA = 1;
    public const int SDX_GPU_TARGET_AMD = 2;
    public const int SDX_GPU_TARGET_VULKAN = 3;
    public const int SDX_GPU_TARGET_METAL = 4;

    // sdxGetPlanPhase return values
    public const int SDX_PHASE_SLOT_BY_SLOT = 0;
    public const int SDX_PHASE_SHAPES_FROZEN = 1;
    public const int SDX_PHASE_REPLAYING = 2;
    public const int SDX_PHASE_REPLAY_BLOCKED = 3;

    /// <summary>ABI version this wrapper targets.</summary>
    public const int SDX_RUNTIME_ABI_VERSION = 1;
}

/// <summary>
/// Options passed to <see cref="SdxRuntime.Create"/>.
/// Populate via <see cref="Default"/> and override only the fields you need.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct SdxRuntimeOptions
{
    /// <summary>Must be set to <c>sizeof(sdx_runtime_options_t)</c> for ABI versioning.</summary>
    public uint struct_size;

    /// <summary>Returns an options struct with all fields set to their defaults.</summary>
    public static SdxRuntimeOptions Default()
    {
        return new SdxRuntimeOptions
        {
            struct_size = (uint)Marshal.SizeOf<SdxRuntimeOptions>()
        };
    }
}

/// <summary>
/// Options passed to <see cref="SdxRuntime.LoadModel"/>.
/// Populate via <see cref="Default"/> and override only the fields you need.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct SdxModelOptions
{
    /// <summary>Must be set to <c>sizeof(sdx_model_options_t)</c> for ABI versioning.</summary>
    public uint struct_size;
    /// <summary>Preferred execution backend; one of the <c>SDX_BACKEND_*</c> constants.</summary>
    public int backend;
    /// <summary>When non-zero, fail if the requested backend is unavailable instead of falling back.</summary>
    public int strict_backend;
    /// <summary>When non-zero, allow the runtime to JIT-compile kernels at execution time.</summary>
    public int allow_runtime_jit;
    /// <summary>Preferred GPU target; one of the <c>SDX_GPU_TARGET_*</c> constants.</summary>
    public int gpu_target;

    /// <summary>Returns an options struct with all fields set to their defaults.</summary>
    public static SdxModelOptions Default()
    {
        return new SdxModelOptions
        {
            struct_size = (uint)Marshal.SizeOf<SdxModelOptions>(),
            backend = SdxConstants.SDX_BACKEND_AUTO,
            strict_backend = 0,
            allow_runtime_jit = 0,
            gpu_target = SdxConstants.SDX_GPU_TARGET_AUTO
        };
    }
}

/// <summary>Options controlling context input binding.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct SdxContextOptions
{
    public uint struct_size;
    /// <summary>When non-zero, bind bundle-owned constants and variables internally.</summary>
    public int bind_model_parameters;

    public static SdxContextOptions Inference()
    {
        return new SdxContextOptions
        {
            struct_size = (uint)Marshal.SizeOf<SdxContextOptions>(),
            bind_model_parameters = 1
        };
    }
}

/// <summary>
/// Per-invocation options passed to <see cref="SdxContext.Run"/>.
/// Populate via <see cref="Default"/> and override only the fields you need.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct SdxRunOptions
{
    /// <summary>Must be set to <c>sizeof(sdx_run_options_t)</c> for ABI versioning.</summary>
    public uint struct_size;
    /// <summary>Backend override for this invocation; one of the <c>SDX_BACKEND_*</c> constants.</summary>
    public int backend;
    /// <summary>When non-zero, reject calls whose input/output count does not match the plan contract.</summary>
    public int strict_signature;
    /// <summary>GPU target override; one of the <c>SDX_GPU_TARGET_*</c> constants.</summary>
    public int gpu_target;

    /// <summary>Returns an options struct with all fields set to their defaults.</summary>
    public static SdxRunOptions Default()
    {
        return new SdxRunOptions
        {
            struct_size = (uint)Marshal.SizeOf<SdxRunOptions>(),
            backend = SdxConstants.SDX_BACKEND_AUTO,
            strict_signature = 1,
            gpu_target = SdxConstants.SDX_GPU_TARGET_AUTO
        };
    }
}

/// <summary>
/// Non-owning view of a tensor buffer, passed to <see cref="SdxContext.Run"/>.
/// The caller is responsible for keeping the underlying buffers alive for the
/// duration of the <c>Run</c> call.  Use <see cref="SdxTensorViewLease"/> to
/// manage the unmanaged shape allocation and <see cref="GCHandle"/> pinning for
/// managed data buffers.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct SdxTensorView
{
    /// <summary>Pointer to the tensor element data.</summary>
    public IntPtr data;
    /// <summary>Pointer to an array of <c>int64_t[rank]</c> dimension sizes.</summary>
    public IntPtr shape;
    /// <summary>Number of dimensions.</summary>
    public int rank;
    /// <summary>Element data type (<c>sd::DataType</c> ordinal; 5 = FLOAT32).</summary>
    public int dtype;
    /// <summary>Total size of the data buffer in bytes.</summary>
    public UIntPtr bytes;
    /// <summary>Memory location; one of the <c>SDX_DEVICE_*</c> constants.</summary>
    public int device_type;
    /// <summary>Device ordinal for GPU tensors; -1 for host tensors.</summary>
    public int device_id;
}

/// <summary>Managed copy of one runtime-allocated dynamic output.</summary>
public sealed class SdxDynamicOutput
{
    public byte[] Data { get; }
    public long[] Shape { get; }
    public int DType { get; }
    public string? Name { get; }

    internal SdxDynamicOutput(byte[] data, long[] shape, int dtype, string? name)
    {
        Data = data;
        Shape = shape;
        DType = dtype;
        Name = name;
    }
}

/// <summary>
/// Telemetry snapshot returned by <see cref="SdxContext.ExecutionReport"/>.
/// The struct is populated by <c>sdxGetExecutionReport</c> after at least one
/// successful <see cref="SdxContext.Run"/> call.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct SdxExecutionReport
{
    /// <summary>Must be set to <c>sizeof(sdx_execution_report_t)</c> before calling <c>sdxGetExecutionReport</c>.</summary>
    public uint struct_size;
    /// <summary>Backend that was requested; one of the <c>SDX_BACKEND_*</c> constants.</summary>
    public int requested_backend;
    /// <summary>Backend that was actually applied after clamping and plan-side mode changes.</summary>
    public int applied_backend;
    /// <summary>Status code from the last execution; <see cref="SdxConstants.SDX_STATUS_OK"/> on success.</summary>
    public int status_code;
    /// <summary>
    /// 1 = fallback was observed (a segment failed graph capture, or the plan is REPLAY_BLOCKED);
    /// 0 = the requested/optimal path is in force; -1 = unknown (no execution yet).
    /// </summary>
    public int used_fallback;
    /// <summary>Wall-clock duration of the last <c>sdxRun</c> call in nanoseconds.</summary>
    public ulong execution_time_ns;
    /// <summary>GPU target that was requested; one of the <c>SDX_GPU_TARGET_*</c> constants.</summary>
    public int requested_gpu_target;
    /// <summary>GPU target that was actually applied.</summary>
    public int applied_gpu_target;
    /// <summary>
    /// Plan phase after execution:
    /// 0 = <c>SLOT_BY_SLOT</c> (warmup), 1 = <c>SHAPES_FROZEN</c>,
    /// 2 = <c>REPLAYING</c>, 3 = <c>REPLAY_BLOCKED</c>.
    /// </summary>
    public int plan_phase;
    /// <summary>Total number of executions completed on this context.</summary>
    public int execution_count;

    /// <summary>Returns a report struct with <see cref="struct_size"/> pre-filled and all other fields zeroed.</summary>
    public static SdxExecutionReport Default()
    {
        return new SdxExecutionReport
        {
            struct_size = (uint)Marshal.SizeOf<SdxExecutionReport>()
        };
    }
}

/// <summary>
/// Manages the unmanaged memory that holds the shape array for an
/// <see cref="SdxTensorView"/>.  The caller is still responsible for keeping
/// the <em>data</em> pointer alive (e.g. via a <see cref="GCHandle"/> pin) for
/// the duration of any <see cref="SdxContext.Run"/> call that uses the view.
/// </summary>
/// <remarks>
/// Typical usage:
/// <code>
/// var pin  = GCHandle.Alloc(data, GCHandleType.Pinned);
/// using var lease = SdxTensorViewLease.CreateHost(
///     pin.AddrOfPinnedObject(), shape, dtype,
///     (UIntPtr)(data.Length * sizeof(float)));
/// ctx.Run(new[] { lease.View }, outputs, null);
/// pin.Free();
/// </code>
/// </remarks>
public sealed class SdxTensorViewLease : IDisposable
{
    /// <summary>The <see cref="SdxTensorView"/> whose <c>shape</c> pointer is owned by this lease.</summary>
    public SdxTensorView View;
    private readonly IntPtr _shapePtr;
    private bool _disposed;

    private SdxTensorViewLease(SdxTensorView view, IntPtr shapePtr)
    {
        View = view;
        _shapePtr = shapePtr;
    }

    /// <summary>
    /// Creates a host-side tensor view and allocates unmanaged memory for the shape array.
    /// </summary>
    /// <param name="data">Pinned pointer to the element data.</param>
    /// <param name="shape">Dimension sizes in row-major order.</param>
    /// <param name="dtype">Element data type (<c>sd::DataType</c> ordinal; 5 = FLOAT32).</param>
    /// <param name="bytes">Total size of the element data in bytes.</param>
    /// <returns>A lease that must be disposed after the <c>Run</c> call completes.</returns>
    public static SdxTensorViewLease CreateHost(IntPtr data, long[] shape, int dtype, UIntPtr bytes)
    {
        IntPtr shapePtr = IntPtr.Zero;
        if (shape.Length > 0)
        {
            shapePtr = Marshal.AllocHGlobal(sizeof(long) * shape.Length);
            Marshal.Copy(shape, 0, shapePtr, shape.Length);
        }

        var view = new SdxTensorView
        {
            data = data,
            shape = shapePtr,
            rank = shape.Length,
            dtype = dtype,
            bytes = bytes,
            device_type = SdxConstants.SDX_DEVICE_HOST,
            device_id = -1
        };

        return new SdxTensorViewLease(view, shapePtr);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_shapePtr != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(_shapePtr);
        }
    }
}

internal static class NativeMethods
{
    private const string ImportLibraryName = "nd4jruntime";
    private static string _preferredLibrary = string.Empty;

    static NativeMethods()
    {
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, ResolveLibrary);
    }

    internal static void SetPreferredLibrary(string? libraryNameOrPath)
    {
        _preferredLibrary = libraryNameOrPath ?? string.Empty;
    }

    private static IntPtr ResolveLibrary(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, ImportLibraryName, StringComparison.Ordinal))
        {
            return IntPtr.Zero;
        }

        if (!string.IsNullOrWhiteSpace(_preferredLibrary) &&
            NativeLibrary.TryLoad(_preferredLibrary, assembly, searchPath, out var preferredHandle))
        {
            return preferredHandle;
        }

        // Try SDK-relative path: ../../lib/ from assembly location
        var assemblyDir = Path.GetDirectoryName(typeof(NativeMethods).Assembly.Location);
        if (!string.IsNullOrEmpty(assemblyDir))
        {
            var sdkLib = Path.GetFullPath(Path.Combine(assemblyDir, "..", "..", "lib"));
            if (Directory.Exists(sdkLib))
            {
                foreach (var candidate in DefaultLibraryCandidates())
                {
                    if (NativeLibrary.TryLoad(Path.Combine(sdkLib, candidate), out var sdkHandle))
                    {
                        return sdkHandle;
                    }
                }
            }
        }

        foreach (var candidate in DefaultLibraryCandidates())
        {
            if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out var handle))
            {
                return handle;
            }
        }

        throw new DllNotFoundException(
            $"Unable to load SDX runtime library. Preferred='{_preferredLibrary}'. Tried defaults: " +
            string.Join(", ", DefaultLibraryCandidates()));
    }

    private static IEnumerable<string> DefaultLibraryCandidates()
    {
        // Standalone SDX runtimes first (JVM-free, sdx*-only exports).
        yield return "sdx_cpu";
        yield return "sdx_cuda";
        yield return "sdx_vulkan";
        yield return "sdx_metal";
        yield return "libsdx_cpu.so";
        yield return "libsdx_cuda.so";
        yield return "libsdx_vulkan.so";
        yield return "libsdx_metal.so";
        yield return "libsdx_cpu.dylib";
        yield return "libsdx_cuda.dylib";
        yield return "libsdx_vulkan.dylib";
        yield return "libsdx_metal.dylib";
        yield return "sdx_cpu.dll";
        yield return "sdx_cuda.dll";
        // Monolithic backend libraries export the same sdx* C ABI.
        yield return "nd4jvulkan";
        yield return "nd4jmetal";
        yield return "nd4jcpu";
        yield return "nd4jcuda";
        yield return "nd4jamd";
        yield return "libnd4jcpu.so";
        yield return "libnd4jcuda.so";
        yield return "libnd4jamd.so";
        yield return "libnd4jcpu.dylib";
        yield return "libnd4jcuda.dylib";
        yield return "libnd4jamd.dylib";
        yield return "nd4jcpu.dll";
        yield return "nd4jcuda.dll";
        yield return "nd4jamd.dll";
    }

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxGetRuntimeAbiVersion();

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxCreateRuntime(ref SdxRuntimeOptions options, out IntPtr outRuntime);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void sdxDestroyRuntime(IntPtr runtime);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxLoadBundle(IntPtr runtime, string bundlePath, ref SdxModelOptions options, out IntPtr outModel);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void sdxUnloadModel(IntPtr model);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr sdxGetTokenizerPath(IntPtr model);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr sdxGetTextGenerationConfigPath(IntPtr model);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxCreateContext(
        IntPtr model,
        IntPtr requestedOutputNames,
        int numRequestedOutputs,
        out IntPtr outContext);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxCreateContextWithOptions(
        IntPtr model,
        IntPtr requestedOutputNames,
        int numRequestedOutputs,
        ref SdxContextOptions options,
        out IntPtr outContext);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void sdxDestroyContext(IntPtr context);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxRun(
        IntPtr context,
        [In] SdxTensorView[] inputs,
        int numInputs,
        [In] SdxTensorView[] outputs,
        int numOutputs,
        ref SdxRunOptions options);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxRunAllocating(
        IntPtr context,
        [In] SdxTensorView[] inputs,
        int numInputs,
        ref SdxRunOptions options);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr sdxGetLastError(IntPtr runtime);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxGetExecutionReport(IntPtr context, ref SdxExecutionReport report);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxMarkInputVariable(IntPtr context, int inputIndex);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxMarkInputPlaceholder(IntPtr context, int inputIndex);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxFreezeShapes(IntPtr context);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxGetPlanPhase(IntPtr context);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxGetExecutionCount(IntPtr context);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxGetNumInputs(IntPtr context);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxGetNumOutputs(IntPtr context);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr sdxGetInputName(IntPtr context, int inputIndex);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr sdxGetOutputName(IntPtr context, int outputIndex);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxGetOutputTensor(
        IntPtr context, int outputIndex, out SdxTensorView output);
}

/// <summary>
/// Entry point for the SDX runtime.  Create one instance per process via
/// <see cref="Create"/>, load models with <see cref="LoadModel"/>, and dispose
/// when done.
/// </summary>
public sealed class SdxRuntime : IDisposable
{
    private IntPtr _handle;

    private SdxRuntime(IntPtr handle)
    {
        _handle = handle;
    }

    /// <summary>
    /// Creates the native SDX runtime and returns a managed wrapper.
    /// </summary>
    /// <param name="libraryNameOrPath">
    /// Optional path or name of the native library to load.  When <see langword="null"/>
    /// the wrapper probes a set of well-known names (see <c>NativeMethods</c>).
    /// </param>
    /// <returns>A new <see cref="SdxRuntime"/> instance.  Dispose when done.</returns>
    /// <exception cref="InvalidOperationException">Thrown when <c>sdxCreateRuntime</c> returns a non-OK status.</exception>
    public static SdxRuntime Create(string? libraryNameOrPath = null)
    {
        NativeMethods.SetPreferredLibrary(libraryNameOrPath);
        var options = SdxRuntimeOptions.Default();
        var status = NativeMethods.sdxCreateRuntime(ref options, out var handle);
        if (status != SdxConstants.SDX_STATUS_OK)
        {
            throw new InvalidOperationException($"sdxCreateRuntime failed: status={status}");
        }

        return new SdxRuntime(handle);
    }

    /// <summary>Returns the ABI version reported by the loaded native library.</summary>
    public int AbiVersion() => NativeMethods.sdxGetRuntimeAbiVersion();

    /// <summary>
    /// Loads an SDZ model bundle and returns a <see cref="SdxModel"/> handle.
    /// </summary>
    /// <param name="bundlePath">Path to the <c>.sdz</c> file.</param>
    /// <param name="options">Model load options; defaults to <see cref="SdxModelOptions.Default"/> when <see langword="null"/>.</param>
    /// <exception cref="InvalidOperationException">Thrown when loading fails.</exception>
    /// <exception cref="ObjectDisposedException">Thrown when this runtime has been disposed.</exception>
    public SdxModel LoadModel(string bundlePath, SdxModelOptions? options = null)
    {
        EnsureOpen();
        var opts = options ?? SdxModelOptions.Default();
        var status = NativeMethods.sdxLoadBundle(_handle, bundlePath, ref opts, out var modelHandle);
        ThrowOnError(status, "sdxLoadBundle");
        return new SdxModel(this, modelHandle);
    }

    /// <summary>Returns the last error message from the native runtime, or an empty string.</summary>
    public string LastError()
    {
        EnsureOpen();
        var ptr = NativeMethods.sdxGetLastError(_handle);
        return ptr == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
    }

    internal void ThrowOnError(int status, string op)
    {
        if (status == SdxConstants.SDX_STATUS_OK) return;
        throw new InvalidOperationException($"{op} failed: status={status}, error={LastError()}");
    }

    private void EnsureOpen()
    {
        if (_handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(SdxRuntime));
    }

    internal IntPtr Handle => _handle;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            NativeMethods.sdxDestroyRuntime(_handle);
            _handle = IntPtr.Zero;
        }
    }
}

/// <summary>
/// A loaded SDZ model bundle.  Create via <see cref="SdxRuntime.LoadModel"/>.
/// One model can produce multiple independent execution contexts.
/// </summary>
public sealed class SdxModel : IDisposable
{
    private readonly SdxRuntime _runtime;
    private IntPtr _handle;

    internal SdxModel(SdxRuntime runtime, IntPtr handle)
    {
        _runtime = runtime;
        _handle = handle;
    }

    /// <summary>Resolved offline tokenizer path declared by the bundle.</summary>
    public string? TokenizerPath
    {
        get
        {
            var ptr = NativeMethods.sdxGetTokenizerPath(_handle);
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }
    }

    /// <summary>Resolved text-generation metadata path declared by the bundle.</summary>
    public string? TextGenerationConfigPath
    {
        get
        {
            var ptr = NativeMethods.sdxGetTextGenerationConfigPath(_handle);
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }
    }

    /// <summary>
    /// Creates an execution context bound to this model.
    /// </summary>
    /// <param name="requestedOutputs">
    /// Names of the outputs to materialise per <c>Run</c> call.  Pass
    /// <see langword="null"/> (the default) to materialise all plan outputs.
    /// </param>
    /// <returns>A new <see cref="SdxContext"/>.  Dispose when done.</returns>
    /// <exception cref="ObjectDisposedException">Thrown when this model has been disposed.</exception>
    /// <exception cref="InvalidOperationException">Thrown when context creation fails.</exception>
    public SdxContext CreateContext(
        IReadOnlyList<string>? requestedOutputs = null,
        bool bindModelParameters = false)
    {
        if (_handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(SdxModel));

        IntPtr namesBuffer = IntPtr.Zero;
        var encodedNames = new List<IntPtr>();

        try
        {
            var outputCount = requestedOutputs?.Count ?? 0;
            if (outputCount > 0)
            {
                namesBuffer = Marshal.AllocHGlobal(IntPtr.Size * outputCount);
                for (var i = 0; i < outputCount; i++)
                {
                    var namePtr = AllocUtf8CString(requestedOutputs![i]);
                    encodedNames.Add(namePtr);
                    Marshal.WriteIntPtr(namesBuffer, i * IntPtr.Size, namePtr);
                }
            }

            IntPtr ctxHandle;
            int status;
            if (bindModelParameters)
            {
                var options = SdxContextOptions.Inference();
                status = NativeMethods.sdxCreateContextWithOptions(
                    _handle, namesBuffer, outputCount, ref options, out ctxHandle);
                _runtime.ThrowOnError(status, "sdxCreateContextWithOptions");
            }
            else
            {
                status = NativeMethods.sdxCreateContext(
                    _handle, namesBuffer, outputCount, out ctxHandle);
                _runtime.ThrowOnError(status, "sdxCreateContext");
            }
            return new SdxContext(_runtime, this, ctxHandle);
        }
        finally
        {
            foreach (var ptr in encodedNames) Marshal.FreeHGlobal(ptr);
            if (namesBuffer != IntPtr.Zero) Marshal.FreeHGlobal(namesBuffer);
        }
    }

    /// <summary>
    /// Creates the recommended mobile/offline context with bundle-owned model
    /// parameters bound internally.
    /// </summary>
    public SdxContext CreateInferenceContext(IReadOnlyList<string>? requestedOutputs = null)
    {
        return CreateContext(requestedOutputs, bindModelParameters: true);
    }

    private static IntPtr AllocUtf8CString(string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value + '\0');
        var ptr = Marshal.AllocHGlobal(bytes.Length);
        Marshal.Copy(bytes, 0, ptr, bytes.Length);
        return ptr;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            NativeMethods.sdxUnloadModel(_handle);
            _handle = IntPtr.Zero;
        }
    }
}

/// <summary>
/// Execution context for a loaded model.  Owns a compiled DSP plan and
/// maintains execution state across repeated <see cref="Run"/> calls.
/// Create via <see cref="SdxModel.CreateContext"/>.
/// </summary>
/// <remarks>
/// Typical lifecycle:
/// <list type="number">
/// <item>Optionally call <see cref="MarkInputVariable"/> / <see cref="MarkInputPlaceholder"/> for inputs that change.</item>
/// <item>Run a few warmup calls via <see cref="Run"/>.</item>
/// <item>Call <see cref="FreezeShapes"/> to enable CUDA graph capture and the stable fast path.</item>
/// <item>Continue calling <see cref="Run"/> — the plan transitions to <c>REPLAYING</c>.</item>
/// <item>Inspect telemetry with <see cref="ExecutionReport"/>.</item>
/// </list>
/// </remarks>
public sealed class SdxContext : IDisposable
{
    private readonly SdxRuntime _runtime;
    // Retains bundle-owned arrays while a parameter-bound context is alive.
    private readonly SdxModel _model;
    private IntPtr _handle;

    internal SdxContext(SdxRuntime runtime, SdxModel model, IntPtr handle)
    {
        _runtime = runtime;
        _model = model;
        _handle = handle;
    }

    /// <summary>
    /// Executes the plan with the supplied input and output tensor views.
    /// </summary>
    /// <param name="inputs">
    /// Views for all external inputs in plan binding order.  The count must
    /// match <see cref="NumInputs"/> when <c>strict_signature</c> is set in options.
    /// </param>
    /// <param name="outputs">
    /// Views for all requested outputs.  The caller must pre-allocate the output
    /// buffers; the plan writes results in-place.
    /// </param>
    /// <param name="options">Per-call run options; defaults to <see cref="SdxRunOptions.Default"/> when <see langword="null"/>.</param>
    /// <exception cref="ObjectDisposedException">Thrown when this context has been disposed.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the native call returns a non-OK status.</exception>
    public void Run(SdxTensorView[] inputs, SdxTensorView[] outputs, SdxRunOptions? options = null)
    {
        if (_handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(SdxContext));

        var opts = options ?? SdxRunOptions.Default();
        var status = NativeMethods.sdxRun(_handle, inputs, inputs.Length, outputs, outputs.Length, ref opts);
        _runtime.ThrowOnError(status, "sdxRun");
    }

    /// <summary>
    /// Executes with runtime-owned dynamic outputs and returns managed copies.
    /// </summary>
    public SdxDynamicOutput[] RunAllocating(
        SdxTensorView[] inputs, SdxRunOptions? options = null)
    {
        if (_handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(SdxContext));

        var opts = options ?? SdxRunOptions.Default();
        var status = NativeMethods.sdxRunAllocating(
            _handle, inputs, inputs.Length, ref opts);
        _runtime.ThrowOnError(status, "sdxRunAllocating");

        var count = Math.Max(0, NumOutputs());
        var results = new SdxDynamicOutput[count];
        for (var index = 0; index < count; ++index)
        {
            status = NativeMethods.sdxGetOutputTensor(
                _handle, index, out var view);
            _runtime.ThrowOnError(status, "sdxGetOutputTensor");

            if (view.rank < 0)
                throw new InvalidOperationException(
                    "sdxGetOutputTensor returned a negative rank");
            var shape = new long[view.rank];
            if (shape.Length > 0)
            {
                if (view.shape == IntPtr.Zero)
                    throw new InvalidOperationException(
                        "sdxGetOutputTensor returned a null shape");
                Marshal.Copy(view.shape, shape, 0, shape.Length);
            }

            var byteCount64 = view.bytes.ToUInt64();
            if (byteCount64 > int.MaxValue)
                throw new InvalidOperationException(
                    "Dynamic output is too large for a managed byte array");
            var data = new byte[(int)byteCount64];
            if (data.Length > 0)
            {
                if (view.data == IntPtr.Zero)
                    throw new InvalidOperationException(
                        "sdxGetOutputTensor returned a null data pointer");
                Marshal.Copy(view.data, data, 0, data.Length);
            }
            results[index] = new SdxDynamicOutput(
                data, shape, view.dtype, OutputName(index));
        }
        return results;
    }

    /// <summary>
    /// Returns a telemetry snapshot for the most recent <see cref="Run"/> call.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the native call returns a non-OK status.</exception>
    public SdxExecutionReport ExecutionReport()
    {
        var report = SdxExecutionReport.Default();
        var status = NativeMethods.sdxGetExecutionReport(_handle, ref report);
        _runtime.ThrowOnError(status, "sdxGetExecutionReport");
        return report;
    }

    /// <summary>
    /// Marks an input as VARIABLE: its value changes between runs but its shape
    /// stays fixed.  Enables device-to-device staging buffers for that input.
    /// Call after context creation, before the first <see cref="Run"/>.
    /// </summary>
    /// <param name="inputIndex">Zero-based plan input index (see <see cref="InputNames"/>).</param>
    public void MarkInputVariable(int inputIndex)
    {
        var status = NativeMethods.sdxMarkInputVariable(_handle, inputIndex);
        _runtime.ThrowOnError(status, "sdxMarkInputVariable");
    }

    /// <summary>
    /// Marks an input as PLACEHOLDER: both its value and potentially its shape
    /// may change between runs.  Placeholders always receive a host-to-device sync.
    /// Call after context creation, before the first <see cref="Run"/>.
    /// </summary>
    /// <param name="inputIndex">Zero-based plan input index (see <see cref="InputNames"/>).</param>
    public void MarkInputPlaceholder(int inputIndex)
    {
        var status = NativeMethods.sdxMarkInputPlaceholder(_handle, inputIndex);
        _runtime.ThrowOnError(status, "sdxMarkInputPlaceholder");
    }

    /// <summary>
    /// Freezes plan shapes, enabling CUDA graph capture and the argument-table
    /// stable fast path.  Typically called after a few warmup <see cref="Run"/>
    /// calls to allow shapes to stabilise.
    /// </summary>
    public void FreezeShapes()
    {
        var status = NativeMethods.sdxFreezeShapes(_handle);
        _runtime.ThrowOnError(status, "sdxFreezeShapes");
    }

    /// <summary>
    /// Current plan execution phase.
    /// Returns <see cref="SdxConstants.SDX_PHASE_SLOT_BY_SLOT"/> (0) during warmup,
    /// <see cref="SdxConstants.SDX_PHASE_SHAPES_FROZEN"/> (1) after <see cref="FreezeShapes"/>,
    /// <see cref="SdxConstants.SDX_PHASE_REPLAYING"/> (2) once graph capture completes,
    /// <see cref="SdxConstants.SDX_PHASE_REPLAY_BLOCKED"/> (3) if capture failed,
    /// or -1 on error.
    /// </summary>
    public int PlanPhase() => NativeMethods.sdxGetPlanPhase(_handle);

    /// <summary>
    /// Total number of <see cref="Run"/> calls completed on this context, or -1 on error.
    /// </summary>
    public int ExecutionCount() => NativeMethods.sdxGetExecutionCount(_handle);

    /// <summary>
    /// Number of external inputs the plan expects per <see cref="Run"/> call, or -1 on error.
    /// External inputs cover the model's constants, variables, AND placeholders —
    /// all must be bound positionally in plan order.
    /// </summary>
    public int NumInputs() => NativeMethods.sdxGetNumInputs(_handle);

    /// <summary>Number of outputs the plan produces per <see cref="Run"/> call, or -1 on error.</summary>
    public int NumOutputs() => NativeMethods.sdxGetNumOutputs(_handle);

    /// <summary>
    /// Returns the name of the external input at the given plan index, or
    /// <see langword="null"/> for out-of-range indices or on error.
    /// </summary>
    /// <param name="inputIndex">Zero-based plan input index.</param>
    public string? InputName(int inputIndex)
    {
        var ptr = NativeMethods.sdxGetInputName(_handle, inputIndex);
        return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
    }

    /// <summary>
    /// Returns all external input names in plan binding order.
    /// Use the returned array to build the positional <see cref="SdxTensorView"/>
    /// arrays expected by <see cref="Run"/>.
    /// </summary>
    public string?[] InputNames()
    {
        var count = NumInputs();
        if (count <= 0) return Array.Empty<string?>();

        var names = new string?[count];
        for (var i = 0; i < count; i++) names[i] = InputName(i);
        return names;
    }

    public string? OutputName(int outputIndex)
    {
        var ptr = NativeMethods.sdxGetOutputName(_handle, outputIndex);
        return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
    }

    public string?[] OutputNames()
    {
        var count = NumOutputs();
        if (count <= 0) return Array.Empty<string?>();
        var names = new string?[count];
        for (var i = 0; i < count; ++i) names[i] = OutputName(i);
        return names;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            NativeMethods.sdxDestroyContext(_handle);
            _handle = IntPtr.Zero;
        }
    }
}
}
