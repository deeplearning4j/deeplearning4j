using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Nd4j.Dsp.Runtime
{

public static class SdxConstants
{
    public const int SDX_STATUS_OK = 0;

    public const int SDX_BACKEND_AUTO = 0;
    public const int SDX_BACKEND_SLOT_BY_SLOT = 1;
    public const int SDX_BACKEND_CUDA_GRAPHS = 2;
    public const int SDX_BACKEND_NVRTC = 3;
    public const int SDX_BACKEND_PTX = 4;
    public const int SDX_BACKEND_TRITON = 5;
    public const int SDX_BACKEND_MLX = 6;
    public const int SDX_BACKEND_ARM_HYBRID = 7;
    public const int SDX_BACKEND_NNAPI = 8;

    public const int SDX_DEVICE_HOST = 0;
    public const int SDX_DEVICE_CUDA = 1;
    public const int SDX_DEVICE_AMD = 2;

    public const int SDX_GPU_TARGET_AUTO = 0;
    public const int SDX_GPU_TARGET_CUDA = 1;
    public const int SDX_GPU_TARGET_AMD = 2;
}

[StructLayout(LayoutKind.Sequential)]
public struct SdxRuntimeOptions
{
    public uint struct_size;

    public static SdxRuntimeOptions Default()
    {
        return new SdxRuntimeOptions
        {
            struct_size = (uint)Marshal.SizeOf<SdxRuntimeOptions>()
        };
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct SdxModelOptions
{
    public uint struct_size;
    public int backend;
    public int strict_backend;
    public int allow_runtime_jit;
    public int gpu_target;

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

[StructLayout(LayoutKind.Sequential)]
public struct SdxRunOptions
{
    public uint struct_size;
    public int backend;
    public int strict_signature;
    public int gpu_target;

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

[StructLayout(LayoutKind.Sequential)]
public struct SdxTensorView
{
    public IntPtr data;
    public IntPtr shape;
    public int rank;
    public int dtype;
    public UIntPtr bytes;
    public int device_type;
    public int device_id;
}

[StructLayout(LayoutKind.Sequential)]
public struct SdxExecutionReport
{
    public uint struct_size;
    public int requested_backend;
    public int applied_backend;
    public int status_code;
    public int used_fallback;
    public ulong execution_time_ns;
    public int requested_gpu_target;
    public int applied_gpu_target;

    public static SdxExecutionReport Default()
    {
        return new SdxExecutionReport
        {
            struct_size = (uint)Marshal.SizeOf<SdxExecutionReport>()
        };
    }
}

public sealed class SdxTensorViewLease : IDisposable
{
    public SdxTensorView View;
    private readonly IntPtr _shapePtr;

    private SdxTensorViewLease(SdxTensorView view, IntPtr shapePtr)
    {
        View = view;
        _shapePtr = shapePtr;
    }

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

    public void Dispose()
    {
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

    internal static void SetPreferredLibrary(string libraryNameOrPath)
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
    internal static extern int sdxCreateContext(
        IntPtr model,
        IntPtr requestedOutputNames,
        int numRequestedOutputs,
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
    internal static extern IntPtr sdxGetLastError(IntPtr runtime);

    [DllImport(ImportLibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxGetExecutionReport(IntPtr context, ref SdxExecutionReport report);
}

public sealed class SdxRuntime : IDisposable
{
    private IntPtr _handle;

    private SdxRuntime(IntPtr handle)
    {
        _handle = handle;
    }

    public static SdxRuntime Create(string libraryNameOrPath = null)
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

    public int AbiVersion() => NativeMethods.sdxGetRuntimeAbiVersion();

    public SdxModel LoadModel(string bundlePath, SdxModelOptions? options = null)
    {
        EnsureOpen();
        var opts = options ?? SdxModelOptions.Default();
        var status = NativeMethods.sdxLoadBundle(_handle, bundlePath, ref opts, out var modelHandle);
        ThrowOnError(status, "sdxLoadBundle");
        return new SdxModel(this, modelHandle);
    }

    public string LastError()
    {
        EnsureOpen();
        var ptr = NativeMethods.sdxGetLastError(_handle);
        return PtrToStringUtf8(ptr);
    }

    internal void ThrowOnError(int status, string op)
    {
        if (status == SdxConstants.SDX_STATUS_OK)
        {
            return;
        }

        throw new InvalidOperationException($"{op} failed: status={status}, error={LastError()}");
    }

    private void EnsureOpen()
    {
        if (_handle == IntPtr.Zero)
        {
            throw new ObjectDisposedException(nameof(SdxRuntime));
        }
    }

    internal IntPtr Handle => _handle;

    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            NativeMethods.sdxDestroyRuntime(_handle);
            _handle = IntPtr.Zero;
        }
    }

    private static string PtrToStringUtf8(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero)
        {
            return string.Empty;
        }

        var bytes = new List<byte>();
        var offset = 0;
        while (true)
        {
            var value = Marshal.ReadByte(ptr, offset++);
            if (value == 0)
            {
                break;
            }
            bytes.Add(value);
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }
}

public sealed class SdxModel : IDisposable
{
    private readonly SdxRuntime _runtime;
    private IntPtr _handle;

    internal SdxModel(SdxRuntime runtime, IntPtr handle)
    {
        _runtime = runtime;
        _handle = handle;
    }

    public SdxContext CreateContext(IReadOnlyList<string> requestedOutputs = null)
    {
        if (_handle == IntPtr.Zero)
        {
            throw new ObjectDisposedException(nameof(SdxModel));
        }

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
                    var namePtr = AllocUtf8CString(requestedOutputs[i]);
                    encodedNames.Add(namePtr);
                    Marshal.WriteIntPtr(namesBuffer, i * IntPtr.Size, namePtr);
                }
            }

            var status = NativeMethods.sdxCreateContext(_handle, namesBuffer, outputCount, out var ctxHandle);
            _runtime.ThrowOnError(status, "sdxCreateContext");
            return new SdxContext(_runtime, ctxHandle);
        }
        finally
        {
            foreach (var ptr in encodedNames)
            {
                Marshal.FreeHGlobal(ptr);
            }

            if (namesBuffer != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(namesBuffer);
            }
        }
    }

    private static IntPtr AllocUtf8CString(string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value + '\0');
        var ptr = Marshal.AllocHGlobal(bytes.Length);
        Marshal.Copy(bytes, 0, ptr, bytes.Length);
        return ptr;
    }

    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            NativeMethods.sdxUnloadModel(_handle);
            _handle = IntPtr.Zero;
        }
    }
}

public sealed class SdxContext : IDisposable
{
    private readonly SdxRuntime _runtime;
    private IntPtr _handle;

    internal SdxContext(SdxRuntime runtime, IntPtr handle)
    {
        _runtime = runtime;
        _handle = handle;
    }

    public void Run(SdxTensorView[] inputs, SdxTensorView[] outputs, SdxRunOptions? options = null)
    {
        if (_handle == IntPtr.Zero)
        {
            throw new ObjectDisposedException(nameof(SdxContext));
        }

        var opts = options ?? SdxRunOptions.Default();
        var status = NativeMethods.sdxRun(_handle, inputs, inputs.Length, outputs, outputs.Length, ref opts);
        _runtime.ThrowOnError(status, "sdxRun");
    }

    public SdxExecutionReport ExecutionReport()
    {
        var report = SdxExecutionReport.Default();
        var status = NativeMethods.sdxGetExecutionReport(_handle, ref report);
        _runtime.ThrowOnError(status, "sdxGetExecutionReport");
        return report;
    }

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
