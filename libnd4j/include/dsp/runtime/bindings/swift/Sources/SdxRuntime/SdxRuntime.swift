import Foundation

public let SDX_STATUS_OK: Int32 = 0

public let SDX_BACKEND_AUTO: Int32 = 0
public let SDX_BACKEND_SLOT_BY_SLOT: Int32 = 1
public let SDX_BACKEND_CUDA_GRAPHS: Int32 = 2
public let SDX_BACKEND_NVRTC: Int32 = 3
public let SDX_BACKEND_PTX: Int32 = 4
public let SDX_BACKEND_TRITON: Int32 = 5
public let SDX_BACKEND_MLX: Int32 = 6
public let SDX_BACKEND_ARM_HYBRID: Int32 = 7
public let SDX_BACKEND_NNAPI: Int32 = 8

public let SDX_DEVICE_HOST: Int32 = 0
public let SDX_DEVICE_CUDA: Int32 = 1
public let SDX_DEVICE_AMD: Int32 = 2

public let SDX_GPU_TARGET_AUTO: Int32 = 0
public let SDX_GPU_TARGET_CUDA: Int32 = 1
public let SDX_GPU_TARGET_AMD: Int32 = 2

public struct sdx_runtime_options_t {
    public var struct_size: UInt32

    public init() {
        self.struct_size = UInt32(MemoryLayout<Self>.size)
    }
}

public struct sdx_model_options_t {
    public var struct_size: UInt32
    public var backend: Int32
    public var strict_backend: Int32
    public var allow_runtime_jit: Int32
    public var gpu_target: Int32

    public init() {
        self.struct_size = UInt32(MemoryLayout<Self>.size)
        self.backend = SDX_BACKEND_AUTO
        self.strict_backend = 0
        self.allow_runtime_jit = 0
        self.gpu_target = SDX_GPU_TARGET_AUTO
    }
}

public struct sdx_run_options_t {
    public var struct_size: UInt32
    public var backend: Int32
    public var strict_signature: Int32
    public var gpu_target: Int32

    public init() {
        self.struct_size = UInt32(MemoryLayout<Self>.size)
        self.backend = SDX_BACKEND_AUTO
        self.strict_signature = 1
        self.gpu_target = SDX_GPU_TARGET_AUTO
    }
}

public struct sdx_tensor_view_t {
    public var data: UnsafeMutableRawPointer?
    public var shape: UnsafePointer<Int64>?
    public var rank: Int32
    public var dtype: Int32
    public var bytes: Int
    public var device_type: Int32
    public var device_id: Int32

    public init() {
        self.data = nil
        self.shape = nil
        self.rank = 0
        self.dtype = 0
        self.bytes = 0
        self.device_type = SDX_DEVICE_HOST
        self.device_id = -1
    }
}

public struct sdx_execution_report_t {
    public var struct_size: UInt32
    public var requested_backend: Int32
    public var applied_backend: Int32
    public var status_code: Int32
    public var used_fallback: Int32
    public var execution_time_ns: UInt64
    public var requested_gpu_target: Int32
    public var applied_gpu_target: Int32

    public init() {
        self.struct_size = UInt32(MemoryLayout<Self>.size)
        self.requested_backend = 0
        self.applied_backend = 0
        self.status_code = 0
        self.used_fallback = 0
        self.execution_time_ns = 0
        self.requested_gpu_target = 0
        self.applied_gpu_target = 0
    }
}

@_silgen_name("sdxGetRuntimeAbiVersion")
private func sdxGetRuntimeAbiVersion() -> Int32

@_silgen_name("sdxCreateRuntime")
private func sdxCreateRuntime(_ options: UnsafePointer<sdx_runtime_options_t>?,
                              _ outRuntime: UnsafeMutablePointer<OpaquePointer?>?) -> Int32

@_silgen_name("sdxDestroyRuntime")
private func sdxDestroyRuntime(_ runtime: OpaquePointer?)

@_silgen_name("sdxLoadBundle")
private func sdxLoadBundle(_ runtime: OpaquePointer?,
                           _ bundlePath: UnsafePointer<CChar>?,
                           _ options: UnsafePointer<sdx_model_options_t>?,
                           _ outModel: UnsafeMutablePointer<OpaquePointer?>?) -> Int32

@_silgen_name("sdxUnloadModel")
private func sdxUnloadModel(_ model: OpaquePointer?)

@_silgen_name("sdxCreateContext")
private func sdxCreateContext(_ model: OpaquePointer?,
                              _ requestedOutputNames: UnsafeMutablePointer<UnsafePointer<CChar>?>?,
                              _ numRequestedOutputs: Int32,
                              _ outContext: UnsafeMutablePointer<OpaquePointer?>?) -> Int32

@_silgen_name("sdxDestroyContext")
private func sdxDestroyContext(_ context: OpaquePointer?)

@_silgen_name("sdxRun")
private func sdxRun(_ context: OpaquePointer?,
                    _ inputs: UnsafePointer<sdx_tensor_view_t>?,
                    _ numInputs: Int32,
                    _ outputs: UnsafePointer<sdx_tensor_view_t>?,
                    _ numOutputs: Int32,
                    _ options: UnsafePointer<sdx_run_options_t>?) -> Int32

@_silgen_name("sdxGetLastError")
private func sdxGetLastError(_ runtime: OpaquePointer?) -> UnsafePointer<CChar>?

@_silgen_name("sdxGetExecutionReport")
private func sdxGetExecutionReport(_ context: OpaquePointer?,
                                   _ outReport: UnsafeMutablePointer<sdx_execution_report_t>?) -> Int32

public enum SdxError: Error {
    case nativeStatus(code: Int32, message: String)
}

public final class SdxTensorViewLease {
    public private(set) var view: sdx_tensor_view_t
    private let shapeBuffer: UnsafeMutableBufferPointer<Int64>

    public init(data: UnsafeMutableRawPointer?, shape: [Int64], dtype: Int32, bytes: Int,
                deviceType: Int32 = SDX_DEVICE_HOST, deviceId: Int32 = -1) {
        self.shapeBuffer = UnsafeMutableBufferPointer<Int64>.allocate(capacity: shape.count)
        _ = self.shapeBuffer.initialize(from: shape)

        var v = sdx_tensor_view_t()
        v.data = data
        v.shape = UnsafePointer(self.shapeBuffer.baseAddress)
        v.rank = Int32(shape.count)
        v.dtype = dtype
        v.bytes = bytes
        v.device_type = deviceType
        v.device_id = deviceId
        self.view = v
    }

    deinit {
        shapeBuffer.deinitialize()
        shapeBuffer.deallocate()
    }
}

public final class SdxRuntime {
    private var handle: OpaquePointer?

    public init() throws {
        var runtime: OpaquePointer?
        var options = sdx_runtime_options_t()
        let status = withUnsafePointer(to: &options) { optPtr in
            sdxCreateRuntime(optPtr, &runtime)
        }

        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: "sdxCreateRuntime failed")
        }

        self.handle = runtime
    }

    deinit {
        close()
    }

    public func abiVersion() -> Int32 {
        sdxGetRuntimeAbiVersion()
    }

    public func loadModel(path: String, options: sdx_model_options_t? = nil) throws -> SdxModel {
        var model: OpaquePointer?

        let status = path.withCString { cPath in
            if var mutableOptions = options {
                return withUnsafePointer(to: &mutableOptions) { optPtr in
                    sdxLoadBundle(handle, cPath, optPtr, &model)
                }
            }
            return sdxLoadBundle(handle, cPath, nil, &model)
        }

        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: lastError())
        }

        return SdxModel(runtime: self, handle: model)
    }

    public func lastError() -> String {
        guard let ptr = sdxGetLastError(handle) else {
            return ""
        }
        return String(cString: ptr)
    }

    public func close() {
        if let h = handle {
            sdxDestroyRuntime(h)
            handle = nil
        }
    }

    fileprivate var rawHandle: OpaquePointer? {
        handle
    }
}

public final class SdxModel {
    private unowned let runtime: SdxRuntime
    private var handle: OpaquePointer?

    fileprivate init(runtime: SdxRuntime, handle: OpaquePointer?) {
        self.runtime = runtime
        self.handle = handle
    }

    deinit {
        close()
    }

    public func createContext(requestedOutputs: [String] = []) throws -> SdxContext {
        var context: OpaquePointer?
        var cStrings: [UnsafePointer<CChar>?] = []
        var buffers: [UnsafeMutablePointer<CChar>] = []

        for output in requestedOutputs {
            let dup = strdup(output)
            buffers.append(dup!)
            cStrings.append(UnsafePointer(dup))
        }

        defer {
            for buffer in buffers {
                free(buffer)
            }
        }

        let status = cStrings.withUnsafeMutableBufferPointer { ptr in
            if requestedOutputs.isEmpty {
                return sdxCreateContext(handle, nil, 0, &context)
            }
            return sdxCreateContext(handle, ptr.baseAddress, Int32(requestedOutputs.count), &context)
        }

        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: runtime.lastError())
        }

        return SdxContext(runtime: runtime, handle: context)
    }

    public func close() {
        if let h = handle {
            sdxUnloadModel(h)
            handle = nil
        }
    }
}

public final class SdxContext {
    private unowned let runtime: SdxRuntime
    private var handle: OpaquePointer?

    fileprivate init(runtime: SdxRuntime, handle: OpaquePointer?) {
        self.runtime = runtime
        self.handle = handle
    }

    deinit {
        close()
    }

    public func run(inputs: [sdx_tensor_view_t], outputs: [sdx_tensor_view_t], options: sdx_run_options_t? = nil) throws {
        let status = inputs.withUnsafeBufferPointer { inBuf in
            outputs.withUnsafeBufferPointer { outBuf in
                if var mutableOptions = options {
                    return withUnsafePointer(to: &mutableOptions) { optPtr in
                        sdxRun(
                            handle,
                            inBuf.baseAddress,
                            Int32(inputs.count),
                            outBuf.baseAddress,
                            Int32(outputs.count),
                            optPtr
                        )
                    }
                }

                return sdxRun(
                    handle,
                    inBuf.baseAddress,
                    Int32(inputs.count),
                    outBuf.baseAddress,
                    Int32(outputs.count),
                    nil
                )
            }
        }

        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: runtime.lastError())
        }
    }

    public func executionReport() throws -> sdx_execution_report_t {
        var report = sdx_execution_report_t()
        let status = sdxGetExecutionReport(handle, &report)
        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: runtime.lastError())
        }
        return report
    }

    public func close() {
        if let h = handle {
            sdxDestroyContext(h)
            handle = nil
        }
    }
}
