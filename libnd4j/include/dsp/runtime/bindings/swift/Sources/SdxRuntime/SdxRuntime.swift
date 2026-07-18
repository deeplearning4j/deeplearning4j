// ******************************************************************************
//
// This program and the accompanying materials are made available under the
// terms of the Apache License, Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

import Foundation
import CSdxRuntime

// ── Typed enums ───────────────────────────────────────────────────────────────

/// Execution backend selected at model load or at run time.
///
/// - `auto`:         Let the runtime choose the fastest available backend.
/// - `slotBySlot`:   Eager op-by-op execution (always correct; used for warmup).
/// - `cudaGraphs`:   NVIDIA CUDA Graph capture and replay.
/// - `nvrtc`:        JIT-compiled kernels via NVRTC.
/// - `ptx`:          Pre-compiled PTX kernels.
/// - `triton`:       Triton-compiled fused kernels.
/// - `mlx`:          Apple MLX (Apple Silicon).
/// - `armHybrid`:    ARM CPU/GPU hybrid path.
/// - `nnapi`:        Android NNAPI.
/// - `hipGraphs`:    AMD ROCm HIP graph capture and replay.
/// - `levelZero`:    Intel Level Zero mutable command-list replay.
/// - `vulkan`:       Vulkan compute command-buffer replay.
/// - `metal`:        Apple Metal indirect command-buffer replay.
/// - `tpu`:          TPU HLO compilation and PJRT execution.
/// - `hexagon`:      Qualcomm Hexagon NPU execution.
public enum SdxBackend: Int32, CustomStringConvertible {
    case auto       = 0
    case slotBySlot = 1
    case cudaGraphs = 2
    case nvrtc      = 3
    case ptx        = 4
    case triton     = 5
    case mlx        = 6
    case armHybrid  = 7
    case nnapi      = 8
    case hipGraphs  = 9
    case levelZero  = 10
    case vulkan     = 11
    case metal      = 12
    case tpu        = 13
    case hexagon    = 14

    public var description: String {
        switch self {
        case .auto:        return "AUTO"
        case .slotBySlot:  return "SLOT_BY_SLOT"
        case .cudaGraphs:  return "CUDA_GRAPHS"
        case .nvrtc:       return "NVRTC"
        case .ptx:         return "PTX"
        case .triton:      return "TRITON"
        case .mlx:         return "MLX"
        case .armHybrid:   return "ARM_HYBRID"
        case .nnapi:       return "NNAPI"
        case .hipGraphs:   return "HIP_GRAPHS"
        case .levelZero:   return "LEVEL_ZERO"
        case .vulkan:      return "VULKAN"
        case .metal:       return "METAL"
        case .tpu:         return "TPU"
        case .hexagon:     return "HEXAGON"
        }
    }
}

/// Memory device on which a tensor resides.
public enum SdxDevice: Int32 {
    case host = 0
    case cuda = 1
    case amd  = 2
}

/// GPU compute target.
public enum SdxGpuTarget: Int32 {
    case auto   = 0
    case cuda   = 1
    case amd    = 2
    case vulkan = 3
    case metal  = 4
}

/// DSP execution plan lifecycle phase.
///
/// - `slotBySlot`:    Warmup / eager execution (shapes not yet stable).
/// - `shapesFrozen`:  Shapes frozen; CUDA graph capture may proceed.
/// - `replaying`:     CUDA graph replay active (fast path).
/// - `replayBlocked`: A segment failed capture; fell back to slot-by-slot.
public enum SdxPlanPhase: Int32, CustomStringConvertible {
    case slotBySlot    = 0
    case shapesFrozen  = 1
    case replaying     = 2
    case replayBlocked = 3

    public var description: String {
        switch self {
        case .slotBySlot:    return "SLOT_BY_SLOT (warmup)"
        case .shapesFrozen:  return "SHAPES_FROZEN"
        case .replaying:     return "REPLAYING"
        case .replayBlocked: return "REPLAY_BLOCKED"
        }
    }
}

// Backward-compatible Int32 constants (deprecated; prefer the typed enums above).
public let SDX_STATUS_OK: Int32 = 0

@available(*, deprecated, renamed: "SdxBackend.auto.rawValue")
public let SDX_BACKEND_AUTO:        Int32 = 0
@available(*, deprecated, renamed: "SdxBackend.slotBySlot.rawValue")
public let SDX_BACKEND_SLOT_BY_SLOT: Int32 = 1
@available(*, deprecated, renamed: "SdxBackend.cudaGraphs.rawValue")
public let SDX_BACKEND_CUDA_GRAPHS: Int32 = 2
@available(*, deprecated, renamed: "SdxBackend.nvrtc.rawValue")
public let SDX_BACKEND_NVRTC:       Int32 = 3
@available(*, deprecated, renamed: "SdxBackend.ptx.rawValue")
public let SDX_BACKEND_PTX:         Int32 = 4
@available(*, deprecated, renamed: "SdxBackend.triton.rawValue")
public let SDX_BACKEND_TRITON:      Int32 = 5
@available(*, deprecated, renamed: "SdxBackend.mlx.rawValue")
public let SDX_BACKEND_MLX:         Int32 = 6
@available(*, deprecated, renamed: "SdxBackend.armHybrid.rawValue")
public let SDX_BACKEND_ARM_HYBRID:  Int32 = 7
@available(*, deprecated, renamed: "SdxBackend.nnapi.rawValue")
public let SDX_BACKEND_NNAPI:       Int32 = 8
@available(*, deprecated, renamed: "SdxBackend.hipGraphs.rawValue")
public let SDX_BACKEND_HIP_GRAPHS:  Int32 = 9
@available(*, deprecated, renamed: "SdxBackend.levelZero.rawValue")
public let SDX_BACKEND_LEVEL_ZERO:  Int32 = 10
@available(*, deprecated, renamed: "SdxBackend.vulkan.rawValue")
public let SDX_BACKEND_VULKAN:      Int32 = 11
@available(*, deprecated, renamed: "SdxBackend.metal.rawValue")
public let SDX_BACKEND_METAL:       Int32 = 12
@available(*, deprecated, renamed: "SdxBackend.tpu.rawValue")
public let SDX_BACKEND_TPU:         Int32 = 13
@available(*, deprecated, renamed: "SdxBackend.hexagon.rawValue")
public let SDX_BACKEND_HEXAGON:     Int32 = 14

@available(*, deprecated, renamed: "SdxDevice.host.rawValue")
public let SDX_DEVICE_HOST: Int32 = 0
@available(*, deprecated, renamed: "SdxDevice.cuda.rawValue")
public let SDX_DEVICE_CUDA: Int32 = 1
@available(*, deprecated, renamed: "SdxDevice.amd.rawValue")
public let SDX_DEVICE_AMD:  Int32 = 2

@available(*, deprecated, renamed: "SdxGpuTarget.auto.rawValue")
public let SDX_GPU_TARGET_AUTO: Int32 = 0
@available(*, deprecated, renamed: "SdxGpuTarget.cuda.rawValue")
public let SDX_GPU_TARGET_CUDA: Int32 = 1
@available(*, deprecated, renamed: "SdxGpuTarget.amd.rawValue")
public let SDX_GPU_TARGET_AMD: Int32 = 2
@available(*, deprecated, renamed: "SdxGpuTarget.vulkan.rawValue")
public let SDX_GPU_TARGET_VULKAN: Int32 = 3
@available(*, deprecated, renamed: "SdxGpuTarget.metal.rawValue")
public let SDX_GPU_TARGET_METAL: Int32 = 4

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors thrown by the SDX Swift wrapper.
public enum SdxError: Error, CustomStringConvertible {
    /// The native layer returned a non-zero status code.
    case nativeStatus(code: Int32, message: String)

    public var description: String {
        switch self {
        case let .nativeStatus(code, message):
            return "SdxError(code=\(code)): \(message)"
        }
    }
}

// ── SdxTensor — idiomatic value type ─────────────────────────────────────────

/// An f32 tensor with copy-value semantics.
///
/// `SdxTensor` owns its element data as a plain `[Float]` array, making it
/// safe to pass by value, return from functions, and store in collections
/// without any lifetime management.
///
/// ```swift
/// let x = SdxTensor(shape: [2, 4], scalars: [0.1, 0.2, 0.3, 0.4,
///                                             0.5, 0.6, 0.7, 0.8])
/// let outputs = try context.run(inputs: ["x": x],
///                               outputShapes: ["probs": [2, 3]])
/// print(outputs["probs"]!.scalars)
/// ```
///
/// Use ``SdxTensorViewLease`` when you need zero-copy access to a
/// pre-existing buffer (e.g. a Metal texture or a Core ML feature value).
public struct SdxTensor {
    /// `sd::DataType::FLOAT32` — the only dtype currently supported by the C ABI.
    public static let dtypeFloat32: Int32 = 5

    /// Logical shape, e.g. `[2, 4]` for a batch-2, column-4 matrix.
    public let shape: [Int]

    /// Element data in C-contiguous (row-major) order.
    public let scalars: [Float]

    /// Element data type identifier; currently always ``dtypeFloat32``.
    public let dtype: Int32

    /// Total number of elements (product of `shape`).
    public var count: Int { scalars.count }

    /// `true` when the tensor has no elements.
    public var isEmpty: Bool { scalars.isEmpty }

    /// Create a tensor with the given shape and scalar data.
    ///
    /// - Parameters:
    ///   - shape:   Logical dimensions (e.g. `[2, 4]`).
    ///   - scalars: Flat f32 data in C-contiguous order.
    ///              Must satisfy `scalars.count == shape.reduce(1, *)`.
    ///   - dtype:   Data type identifier; defaults to ``dtypeFloat32``.
    public init(shape: [Int], scalars: [Float], dtype: Int32 = SdxTensor.dtypeFloat32) {
        let expected = shape.isEmpty ? 0 : shape.reduce(1, *)
        precondition(scalars.count == expected,
            "SdxTensor: scalars.count \(scalars.count) ≠ shape product \(expected) for shape \(shape)")
        self.shape   = shape
        self.scalars = scalars
        self.dtype   = dtype
    }

    /// Create an all-zero output placeholder with the given shape.
    public static func zeros(shape: [Int]) -> SdxTensor {
        SdxTensor(shape: shape,
                  scalars: [Float](repeating: 0, count: shape.reduce(1, *)))
    }

    /// Reshape to a new shape (element count must be unchanged).
    public func reshaped(to newShape: [Int]) -> SdxTensor {
        SdxTensor(shape: newShape, scalars: scalars, dtype: dtype)
    }

    // MARK: - Internal

    /// Execute `body` with a stable `sdx_tensor_view_t` pointer for this tensor.
    ///
    /// The pointer (and all buffers it references) is guaranteed to be valid
    /// only for the duration of `body`.  Do not escape the pointer.
    internal func withUnsafeViewPointer<R>(
        _ body: (UnsafePointer<sdx_tensor_view_t>) throws -> R
    ) rethrows -> R {
        try scalars.withUnsafeBytes { rawBytes in
            // shape must be Int64 for the C ABI
            let shape64 = shape.map { Int64($0) }
            return try shape64.withUnsafeBufferPointer { shapeBuf in
                var view = sdx_tensor_view_t()
                view.data       = UnsafeMutableRawPointer(mutating: rawBytes.baseAddress)
                view.shape      = shapeBuf.baseAddress
                view.rank       = Int32(shape.count)
                view.dtype      = dtype
                view.bytes      = scalars.count * MemoryLayout<Float>.size
                view.device_type = SdxDevice.host.rawValue
                view.device_id  = -1
                return try withUnsafePointer(to: view, body)
            }
        }
    }
}

/// An owning copy of a dynamically allocated SDX output. This is the safe
/// mobile-facing result type for logits whose shape is not known before a run.
public struct SdxRawTensor {
    public let shape: [Int64]
    public let dtype: Int32
    public let data: Data

    internal init(view: sdx_tensor_view_t) {
        let rank = max(0, Int(view.rank))
        if rank == 0 || view.shape == nil {
            shape = []
        } else {
            shape = Array(
                UnsafeBufferPointer(start: view.shape, count: rank)
            )
        }
        dtype = view.dtype
        if view.bytes == 0 {
            data = Data()
        } else {
            precondition(view.data != nil,
                         "SDX returned non-empty output with a null data pointer")
            data = Data(bytes: view.data!, count: view.bytes)
        }
    }

    /// Decode FLOAT32 output storage, or return nil for another dtype.
    public var float32Scalars: [Float]? {
        guard dtype == SdxTensor.dtypeFloat32,
              data.count % MemoryLayout<Float>.size == 0 else {
            return nil
        }
        return data.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
    }
}

// ── OutputBuffer — mutable heap storage for a single output tensor ────────────

/// Heap-allocated mutable buffer for a single plan output.
///
/// `SdxContext.run(inputs:outputShapes:)` allocates one `OutputBuffer` per
/// requested output, passes its raw pointer to `sdxRun`, then reads back the
/// results into a value-type ``SdxTensor``.
private final class OutputBuffer {
    let shape: [Int]
    let dtype: Int32
    let buf: UnsafeMutableBufferPointer<Float>

    init(shape: [Int], dtype: Int32 = SdxTensor.dtypeFloat32) {
        self.shape = shape
        self.dtype = dtype
        let n = shape.isEmpty ? 0 : shape.reduce(1, *)
        self.buf = UnsafeMutableBufferPointer<Float>.allocate(capacity: max(n, 1))
        self.buf.initialize(repeating: 0)
    }

    deinit {
        buf.deinitialize()
        buf.deallocate()
    }

    /// Build the C view that points into this buffer.  Caller must hold `self` alive.
    func makeLease() -> SdxTensorViewLease {
        SdxTensorViewLease(
            data:  UnsafeMutableRawPointer(buf.baseAddress),
            shape: shape.map { Int64($0) },
            dtype: dtype,
            bytes: buf.count * MemoryLayout<Float>.size
        )
    }

    /// Read the contents back as a value-type ``SdxTensor``.
    func readBack() -> SdxTensor {
        SdxTensor(shape: shape, scalars: Array(buf), dtype: dtype)
    }
}

// ── SdxTensorViewLease — zero-copy lease for power users ─────────────────────

/// A managed wrapper around `sdx_tensor_view_t` that borrows a caller-owned buffer.
///
/// The lease owns the `Int64` shape array allocated in C memory, but does **not**
/// own the data pointer.  Callers must ensure the data buffer outlives the lease.
///
/// Prefer ``SdxTensor`` for everyday use.  Use `SdxTensorViewLease` only when
/// you need zero-copy access to an existing allocation — for example, a Metal
/// buffer or a Core ML `MLMultiArray`.
public final class SdxTensorViewLease {
    public private(set) var view: sdx_tensor_view_t
    private let shapeBuffer: UnsafeMutableBufferPointer<Int64>

    /// Create a view that borrows `data`.
    ///
    /// - Parameters:
    ///   - data:       Raw pointer to element data; must stay valid for the lease lifetime.
    ///   - shape:      Logical dimensions as `[Int64]`.
    ///   - dtype:      Data type (e.g. `SdxTensor.dtypeFloat32`).
    ///   - bytes:      Total byte count of the data buffer.
    ///   - deviceType: Device the data lives on; defaults to ``SdxDevice/host``.
    ///   - deviceId:   Device ordinal for CUDA/AMD tensors; -1 for host.
    public init(data: UnsafeMutableRawPointer?, shape: [Int64], dtype: Int32, bytes: Int,
                deviceType: Int32 = SdxDevice.host.rawValue, deviceId: Int32 = -1) {
        shapeBuffer = UnsafeMutableBufferPointer<Int64>.allocate(capacity: shape.count)
        _ = shapeBuffer.initialize(from: shape)

        var v = sdx_tensor_view_t()
        v.data        = data
        v.shape       = UnsafePointer(shapeBuffer.baseAddress)
        v.rank        = Int32(shape.count)
        v.dtype       = dtype
        v.bytes       = bytes
        v.device_type = deviceType
        v.device_id   = deviceId
        self.view = v
    }

    deinit {
        shapeBuffer.deinitialize()
        shapeBuffer.deallocate()
    }
}

// ── SdxExecutionReport — Swifty telemetry snapshot ───────────────────────────

/// Telemetry snapshot returned by ``SdxContext/executionReport()``.
public struct SdxExecutionReport: CustomStringConvertible {
    /// Backend requested when the context was created.
    public let requestedBackend: SdxBackend?
    /// Backend actually used after clamping and plan-side mode changes.
    public let appliedBackend: SdxBackend?
    /// Raw native status code from the last execution (0 = OK).
    public let statusCode: Int32
    /// Whether the runtime fell back to a slower execution path.
    ///
    /// `true`  — a segment failed graph capture, or plan is ``SdxPlanPhase/replayBlocked``.
    /// `false` — requested / optimal path is in force.
    /// `nil`   — no execution yet.
    public let usedFallback: Bool?
    /// Wall-clock duration of the last `run()` call.
    public let executionTimeNanoseconds: UInt64
    /// GPU target originally requested.
    public let requestedGpuTarget: SdxGpuTarget?
    /// GPU target actually applied.
    public let appliedGpuTarget: SdxGpuTarget?
    /// Plan lifecycle phase after the last execution.
    public let planPhase: SdxPlanPhase?
    /// Number of `run()` calls completed on this context.
    public let executionCount: Int

    internal init(raw: sdx_execution_report_t) {
        requestedBackend   = SdxBackend(rawValue: raw.requested_backend)
        appliedBackend     = SdxBackend(rawValue: raw.applied_backend)
        statusCode         = raw.status_code
        usedFallback       = raw.used_fallback < 0 ? nil : (raw.used_fallback == 1)
        executionTimeNanoseconds = raw.execution_time_ns
        requestedGpuTarget = SdxGpuTarget(rawValue: raw.requested_gpu_target)
        appliedGpuTarget   = SdxGpuTarget(rawValue: raw.applied_gpu_target)
        planPhase          = SdxPlanPhase(rawValue: raw.plan_phase)
        executionCount     = Int(raw.execution_count)
    }

    /// Execution time in milliseconds.
    public var executionTimeMilliseconds: Double {
        Double(executionTimeNanoseconds) / 1_000_000.0
    }

    public var description: String {
        let fb: String
        switch usedFallback {
        case .none:         fb = "unknown"
        case .some(true):   fb = "yes"
        case .some(false):  fb = "no"
        }
        return """
        SdxExecutionReport {
          statusCode       = \(statusCode)
          requestedBackend = \(requestedBackend?.description ?? "?")
          appliedBackend   = \(appliedBackend?.description ?? "?")
          usedFallback     = \(fb)
          planPhase        = \(planPhase?.description ?? "?")
          executionCount   = \(executionCount)
          executionTime    = \(String(format: "%.3f", executionTimeMilliseconds)) ms
        }
        """
    }
}

// ── SdxRuntime ────────────────────────────────────────────────────────────────

/// Root object of the SDX inference runtime.
///
/// Create one runtime per process.  Resources are released automatically on
/// `deinit`, or explicitly via ``close()``.
///
/// ```swift
/// let runtime = try SdxRuntime()
/// let model   = try runtime.loadModel(path: "model.sdz")
/// ```
public final class SdxRuntime {
    private var handle: OpaquePointer?

    /// Create a runtime with default options.
    public init() throws {
        var runtime: OpaquePointer?
        var options = sdx_runtime_options_t()
        let status = withUnsafePointer(to: &options) { sdxCreateRuntime($0, &runtime) }

        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: "sdxCreateRuntime failed")
        }
        self.handle = runtime
    }

    deinit { close() }

    /// ABI version of the linked SDX runtime library.
    public func abiVersion() -> Int32 { sdxGetRuntimeAbiVersion() }

    /// Load a compiled model bundle from `path`.
    ///
    /// - Parameters:
    ///   - path:    File-system path to the `.sdz` bundle.
    ///   - options: Optional low-level model options (backend, JIT policy, etc.).
    /// - Returns: A loaded ``SdxModel`` ready for context creation.
    /// - Throws:  ``SdxError/nativeStatus(code:message:)`` on failure.
    public func loadModel(path: String, options: sdx_model_options_t? = nil) throws -> SdxModel {
        var model: OpaquePointer?
        let status = path.withCString { cPath in
            if var opts = options {
                return withUnsafePointer(to: &opts) { sdxLoadBundle(handle, cPath, $0, &model) }
            }
            return sdxLoadBundle(handle, cPath, nil, &model)
        }
        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: lastError())
        }
        return SdxModel(runtime: self, handle: model)
    }

    /// The last error message from the native runtime, or an empty string.
    public func lastError() -> String {
        guard let ptr = sdxGetLastError(handle) else { return "" }
        return String(cString: ptr)
    }

    /// Release the native runtime handle.  Called automatically on `deinit`.
    public func close() {
        guard let h = handle else { return }
        sdxDestroyRuntime(h)
        handle = nil
    }

    fileprivate var rawHandle: OpaquePointer? { handle }
}

// ── SdxModel ──────────────────────────────────────────────────────────────────

/// A loaded SDX model bundle.
///
/// A model is a compiled DSP plan container.  Create one or more ``SdxContext``
/// objects from a model to run inference.  Resources are released on `deinit`
/// or via ``close()``.
public final class SdxModel {
    private let runtime: SdxRuntime
    private var handle: OpaquePointer?

    fileprivate init(runtime: SdxRuntime, handle: OpaquePointer?) {
        self.runtime = runtime
        self.handle  = handle
    }

    deinit { close() }

    /// Resolved offline tokenizer path declared in the loaded bundle.
    public var tokenizerPath: String? {
        guard let ptr = sdxGetTokenizerPath(handle) else { return nil }
        return String(cString: ptr)
    }

    /// Resolved text-generation IO/KV/sampling metadata path.
    public var textGenerationConfigPath: String? {
        guard let ptr = sdxGetTextGenerationConfigPath(handle) else {
            return nil
        }
        return String(cString: ptr)
    }

    /// Create an inference context for this model.
    ///
    /// A context holds the plan's per-session mutable state: phase counter,
    /// frozen shape snapshot, CUDA graph handles, staging buffers.  **Not
    /// thread-safe** — create one context per concurrent inference stream.
    ///
    /// - Parameter requestedOutputs: Names of outputs to materialise on every
    ///   `run()` call.  Pass `[]` (the default) to request all outputs.
    /// - Returns: A ready ``SdxContext``.
    /// - Throws:  ``SdxError/nativeStatus(code:message:)`` on failure.
    public func createContext(
        requestedOutputs: [String] = [],
        bindModelParameters: Bool = false
    ) throws -> SdxContext {
        var context: OpaquePointer?
        var buffers: [UnsafeMutablePointer<CChar>] = []
        var cStrings: [UnsafePointer<CChar>?] = requestedOutputs.map { name in
            let dup = strdup(name)!
            buffers.append(dup)
            return UnsafePointer(dup)
        }
        defer { buffers.forEach { free($0) } }

        var options = sdx_context_options_t()
        options.struct_size = UInt32(MemoryLayout<sdx_context_options_t>.size)
        options.bind_model_parameters = bindModelParameters ? 1 : 0
        let status = cStrings.withUnsafeMutableBufferPointer { ptr in
            let names = requestedOutputs.isEmpty ? nil : ptr.baseAddress
            if bindModelParameters {
                return sdxCreateContextWithOptions(
                    handle, names, Int32(requestedOutputs.count), &options, &context)
            }
            return sdxCreateContext(
                handle, names, Int32(requestedOutputs.count), &context)
        }
        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: runtime.lastError())
        }
        return SdxContext(runtime: runtime, model: self, handle: context)
    }

    /// Create the recommended mobile/offline context. Bundle-owned weights are
    /// retained by this model and omitted from the public input list.
    public func createInferenceContext(
        requestedOutputs: [String] = []
    ) throws -> SdxContext {
        try createContext(
            requestedOutputs: requestedOutputs, bindModelParameters: true)
    }

    /// Release the native model handle.  Called automatically on `deinit`.
    public func close() {
        guard let h = handle else { return }
        sdxUnloadModel(h)
        handle = nil
    }
}

// ── SdxContext ────────────────────────────────────────────────────────────────

/// An inference context bound to a single ``SdxModel``.
///
/// A context owns the DSP plan's per-session state — warmup counter, frozen
/// shape snapshot, CUDA graph handles, staging buffers.  **Not thread-safe**;
/// create one context per concurrent inference stream.
///
/// ## Typical lifecycle
///
/// ```swift
/// let ctx = try model.createContext(requestedOutputs: ["probs"])
///
/// // 1. Mark the batch input as a placeholder (value changes each call).
/// let xIdx = ctx.inputNames().firstIndex(of: "x").map(Int32.init)!
/// try ctx.markInputPlaceholder(xIdx)
///
/// // 2. Warmup: a few runs to stabilise shapes.
/// for step in 1...3 {
///     let _ = try ctx.run(inputs: myWeightsAndInput,
///                         outputShapes: ["probs": [2, 3]])
/// }
///
/// // 3. Freeze shapes → enables CUDA graph capture and the stable fast path.
/// try ctx.freezeShapes()
///
/// // 4. Production loop.
/// let outputs = try ctx.run(inputs: myWeightsAndInput,
///                           outputShapes: ["probs": [2, 3]])
/// print(outputs["probs"]!.scalars)
/// ```
public final class SdxContext {
    private let runtime: SdxRuntime
    // Keeps bundle-owned arrays alive for parameter-bound contexts.
    private let model: SdxModel
    private var handle: OpaquePointer?

    fileprivate init(runtime: SdxRuntime, model: SdxModel, handle: OpaquePointer?) {
        self.runtime = runtime
        self.model = model
        self.handle  = handle
    }

    deinit { close() }

    // MARK: - High-level named-tensor API

    /// Run the plan with named input tensors; return named output tensors.
    ///
    /// Inputs are matched to the plan's positional binding order via
    /// ``inputNames()``.  Every name returned by `inputNames()` must appear in
    /// `inputs`; a `precondition` failure is raised for any missing name.
    ///
    /// For each key in `outputShapes`, a zero-initialised output buffer of that
    /// shape is allocated, passed to the native runtime, and read back into a
    /// fresh ``SdxTensor`` after the call.
    ///
    /// - Parameters:
    ///   - inputs:       Dictionary mapping input name → ``SdxTensor``.
    ///   - outputShapes: Dictionary mapping output name → shape for each output
    ///                   that should be collected.  Keys must match the names
    ///                   passed to ``SdxModel/createContext(requestedOutputs:)``.
    /// - Returns: Dictionary mapping output name → ``SdxTensor`` with results.
    /// - Throws:  ``SdxError/nativeStatus(code:message:)`` on execution failure.
    public func run(
        inputs: [String: SdxTensor],
        outputShapes: [String: [Int]]
    ) throws -> [String: SdxTensor] {
        let planInputNames = inputNames()

        // Validate: every plan input must be supplied.
        for name in planInputNames {
            precondition(
                inputs[name] != nil,
                "SdxContext.run: missing tensor for plan input '\(name)'. " +
                "Required: \(planInputNames)"
            )
        }

        // Ordered input tensors following plan binding order.
        let orderedInputs = planInputNames.map { inputs[$0]! }

        // Allocate mutable output buffers on the heap.
        let outputKeys    = Array(outputShapes.keys)
        let outputBuffers = outputKeys.map { key in OutputBuffer(shape: outputShapes[key]!) }
        let outputLeases  = outputBuffers.map { $0.makeLease() }
        var outputViews   = outputLeases.map { $0.view }

        // Pin every input scalar array in memory simultaneously via a recursive
        // scope chain, then fire sdxRun with all pointers stable.
        var inputViews = [sdx_tensor_view_t](repeating: sdx_tensor_view_t(), count: orderedInputs.count)

        func pinInputs(_ idx: Int, then body: () throws -> Void) rethrows {
            guard idx < orderedInputs.count else {
                try body()
                return
            }
            try orderedInputs[idx].withUnsafeViewPointer { ptr in
                inputViews[idx] = ptr.pointee
                try pinInputs(idx + 1, then: body)
            }
        }

        try pinInputs(0) {
            try self.run(inputs: inputViews, outputs: outputViews)
        }

        // Suppress "unused" warnings; leases and buffers must survive through run().
        _ = outputLeases
        _ = outputViews

        // Read results back into value-type tensors.
        return Dictionary(uniqueKeysWithValues: zip(outputKeys, outputBuffers.map { $0.readBack() }))
    }

    // MARK: - Low-level lease API (zero-copy, power users)

    /// Run the plan with pre-built `sdx_tensor_view_t` arrays.
    ///
    /// Inputs and outputs are bound positionally in plan order.  All views (and
    /// the buffers they reference) must remain valid for the duration of this call.
    ///
    /// Prefer ``run(inputs:outputShapes:)`` for everyday use.
    ///
    /// - Parameters:
    ///   - inputs:  Views for the plan's external inputs in plan binding order.
    ///   - outputs: Views for the plan's output buffers (written in place).
    ///   - options: Optional low-level run options.
    /// - Throws: ``SdxError/nativeStatus(code:message:)`` on failure.
    public func run(inputs: [sdx_tensor_view_t], outputs: [sdx_tensor_view_t],
                    options: sdx_run_options_t? = nil) throws {
        let status = inputs.withUnsafeBufferPointer { inBuf in
            outputs.withUnsafeBufferPointer { outBuf in
                if var opts = options {
                    return withUnsafePointer(to: &opts) {
                        sdxRun(handle,
                               inBuf.baseAddress, Int32(inputs.count),
                               outBuf.baseAddress, Int32(outputs.count), $0)
                    }
                }
                return sdxRun(handle,
                              inBuf.baseAddress, Int32(inputs.count),
                              outBuf.baseAddress, Int32(outputs.count), nil)
            }
        }
        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: runtime.lastError())
        }
    }

    /// Run without preallocating output buffers and copy the dynamic results
    /// into Swift-owned storage. This is the preferred logits path on iOS.
    public func runAllocating(
        inputs: [sdx_tensor_view_t],
        options: sdx_run_options_t? = nil
    ) throws -> [SdxRawTensor] {
        let status = inputs.withUnsafeBufferPointer { inBuf in
            if var opts = options {
                return withUnsafePointer(to: &opts) {
                    sdxRunAllocating(
                        handle, inBuf.baseAddress, Int32(inputs.count), $0)
                }
            }
            return sdxRunAllocating(
                handle, inBuf.baseAddress, Int32(inputs.count), nil)
        }
        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(
                code: status, message: runtime.lastError())
        }

        let count = max(0, Int(numOutputs()))
        var results: [SdxRawTensor] = []
        results.reserveCapacity(count)
        for index in 0..<count {
            var view = sdx_tensor_view_t()
            let outputStatus =
                sdxGetOutputTensor(handle, Int32(index), &view)
            guard outputStatus == SDX_STATUS_OK else {
                throw SdxError.nativeStatus(
                    code: outputStatus, message: runtime.lastError())
            }
            results.append(SdxRawTensor(view: view))
        }
        return results
    }

    // MARK: - Telemetry

    /// Execution telemetry snapshot from the last `run()` call.
    ///
    /// - Returns: An ``SdxExecutionReport`` with Swifty names and typed enums.
    /// - Throws:  ``SdxError/nativeStatus(code:message:)`` on failure.
    public func executionReport() throws -> SdxExecutionReport {
        var raw = sdx_execution_report_t()
        let status = sdxGetExecutionReport(handle, &raw)
        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: runtime.lastError())
        }
        return SdxExecutionReport(raw: raw)
    }

    // MARK: - Input marking

    /// Mark an input as *variable* (value changes between runs; shape stays fixed).
    ///
    /// Enables D2D staging buffers for the input, reducing host round-trips.
    /// Call after context creation, before the first `run()`.
    ///
    /// - Parameter inputIndex: Zero-based plan input index (see ``inputNames()``).
    /// - Throws: ``SdxError/nativeStatus(code:message:)`` on failure.
    public func markInputVariable(_ inputIndex: Int32) throws {
        let status = sdxMarkInputVariable(handle, inputIndex)
        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: runtime.lastError())
        }
    }

    /// Mark an input as *placeholder* (both value and shape may change between runs).
    ///
    /// Placeholders receive a full host-to-device sync on every `run()` call.
    /// Use this for batch inputs and attention masks.
    ///
    /// - Parameter inputIndex: Zero-based plan input index (see ``inputNames()``).
    /// - Throws: ``SdxError/nativeStatus(code:message:)`` on failure.
    public func markInputPlaceholder(_ inputIndex: Int32) throws {
        let status = sdxMarkInputPlaceholder(handle, inputIndex)
        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: runtime.lastError())
        }
    }

    // MARK: - Plan lifecycle

    /// Freeze plan shapes, enabling CUDA graph capture and the argTableStable fast path.
    ///
    /// Call this after a warmup phase (typically 2–3 runs) once shapes have
    /// stabilised.  After freezing, the plan transitions from
    /// ``SdxPlanPhase/slotBySlot`` toward ``SdxPlanPhase/replaying``.
    ///
    /// - Throws: ``SdxError/nativeStatus(code:message:)`` on failure.
    public func freezeShapes() throws {
        let status = sdxFreezeShapes(handle)
        guard status == SDX_STATUS_OK else {
            throw SdxError.nativeStatus(code: status, message: runtime.lastError())
        }
    }

    /// Current plan phase, or `nil` when the native layer reports an error.
    public var planPhase: SdxPlanPhase? {
        SdxPlanPhase(rawValue: sdxGetPlanPhase(handle))
    }

    /// Number of `run()` calls completed on this context, or -1 on error.
    public func executionCount() -> Int32 { sdxGetExecutionCount(handle) }

    // MARK: - Input introspection

    /// Number of external inputs the plan expects per `run()` call, or -1 on error.
    ///
    /// External inputs cover the model's constants, variables, and placeholders,
    /// all bound positionally in plan order.  Use ``inputNames()`` to obtain
    /// the name for each slot.
    public func numInputs() -> Int32 { sdxGetNumInputs(handle) }

    /// Number of output tensors produced per `run()` call, or -1 on error.
    public func numOutputs() -> Int32 { sdxGetNumOutputs(handle) }

    /// Name of the external input at the given plan index, or `nil` on error.
    public func inputName(_ inputIndex: Int32) -> String? {
        guard let ptr = sdxGetInputName(handle, inputIndex) else { return nil }
        return String(cString: ptr)
    }

    /// All external input names in plan binding order.
    ///
    /// Discover what tensors a model requires before the first call:
    /// ```swift
    /// let names = ctx.inputNames()
    /// // e.g. ["w1", "b1", "w2", "b2", "x"]
    /// ```
    public func inputNames() -> [String] {
        let count = numInputs()
        guard count > 0 else { return [] }
        return (0..<count).map { inputName($0) ?? "" }
    }

    /// Explicit requested output name at the given index, if supplied.
    public func outputName(_ outputIndex: Int32) -> String? {
        guard let ptr = sdxGetOutputName(handle, outputIndex) else {
            return nil
        }
        return String(cString: ptr)
    }

    /// Explicit requested output names in plan order.
    public func outputNames() -> [String?] {
        let count = numOutputs()
        guard count > 0 else { return [] }
        return (0..<count).map { outputName($0) }
    }

    // MARK: - Resource management

    /// Release the native context handle.  Called automatically on `deinit`.
    public func close() {
        guard let h = handle else { return }
        sdxDestroyContext(h)
        handle = nil
    }
}
