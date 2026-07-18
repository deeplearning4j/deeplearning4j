// ******************************************************************************
//
// This program and the accompanying materials are made available under the
// terms of the Apache License, Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

/**
 * SdxLlm — idiomatic Swift wrapper over ``sdx_llm_c.h``.
 *
 * ## Overview
 *
 * ``SdxLlmRuntime`` is the entry point. Create one per OS thread (the
 * underlying GraalVM isolate has single-thread affinity — see `sdx_llm_c.h`).
 *
 * ```swift
 * // Set SDX_NATIVE_LIB_DIR before the first call when using the AOT SDK.
 * // SdxLlmRuntime() does this automatically from SDX_LLM_AOT_HOME if unset.
 * let rt = try SdxLlmRuntime()
 * print("ABI version:", rt.abiVersion())
 *
 * let model = try rt.loadModel(
 *     modelPath:      "/path/to/model.gguf",
 *     tokenizerPath:  "/path/to/tokenizer.json",
 *     optionsJson:    #"{"maxNewTokens":64,"sampling":{"preset":"greedy"}}"#
 * )
 *
 * let text = try model.generate("The capital of France is")
 * print(text)            // " Paris."
 *
 * let stats = try model.lastResultStats()
 * print(stats?.tokensPerSecond ?? 0, "tok/s")
 * // model.close() and rt.close() called automatically on deinit.
 * ```
 *
 * ## Threading
 *
 * Each ``SdxLlmRuntime`` handle is bound to the OS thread that created it.
 * Create, use, and destroy each runtime from the same thread. Use one runtime
 * per thread for concurrent generation; models are not shared across runtimes.
 *
 * ## Deployment
 *
 * Point the linker at the AOT SDK `lib/` directory:
 * ```bash
 * swift build -Xcc  -I$SDX_LLM_AOT_HOME/include \
 *             -Xlinker -L$SDX_LLM_AOT_HOME/lib
 * ```
 * The `module.modulemap` declares `link "sdx_llm"` so you do **not** need
 * `-lsdx_llm` explicitly — the Swift build system adds it.
 *
 * `libsdx_llm` side-loads its native companions (ND4J CPU kernels, tokenizers,
 * etc.) relative to the **host executable**. When the executable and lib/ are
 * not co-located, set `SDX_NATIVE_LIB_DIR` to `lib/` before the first call:
 * ```swift
 * setenv("SDX_NATIVE_LIB_DIR", "\(aotHome)/lib", 0)
 * ```
 * ``SdxLlmRuntime.init()`` does this automatically when `SDX_LLM_AOT_HOME` is
 * set and `SDX_NATIVE_LIB_DIR` is not.
 */

import Foundation
import CSdxLlm

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors thrown by the SdxLlm Swift wrapper.
public enum SdxLlmError: Error, CustomStringConvertible {
    /// The native layer returned a non-zero status code.
    case nativeStatus(code: SdxLlmStatusCode, message: String)
    /// ``sdxLlmCreateRuntime`` returned NULL.
    case runtimeCreationFailed

    public var description: String {
        switch self {
        case let .nativeStatus(code, message):
            return "SdxLlmError(\(code)): \(message)"
        case .runtimeCreationFailed:
            return "SdxLlmError: sdxLlmCreateRuntime returned NULL — " +
                   "check that SDX_NATIVE_LIB_DIR is set before the process starts."
        }
    }
}

// ── Status codes ──────────────────────────────────────────────────────────────

/// Swift mirror of `sdx_llm_status_t`.
public enum SdxLlmStatusCode: Int32, CustomStringConvertible {
    case ok              = 0
    case invalidArgument = 1
    case modelLoadFailed = 3
    case executionFailed = 4
    case ioError         = 6

    public var description: String {
        switch self {
        case .ok:              return "OK"
        case .invalidArgument: return "INVALID_ARGUMENT"
        case .modelLoadFailed: return "MODEL_LOAD_FAILED"
        case .executionFailed: return "EXECUTION_FAILED"
        case .ioError:         return "IO_ERROR"
        }
    }

    /// Returns the enum case for `rawValue`, or `.executionFailed` as a safe fallback.
    static func from(_ raw: sdx_llm_status_t) -> SdxLlmStatusCode {
        SdxLlmStatusCode(rawValue: raw.rawValue) ?? .executionFailed
    }
}

// ── Generation statistics ─────────────────────────────────────────────────────

/// Parsed subset of the JSON returned by ``sdxLlmLastResultJson``.
///
/// All fields are optional; the raw JSON is in `rawJson` for forward-compatibility.
public struct SdxLlmStats: CustomStringConvertible {
    /// Number of tokens generated (excluding prompt).
    public let newTokens: Int?
    /// Number of input prompt tokens.
    public let promptTokens: Int?
    /// Total wall-clock generation time in milliseconds.
    public let generationTimeMs: Double?
    /// Throughput in tokens per second.
    public let tokensPerSecond: Double?
    /// Why generation stopped (e.g. `"max_tokens"`, `"eos"`).
    public let finishReason: String?

    /// Parse from the raw JSON string returned by ``sdxLlmLastResultJson``.
    public init(json: String) {
        guard
            let data = json.data(using: .utf8),
            let obj  = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            newTokens = nil; promptTokens = nil
            generationTimeMs = nil; tokensPerSecond = nil
            finishReason = nil
            return
        }
        newTokens        = obj["newTokens"]        as? Int
        promptTokens     = obj["promptTokens"]     as? Int
        generationTimeMs = obj["generationTimeMs"] as? Double
        tokensPerSecond  = obj["tokensPerSec"]     as? Double
        finishReason     = obj["finishReason"]     as? String
    }

    public var description: String {
        var parts: [String] = []
        if let v = promptTokens    { parts.append("prompt=\(v) tokens") }
        if let v = newTokens       { parts.append("generated=\(v) tokens") }
        if let v = tokensPerSecond { parts.append(String(format: "%.1f tok/s", v)) }
        if let v = finishReason    { parts.append("finish=\(v)") }
        return "SdxLlmStats { \(parts.joined(separator: ", ")) }"
    }
}

// ── SdxLlmModel ───────────────────────────────────────────────────────────────

/// A loaded model pipeline.
///
/// Obtain via ``SdxLlmRuntime/loadModel(modelPath:tokenizerPath:optionsJson:)``.
/// The model owns the KV cache, sampling state, and compiled generation plan;
/// it reuses the compiled plan and KV state across calls to ``generate(_:optionsJson:)``.
///
/// Resources are released on `deinit` or by calling ``close()``.
public final class SdxLlmModel {
    private unowned let runtime: SdxLlmRuntime
    private var handle: OpaquePointer?

    fileprivate init(runtime: SdxLlmRuntime, handle: OpaquePointer?) {
        self.runtime = runtime
        self.handle  = handle
    }

    deinit { close() }

    // MARK: - Generation

    /// Generate text continuing `prompt`.
    ///
    /// - Parameters:
    ///   - prompt:      Input text (UTF-8).
    ///   - optionsJson: Optional per-call overrides, e.g.
    ///                  `{"maxNewTokens":N,"sampling":{"preset":"greedy"}}` or
    ///                  `{"temperature":0.8,"topK":40,"topP":0.9,"seed":42}`.
    ///                  Pass `nil` to use the load-time defaults.
    /// - Returns: The generated text (does **not** include the prompt).
    /// - Throws:  ``SdxLlmError/nativeStatus(code:message:)`` on failure.
    public func generate(_ prompt: String, optionsJson: String? = nil) throws -> String {
        var outText: UnsafeMutablePointer<CChar>? = nil
        let rawStatus = prompt.withCString { promptC in
            optionsJson?.withCString { optsC in
                sdxLlmGenerate(runtime.handle, handle, promptC, optsC, &outText)
            } ?? sdxLlmGenerate(runtime.handle, handle, promptC, nil, &outText)
        }
        let s = SdxLlmStatusCode.from(rawStatus)
        guard s == .ok, let ptr = outText else {
            throw SdxLlmError.nativeStatus(code: s, message: runtime.lastError())
        }
        defer { sdxLlmFree(runtime.handle, ptr) }
        return String(cString: ptr)
    }

    // MARK: - Tokenization

    /// Encode `text` to token IDs.
    ///
    /// - Parameters:
    ///   - text:            Input text (UTF-8).
    ///   - addSpecialTokens: When `true`, prepend/append BOS/EOS as appropriate.
    /// - Returns: Array of `Int32` token IDs.
    /// - Throws:  ``SdxLlmError/nativeStatus(code:message:)`` on failure.
    public func tokenize(_ text: String, addSpecialTokens: Bool = true) throws -> [Int32] {
        var ids: UnsafeMutablePointer<Int32>? = nil
        var count: Int32 = 0
        let rawStatus = text.withCString {
            sdxLlmTokenize(runtime.handle, handle, $0, addSpecialTokens ? 1 : 0, &ids, &count)
        }
        let s = SdxLlmStatusCode.from(rawStatus)
        guard s == .ok, let ptr = ids else {
            throw SdxLlmError.nativeStatus(code: s, message: runtime.lastError())
        }
        defer { sdxLlmFree(runtime.handle, ptr) }
        return Array(UnsafeBufferPointer(start: ptr, count: Int(count)))
    }

    /// Decode token IDs back to text.
    ///
    /// - Parameters:
    ///   - ids:               Token IDs to decode.
    ///   - skipSpecialTokens: When `true`, omit BOS/EOS/PAD from output.
    /// - Returns: Decoded UTF-8 string.
    /// - Throws:  ``SdxLlmError/nativeStatus(code:message:)`` on failure.
    public func detokenize(_ ids: [Int32], skipSpecialTokens: Bool = true) throws -> String {
        var outText: UnsafeMutablePointer<CChar>? = nil
        let rawStatus = ids.withUnsafeBufferPointer { buf in
            sdxLlmDetokenize(
                runtime.handle, handle,
                buf.baseAddress, Int32(ids.count),
                skipSpecialTokens ? 1 : 0,
                &outText)
        }
        let s = SdxLlmStatusCode.from(rawStatus)
        guard s == .ok, let ptr = outText else {
            throw SdxLlmError.nativeStatus(code: s, message: runtime.lastError())
        }
        defer { sdxLlmFree(runtime.handle, ptr) }
        return String(cString: ptr)
    }

    // MARK: - Metadata

    /// Statistics from the most recent ``generate(_:optionsJson:)`` call.
    ///
    /// - Returns: Parsed ``SdxLlmStats``, or `nil` if no generation has run yet.
    /// - Throws:  ``SdxLlmError/nativeStatus(code:message:)`` on failure.
    public func lastResultStats() throws -> SdxLlmStats? {
        var outJson: UnsafeMutablePointer<CChar>? = nil
        let rawStatus = sdxLlmLastResultJson(runtime.handle, handle, &outJson)
        let s = SdxLlmStatusCode.from(rawStatus)
        guard s == .ok else {
            throw SdxLlmError.nativeStatus(code: s, message: runtime.lastError())
        }
        guard let ptr = outJson else { return nil }
        defer { sdxLlmFree(runtime.handle, ptr) }
        return SdxLlmStats(json: String(cString: ptr))
    }

    /// Model/tokenizer summary JSON (inputs, outputs, vocab size, chat-template flag).
    ///
    /// - Throws: ``SdxLlmError/nativeStatus(code:message:)`` on failure.
    public func infoJson() throws -> String {
        var outJson: UnsafeMutablePointer<CChar>? = nil
        let rawStatus = sdxLlmInfoJson(runtime.handle, handle, &outJson)
        let s = SdxLlmStatusCode.from(rawStatus)
        guard s == .ok, let ptr = outJson else {
            throw SdxLlmError.nativeStatus(code: s, message: runtime.lastError())
        }
        defer { sdxLlmFree(runtime.handle, ptr) }
        return String(cString: ptr)
    }

    // MARK: - Resource management

    /// Release the native model handle.  Called automatically on `deinit`.
    public func close() {
        guard let h = handle else { return }
        _ = sdxLlmUnloadModel(runtime.handle, h)
        handle = nil
    }
}

// ── SdxLlmRuntime ─────────────────────────────────────────────────────────────

/// Root object for the SDX LLM AOT runtime.
///
/// Create one runtime per OS thread. The GraalVM isolate is bound to the
/// creating OS thread — never share a runtime across threads.
///
/// ```swift
/// let rt = try SdxLlmRuntime()
/// let model = try rt.loadModel(modelPath: "qwen.gguf",
///                              tokenizerPath: "tokenizer.json")
/// let text = try model.generate("The capital of France is")
/// ```
///
/// Resources are released automatically on `deinit` or explicitly via ``close()``.
public final class SdxLlmRuntime {
    fileprivate var handle: OpaquePointer?

    /// Create a runtime, automatically configuring `SDX_NATIVE_LIB_DIR` from
    /// `SDX_LLM_AOT_HOME` when the environment variable is not already set.
    ///
    /// - Throws: ``SdxLlmError/runtimeCreationFailed`` when the GraalVM isolate
    ///           cannot be initialised (e.g. the shared library is not found or
    ///           `SDX_NATIVE_LIB_DIR` was not exported before the process started).
    public init() throws {
        // Ensure side-loaded natives are found.
        if let aotHome = ProcessInfo.processInfo.environment["SDX_LLM_AOT_HOME"],
           ProcessInfo.processInfo.environment["SDX_NATIVE_LIB_DIR"] == nil {
            let libDir = (aotHome as NSString).appendingPathComponent("lib")
            setenv("SDX_NATIVE_LIB_DIR", libDir, 0)
        }

        guard let h = sdxLlmCreateRuntime() else {
            throw SdxLlmError.runtimeCreationFailed
        }
        self.handle = h
    }

    deinit { close() }

    // MARK: - ABI

    /// ABI version compiled into the linked `libsdx_llm`.
    public func abiVersion() -> Int32 {
        sdxLlmAbiVersion(handle)
    }

    // MARK: - Model lifecycle

    /// Load a model and build the generation pipeline.
    ///
    /// - Parameters:
    ///   - modelPath:     Path to a `.gguf`/`.ggml`/`.sdz`/`.sdnb` file.
    ///   - tokenizerPath: Path to a `tokenizer.json` file or directory.
    ///                    Pass `nil` to let the library search the model's directory.
    ///   - optionsJson:   Optional JSON options, e.g.
    ///                    `{"maxNewTokens":128,"graphOptimizer":true,`
    ///                    `"sampling":{"preset":"greedy"}}` or
    ///                    `{"temperature":0.8,"topK":40,"topP":0.9}`.
    ///                    Pass `nil` to use built-in defaults.
    /// - Returns: A loaded ``SdxLlmModel`` ready to generate.
    /// - Throws:  ``SdxLlmError/nativeStatus(code:message:)`` on failure.
    public func loadModel(
        modelPath: String,
        tokenizerPath: String? = nil,
        optionsJson: String? = nil
    ) throws -> SdxLlmModel {
        let modelHandle: OpaquePointer? = modelPath.withCString { mp in
            let tok: OpaquePointer? = tokenizerPath.flatMap { tp in
                tp.withCString { tpC in
                    optionsJson.flatMap { oj in
                        oj.withCString { ojC in
                            OpaquePointer(sdxLlmLoadModel(handle, mp, tpC, ojC))
                        }
                    } ?? OpaquePointer(sdxLlmLoadModel(handle, mp, tpC, nil))
                }
            }
            if tok != nil { return tok }
            return optionsJson.flatMap { oj in
                oj.withCString { ojC in
                    OpaquePointer(sdxLlmLoadModel(handle, mp, nil, ojC))
                }
            } ?? OpaquePointer(sdxLlmLoadModel(handle, mp, nil, nil))
        }

        guard let h = modelHandle else {
            throw SdxLlmError.nativeStatus(
                code: .modelLoadFailed,
                message: lastError())
        }
        return SdxLlmModel(runtime: self, handle: h)
    }

    // MARK: - VLM and audio (stateless, convenience)

    /// Extract content from an image or PDF using the SmolDocling VLM.
    ///
    /// Each call loads the model, processes the input, and releases the model.
    ///
    /// - Parameters:
    ///   - modelPath:     Path to the SmolDocling model directory or bundle.
    ///   - tokenizerPath: Path to the tokenizer (or `nil` for auto-discovery).
    ///   - inputPath:     Path to the image (PNG/JPEG) or PDF file.
    ///   - optionsJson:   Optional JSON, e.g. `{"format":"markdown","maxNewTokens":512}`.
    /// - Returns: Extracted text in the requested format.
    /// - Throws:  ``SdxLlmError/nativeStatus(code:message:)`` on failure.
    public func vlmExtract(
        modelPath: String,
        tokenizerPath: String? = nil,
        inputPath: String,
        optionsJson: String? = nil
    ) throws -> String {
        var outText: UnsafeMutablePointer<CChar>? = nil
        let rawStatus = modelPath.withCString { mp in
            inputPath.withCString { ip in
                let inner: sdx_llm_status_t = tokenizerPath.flatMap { tp in
                    tp.withCString { tpC in
                        optionsJson.flatMap { oj in
                            oj.withCString { ojC in
                                sdxVlmExtract(handle, mp, tpC, ip, ojC, &outText)
                            }
                        } ?? sdxVlmExtract(handle, mp, tpC, ip, nil, &outText)
                    }
                } ?? (optionsJson.flatMap { oj in
                    oj.withCString { ojC in sdxVlmExtract(handle, mp, nil, ip, ojC, &outText) }
                } ?? sdxVlmExtract(handle, mp, nil, ip, nil, &outText))
                return inner
            }
        }
        let s = SdxLlmStatusCode.from(rawStatus)
        guard s == .ok, let ptr = outText else {
            throw SdxLlmError.nativeStatus(code: s, message: lastError())
        }
        defer { sdxLlmFree(handle, ptr) }
        return String(cString: ptr)
    }

    /// Transcribe audio using a Whisper model (stateless, loads and releases per call).
    ///
    /// - Parameters:
    ///   - modelPath:   Path to the Whisper ONNX model directory.
    ///   - audioPath:   Path to the audio file (WAV, 16 kHz mono recommended).
    ///   - optionsJson: Optional JSON, e.g. `{"language":"en","maxNewTokens":448}`.
    /// - Returns: Transcribed text.
    /// - Throws:  ``SdxLlmError/nativeStatus(code:message:)`` on failure.
    public func audioTranscribe(
        modelPath: String,
        audioPath: String,
        optionsJson: String? = nil
    ) throws -> String {
        var outText: UnsafeMutablePointer<CChar>? = nil
        let rawStatus = modelPath.withCString { mp in
            audioPath.withCString { ap in
                optionsJson.flatMap { oj in
                    oj.withCString { ojC in sdxAudioTranscribe(handle, mp, ap, ojC, &outText) }
                } ?? sdxAudioTranscribe(handle, mp, ap, nil, &outText)
            }
        }
        let s = SdxLlmStatusCode.from(rawStatus)
        guard s == .ok, let ptr = outText else {
            throw SdxLlmError.nativeStatus(code: s, message: lastError())
        }
        defer { sdxLlmFree(handle, ptr) }
        return String(cString: ptr)
    }

    // MARK: - Errors

    /// The last error message from the native runtime, or an empty string.
    public func lastError() -> String {
        var buf = [CChar](repeating: 0, count: 4096)
        _ = sdxLlmGetLastError(handle, &buf, Int32(buf.count))
        return String(cString: buf)
    }

    // MARK: - Resource management

    /// Tear down the GraalVM isolate.  Called automatically on `deinit`.
    ///
    /// All ``SdxLlmModel`` handles created from this runtime become invalid
    /// after this call.  Always close models before the runtime.
    public func close() {
        guard let h = handle else { return }
        _ = sdxLlmDestroyRuntime(h)
        handle = nil
    }
}
