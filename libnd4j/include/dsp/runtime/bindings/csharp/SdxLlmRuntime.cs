// SPDX-License-Identifier: Apache-2.0
//
// SdxLlmRuntime.cs — C# P/Invoke wrapper for the SDX LLM C ABI (sdx_llm_c.h).
//
// Binds libsdx_llm.so, the AOT-compiled (GraalVM native-image) LLM/VLM/STT
// library.  No JVM is embedded in the library.  This wrapper is dependency-free:
// it imports only System.* and System.Runtime.InteropServices.
//
// Threading (v1): each SdxLlmRuntime is bound to the OS thread that created it.
// Create, use, and destroy a runtime from one thread.  For concurrent generation
// use one SdxLlmRuntime per thread — models are not shared across runtimes.
//
// Side-loaded natives (C# CAN set process env):
//   SdxLlmRuntime.Create() sets Environment.SetEnvironmentVariable("SDX_NATIVE_LIB_DIR", ...)
//   when SDX_LLM_AOT_HOME is set, so callers do not need to manage this manually.

#nullable enable

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Nd4j.Dsp.Runtime.Llm
{

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

/// <summary>Status codes returned by the SDX LLM C ABI.</summary>
public static class SdxLlmConstants
{
    /// <summary>ABI version this wrapper was written against.</summary>
    public const int SDX_LLM_ABI_VERSION = 1;

    /// <summary>Successful completion.</summary>
    public const int SDX_LLM_STATUS_OK = 0;
    /// <summary>Null or invalid argument.</summary>
    public const int SDX_LLM_STATUS_INVALID_ARGUMENT = 1;
    /// <summary>Model file not found or not parseable.</summary>
    public const int SDX_LLM_STATUS_MODEL_LOAD_FAILED = 3;
    /// <summary>Generation or tokenization failed at runtime.</summary>
    public const int SDX_LLM_STATUS_EXECUTION_FAILED = 4;
    /// <summary>I/O error (file read, audio decode, …).</summary>
    public const int SDX_LLM_STATUS_IO_ERROR = 6;
}

// ---------------------------------------------------------------------------
// Generation result stats DTO
// ---------------------------------------------------------------------------

/// <summary>
/// Immutable telemetry snapshot for a completed generation call.
/// Parsed from the JSON string returned by <c>sdxLlmLastResultJson</c>.
/// Value equality and auto-formatted <c>ToString</c> are synthesised by the
/// record compiler — no boilerplate needed.
/// </summary>
/// <param name="PromptTokens">Number of input (prompt) tokens processed.</param>
/// <param name="NewTokens">Number of tokens generated.</param>
/// <param name="TokensPerSec">Generation throughput in tokens/second; null if not reported.</param>
/// <param name="FinishReason">Why generation stopped (e.g. <c>"max_tokens"</c>, <c>"eos"</c>).</param>
/// <param name="RawJson">Full JSON string from <c>sdxLlmLastResultJson</c>.</param>
public record LlmResultStats(
    int? PromptTokens,
    int? NewTokens,
    double? TokensPerSec,
    string? FinishReason,
    string RawJson);

// ---------------------------------------------------------------------------
// Native entry-point declarations
// ---------------------------------------------------------------------------

/// <summary>
/// P/Invoke signatures for <c>sdx_llm_c.h</c>.  Internal — callers use
/// <see cref="SdxLlmRuntime"/> and <see cref="SdxLlmModel"/>.
/// </summary>
/// <remarks>
/// The library path is resolved at first call via
/// <see cref="NativeLibrary.SetDllImportResolver"/> registered in
/// <see cref="SdxLlmRuntime.Create()"/>.
/// </remarks>
internal static class SdxLlmNative
{
    internal const string LIB = "sdx_llm";

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr sdxLlmCreateRuntime();

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxLlmDestroyRuntime(IntPtr runtime);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxLlmAbiVersion(IntPtr runtime);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr sdxLlmLoadModel(
        IntPtr runtime,
        string model_path,
        [MarshalAs(UnmanagedType.LPStr)] string? tokenizer_path,
        [MarshalAs(UnmanagedType.LPStr)] string? options_json);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxLlmUnloadModel(IntPtr runtime, IntPtr model);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern int sdxLlmGenerate(
        IntPtr runtime,
        IntPtr model,
        string prompt,
        [MarshalAs(UnmanagedType.LPStr)] string? options_json,
        out IntPtr out_text);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxLlmLastResultJson(
        IntPtr runtime,
        IntPtr model,
        out IntPtr out_json);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxLlmInfoJson(
        IntPtr runtime,
        IntPtr model,
        out IntPtr out_json);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern int sdxLlmTokenize(
        IntPtr runtime,
        IntPtr model,
        string text,
        int add_special_tokens,
        out IntPtr out_ids,
        out int out_count);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxLlmDetokenize(
        IntPtr runtime,
        IntPtr model,
        int[] ids,
        int count,
        int skip_special_tokens,
        out IntPtr out_text);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern int sdxVlmExtract(
        IntPtr runtime,
        string model_path,
        [MarshalAs(UnmanagedType.LPStr)] string? tokenizer_path,
        string input_path,
        [MarshalAs(UnmanagedType.LPStr)] string? options_json,
        out IntPtr out_text);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern int sdxAudioTranscribe(
        IntPtr runtime,
        string model_path,
        string audio_path,
        [MarshalAs(UnmanagedType.LPStr)] string? options_json,
        out IntPtr out_text);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void sdxLlmFree(IntPtr runtime, IntPtr pointer);

    [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int sdxLlmGetLastError(IntPtr runtime, byte[]? buffer, int capacity);
}

// ---------------------------------------------------------------------------
// SdxLlmRuntime — top-level lifecycle object
// ---------------------------------------------------------------------------

/// <summary>
/// Global environment for the SDX LLM C ABI — analogous to
/// <c>OrtEnvironment</c> in the ONNX Runtime C# API.
///
/// <para>
/// Binds <c>libsdx_llm.so</c>, the AOT-compiled (GraalVM native-image) LLM
/// library, via P/Invoke.  No JVM is embedded — the entire Java LLM stack was
/// compiled ahead-of-time into a plain C ABI.  The point of this example is
/// showing that a .NET host can embed the AOT library without any Java runtime
/// on the system.
/// </para>
///
/// <para>
/// Unlike the JVM wrappers, this C# class <strong>can</strong> set
/// <c>SDX_NATIVE_LIB_DIR</c> in process environment before the first P/Invoke
/// call — <c>Environment.SetEnvironmentVariable</c> propagates to
/// <c>getenv</c> on Linux.  <see cref="Create"/> does this automatically when
/// <c>SDX_LLM_AOT_HOME</c> is set.
/// </para>
///
/// <example>
/// <code>
/// using var runtime = SdxLlmRuntime.Create();
/// using var model   = runtime.LoadModel(modelPath, tokenizerPath);
/// string text = model.Generate(
///     "The capital of France is",
///     """{"maxNewTokens":8,"sampling":{"preset":"greedy"}}""");
/// Console.WriteLine(text);  // " Paris."
/// </code>
/// </example>
/// </summary>
public sealed class SdxLlmRuntime : IDisposable
{
    private IntPtr _runtime;
    private bool _disposed;

    // NativeLibrary resolver: registered once per process.
    private static bool _resolverRegistered;
    private static readonly object _resolverLock = new();
    private static string? _resolvedLibPath;

    private SdxLlmRuntime(IntPtr runtime) => _runtime = runtime;

    // ── Factory ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a runtime by auto-detecting <c>libsdx_llm.so</c>.
    ///
    /// <para>Resolution order (first match wins):</para>
    /// <list type="number">
    ///   <item>Environment variable <c>SDX_LLM_LIBRARY</c> — explicit absolute path.</item>
    ///   <item><c>SDX_LLM_AOT_HOME/lib/libsdx_llm.so</c>.</item>
    ///   <item>System <c>LD_LIBRARY_PATH</c> / <c>PATH</c> default search.</item>
    /// </list>
    ///
    /// <para>
    /// When <c>SDX_LLM_AOT_HOME</c> is set and <c>SDX_NATIVE_LIB_DIR</c> is not,
    /// this method sets <c>SDX_NATIVE_LIB_DIR</c> to <c>SDX_LLM_AOT_HOME/lib</c>
    /// before the first P/Invoke call — unlike JVM wrappers, C# <em>can</em>
    /// propagate environment variables to native code at runtime.
    /// </para>
    /// </summary>
    /// <param name="libraryPath">
    /// Optional explicit path to <c>libsdx_llm.so</c>; <see langword="null"/> triggers auto-detection.
    /// </param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the library cannot be loaded or the runtime cannot be initialised.
    /// </exception>
    public static SdxLlmRuntime Create(string? libraryPath = null)
    {
        EnsureNativeLibDirSet();
        RegisterResolver(libraryPath ?? ResolveLibraryPath());

        var rt = SdxLlmNative.sdxLlmCreateRuntime();
        if (rt == IntPtr.Zero)
            throw new InvalidOperationException(
                "sdxLlmCreateRuntime() returned null. " +
                "Ensure SDX_LLM_AOT_HOME is set and the library and its side-loads are present.");
        return new SdxLlmRuntime(rt);
    }

    // ── API ───────────────────────────────────────────────────────────────────

    /// <summary>ABI version reported by the loaded library.</summary>
    public int AbiVersion()
    {
        CheckNotDisposed();
        return SdxLlmNative.sdxLlmAbiVersion(_runtime);
    }

    /// <summary>
    /// Loads a GGUF/SDZ model and builds the generation pipeline.
    /// Analogous to <c>new InferenceSession(modelPath, options)</c> in ONNX Runtime.
    ///
    /// <para>
    /// First load compiles the DSP plan (1–3 min on CPU); subsequent loads reuse
    /// the cached plan.
    /// </para>
    /// </summary>
    /// <param name="modelPath">
    /// Path to a <c>.gguf</c> / <c>.ggml</c> / <c>.sdz</c> model file.
    /// </param>
    /// <param name="tokenizerPath">
    /// Path to <c>tokenizer.json</c> or its directory; <see langword="null"/> probes
    /// the model's directory.
    /// </param>
    /// <param name="optionsJson">
    /// Optional generation defaults JSON, e.g.
    /// <c>{"maxNewTokens":128,"sampling":{"preset":"greedy"}}</c>;
    /// <see langword="null"/> for library defaults.
    /// </param>
    /// <returns>A new <see cref="SdxLlmModel"/>; dispose it when done.</returns>
    /// <exception cref="InvalidOperationException">Thrown if loading fails.</exception>
    public SdxLlmModel LoadModel(
        string modelPath,
        string? tokenizerPath = null,
        string? optionsJson = null)
    {
        CheckNotDisposed();
        var handle = SdxLlmNative.sdxLlmLoadModel(_runtime, modelPath, tokenizerPath, optionsJson);
        if (handle == IntPtr.Zero)
            throw new InvalidOperationException(
                $"sdxLlmLoadModel failed: {LastError()}\n  model_path={modelPath}");
        return new SdxLlmModel(this, handle);
    }

    /// <summary>Returns the last error message from the native runtime, or an empty string.</summary>
    public string LastError()
    {
        if (_runtime == IntPtr.Zero) return string.Empty;
        var buf = new byte[2048];
        int len = SdxLlmNative.sdxLlmGetLastError(_runtime, buf, buf.Length);
        if (len <= 0) return string.Empty;
        return Encoding.UTF8.GetString(buf, 0, Math.Min(len, buf.Length - 1));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_runtime != IntPtr.Zero)
        {
            SdxLlmNative.sdxLlmDestroyRuntime(_runtime);
            _runtime = IntPtr.Zero;
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    internal IntPtr RuntimeHandle => _runtime;

    private void CheckNotDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(SdxLlmRuntime));
    }

    /// <summary>
    /// Sets <c>SDX_NATIVE_LIB_DIR</c> when <c>SDX_LLM_AOT_HOME</c> is present.
    /// Unlike JVM code, C# <c>Environment.SetEnvironmentVariable</c> propagates
    /// to <c>getenv()</c> in the same process on Linux.
    /// </summary>
    private static void EnsureNativeLibDirSet()
    {
        var aotHome = Environment.GetEnvironmentVariable("SDX_LLM_AOT_HOME");
        if (string.IsNullOrEmpty(aotHome)) return;

        var existing = Environment.GetEnvironmentVariable("SDX_NATIVE_LIB_DIR");
        if (!string.IsNullOrEmpty(existing)) return;

        var libDir = Path.Combine(aotHome, "lib");
        if (Directory.Exists(libDir))
        {
            Environment.SetEnvironmentVariable("SDX_NATIVE_LIB_DIR", libDir);
            Console.Error.WriteLine(
                $"[SdxLlmRuntime] Set SDX_NATIVE_LIB_DIR={libDir}");
        }
    }

    private static string ResolveLibraryPath()
    {
        // 1) Explicit env override.
        var explicit_ = Environment.GetEnvironmentVariable("SDX_LLM_LIBRARY");
        if (!string.IsNullOrEmpty(explicit_)) return explicit_!;

        // 2) AOT SDK home.
        var aotHome = Environment.GetEnvironmentVariable("SDX_LLM_AOT_HOME");
        if (!string.IsNullOrEmpty(aotHome))
        {
            foreach (var name in new[] { "libsdx_llm.so", "sdx_llm.so", "libsdx_llm.dylib", "sdx_llm.dll" })
            {
                var candidate = Path.Combine(aotHome!, "lib", name);
                if (File.Exists(candidate)) return candidate;
            }
        }

        // 3) System default.
        return SdxLlmNative.LIB;
    }

    private static void RegisterResolver(string libraryPath)
    {
        lock (_resolverLock)
        {
            if (_resolverRegistered) return;
            _resolverRegistered = true;
            _resolvedLibPath = libraryPath;

            NativeLibrary.SetDllImportResolver(
                typeof(SdxLlmNative).Assembly,
                (libName, assembly, searchPath) =>
                {
                    if (libName != SdxLlmNative.LIB) return IntPtr.Zero;
                    var path = _resolvedLibPath;
                    if (!string.IsNullOrEmpty(path) && File.Exists(path))
                    {
                        // Also add the library's directory to LD_LIBRARY_PATH so
                        // any transitive dlopen calls from the native side succeed.
                        var dir = Path.GetDirectoryName(path);
                        if (!string.IsNullOrEmpty(dir))
                        {
                            var current = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "";
                            if (!current.Contains(dir!))
                                Environment.SetEnvironmentVariable("LD_LIBRARY_PATH",
                                    string.IsNullOrEmpty(current) ? dir : $"{dir}:{current}");
                        }
                        if (NativeLibrary.TryLoad(path!, out var handle)) return handle;
                    }
                    return NativeLibrary.Load(libName, assembly, searchPath);
                });
        }
    }

    // ── Internal: readAndFree ─────────────────────────────────────────────────

    internal string ReadAndFree(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) return string.Empty;
        try
        {
            return Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
        }
        finally
        {
            SdxLlmNative.sdxLlmFree(_runtime, ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// SdxLlmModel — per-model lifecycle object
// ---------------------------------------------------------------------------

/// <summary>
/// A loaded LLM/VLM model.  Obtain via <see cref="SdxLlmRuntime.LoadModel"/>;
/// dispose via <c>using</c> declarations or <c>using(…){}</c> blocks.
///
/// <para>
/// <b>Threading:</b> the model handle is bound to the same OS thread as its
/// parent <see cref="SdxLlmRuntime"/>.
/// </para>
/// </summary>
public sealed class SdxLlmModel : IDisposable
{
    private readonly SdxLlmRuntime _rt;
    private IntPtr _model;
    private bool _disposed;

    internal SdxLlmModel(SdxLlmRuntime rt, IntPtr model)
    {
        _rt    = rt;
        _model = model;
    }

    // ── Generation ────────────────────────────────────────────────────────────

    /// <summary>
    /// Generates a text continuation for <paramref name="prompt"/>.
    ///
    /// <para>
    /// Blocking call.  First call compiles the DSP plan (warmup); subsequent
    /// calls reuse the captured plan and KV state.
    /// </para>
    /// </summary>
    /// <param name="prompt">Input text (UTF-8).</param>
    /// <param name="optionsJson">
    /// Per-call generation overrides, e.g.
    /// <c>{"maxNewTokens":64,"sampling":{"temperature":0.8}}</c>;
    /// <see langword="null"/> uses load-time defaults.
    /// </param>
    /// <returns>The generated continuation (not the prompt).</returns>
    /// <exception cref="InvalidOperationException">Thrown if generation fails.</exception>
    public string Generate(string prompt, string? optionsJson = null)
    {
        CheckNotDisposed();
        int status = SdxLlmNative.sdxLlmGenerate(
            _rt.RuntimeHandle, _model, prompt, optionsJson, out IntPtr outText);
        if (status != SdxLlmConstants.SDX_LLM_STATUS_OK)
            throw new InvalidOperationException(
                $"sdxLlmGenerate failed (status={status}): {_rt.LastError()}");
        return _rt.ReadAndFree(outText);
    }

    // ── Stats + info ──────────────────────────────────────────────────────────

    /// <summary>
    /// Returns a <see cref="LlmResultStats"/> record parsed from the JSON stats
    /// of the most recent <see cref="Generate"/> call.
    /// </summary>
    public LlmResultStats LastResultStats()
    {
        var json = LastResultJson();
        return new LlmResultStats(
            PromptTokens: ExtractInt(json, "promptTokens"),
            NewTokens:    ExtractInt(json, "newTokens"),
            TokensPerSec: ExtractDouble(json, "tokensPerSec"),
            FinishReason: ExtractString(json, "finishReason"),
            RawJson:      json);
    }

    /// <summary>
    /// Raw JSON stats string from <c>sdxLlmLastResultJson</c>.
    /// Callers normally use <see cref="LastResultStats"/> instead.
    /// </summary>
    public string LastResultJson()
    {
        CheckNotDisposed();
        int status = SdxLlmNative.sdxLlmLastResultJson(
            _rt.RuntimeHandle, _model, out IntPtr outJson);
        if (status != SdxLlmConstants.SDX_LLM_STATUS_OK)
            throw new InvalidOperationException(
                $"sdxLlmLastResultJson failed (status={status}): {_rt.LastError()}");
        return _rt.ReadAndFree(outJson);
    }

    /// <summary>
    /// JSON summary of this model: inputs, outputs, vocab size, chat template flag.
    /// </summary>
    public string InfoJson()
    {
        CheckNotDisposed();
        int status = SdxLlmNative.sdxLlmInfoJson(
            _rt.RuntimeHandle, _model, out IntPtr outJson);
        if (status != SdxLlmConstants.SDX_LLM_STATUS_OK)
            throw new InvalidOperationException(
                $"sdxLlmInfoJson failed (status={status}): {_rt.LastError()}");
        return _rt.ReadAndFree(outJson);
    }

    // ── Tokenization ──────────────────────────────────────────────────────────

    /// <summary>Encodes <paramref name="text"/> to an array of token IDs.</summary>
    /// <param name="text">Input text (UTF-8).</param>
    /// <param name="addSpecialTokens">
    /// <see langword="true"/> to prepend/append BOS/EOS tokens.
    /// </param>
    /// <returns>Array of token IDs.</returns>
    public int[] Tokenize(string text, bool addSpecialTokens = false)
    {
        CheckNotDisposed();
        int status = SdxLlmNative.sdxLlmTokenize(
            _rt.RuntimeHandle, _model, text,
            addSpecialTokens ? 1 : 0,
            out IntPtr outIds, out int count);
        if (status != SdxLlmConstants.SDX_LLM_STATUS_OK)
            throw new InvalidOperationException(
                $"sdxLlmTokenize failed (status={status}): {_rt.LastError()}");
        var ids = new int[count];
        Marshal.Copy(outIds, ids, 0, count);
        SdxLlmNative.sdxLlmFree(_rt.RuntimeHandle, outIds);
        return ids;
    }

    /// <summary>Decodes <paramref name="ids"/> back to text.</summary>
    /// <param name="ids">Token ID array.</param>
    /// <param name="skipSpecialTokens">
    /// <see langword="true"/> to omit BOS/EOS/pad tokens from the output.
    /// </param>
    /// <returns>Decoded text (UTF-8).</returns>
    public string Detokenize(int[] ids, bool skipSpecialTokens = true)
    {
        CheckNotDisposed();
        int status = SdxLlmNative.sdxLlmDetokenize(
            _rt.RuntimeHandle, _model,
            ids, ids.Length,
            skipSpecialTokens ? 1 : 0,
            out IntPtr outText);
        if (status != SdxLlmConstants.SDX_LLM_STATUS_OK)
            throw new InvalidOperationException(
                $"sdxLlmDetokenize failed (status={status}): {_rt.LastError()}");
        return _rt.ReadAndFree(outText);
    }

    // ── Resource management ───────────────────────────────────────────────────

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_model != IntPtr.Zero)
        {
            SdxLlmNative.sdxLlmUnloadModel(_rt.RuntimeHandle, _model);
            _model = IntPtr.Zero;
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private void CheckNotDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(SdxLlmModel));
    }

    private static int? ExtractInt(string json, string key)
    {
        var pat = $"\"{key}\"\\s*:\\s*(-?\\d+)";
        var m = System.Text.RegularExpressions.Regex.Match(json, pat);
        return m.Success && int.TryParse(m.Groups[1].Value, out var v) ? v : (int?)null;
    }

    private static double? ExtractDouble(string json, string key)
    {
        var pat = $"\"{key}\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)";
        var m = System.Text.RegularExpressions.Regex.Match(json, pat);
        return m.Success && double.TryParse(m.Groups[1].Value,
            System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : (double?)null;
    }

    private static string? ExtractString(string json, string key)
    {
        var pat = $"\"{key}\"\\s*:\\s*\"([^\"\\\\]*)\"";
        var m = System.Text.RegularExpressions.Regex.Match(json, pat);
        return m.Success ? m.Groups[1].Value : null;
    }
}

// ---------------------------------------------------------------------------
// Top-level helpers: VLM + audio (stateless)
// ---------------------------------------------------------------------------

/// <summary>
/// Top-level VLM and audio helpers that are stateless — they load and release
/// the model per call via the C ABI's <c>sdxVlmExtract</c> /
/// <c>sdxAudioTranscribe</c> functions.
/// </summary>
public static class SdxLlmHelpers
{
    /// <summary>
    /// VLM document extraction (SmolDocling): image/PDF → doctags/markdown/text.
    /// </summary>
    /// <param name="runtime">An open <see cref="SdxLlmRuntime"/>.</param>
    /// <param name="modelPath">Path to the SmolDocling model directory.</param>
    /// <param name="inputPath">Path to an image (PNG/JPEG) or PDF file.</param>
    /// <param name="tokenizerPath">Path to <c>tokenizer.json</c> or <see langword="null"/>.</param>
    /// <param name="optionsJson">e.g. <c>{"maxNewTokens":512,"format":"doctags"}</c>.</param>
    /// <returns>Extracted text in the requested format.</returns>
    public static string VlmExtract(
        SdxLlmRuntime runtime,
        string modelPath,
        string inputPath,
        string? tokenizerPath = null,
        string? optionsJson = null)
    {
        int status = SdxLlmNative.sdxVlmExtract(
            runtime.RuntimeHandle, modelPath, tokenizerPath, inputPath, optionsJson,
            out IntPtr outText);
        if (status != SdxLlmConstants.SDX_LLM_STATUS_OK)
            throw new InvalidOperationException(
                $"sdxVlmExtract failed (status={status}): {runtime.LastError()}");
        return runtime.ReadAndFree(outText);
    }

    /// <summary>
    /// Whisper speech-to-text — stateless, loads and releases the model per call.
    /// </summary>
    /// <param name="runtime">An open <see cref="SdxLlmRuntime"/>.</param>
    /// <param name="modelPath">Path to a Whisper ONNX model directory.</param>
    /// <param name="audioPath">Path to a WAV file.</param>
    /// <param name="optionsJson">e.g. <c>{"language":"en","maxNewTokens":448}</c>.</param>
    /// <returns>Transcription text (UTF-8).</returns>
    public static string AudioTranscribe(
        SdxLlmRuntime runtime,
        string modelPath,
        string audioPath,
        string? optionsJson = null)
    {
        int status = SdxLlmNative.sdxAudioTranscribe(
            runtime.RuntimeHandle, modelPath, audioPath, optionsJson, out IntPtr outText);
        if (status != SdxLlmConstants.SDX_LLM_STATUS_OK)
            throw new InvalidOperationException(
                $"sdxAudioTranscribe failed (status={status}): {runtime.LastError()}");
        return runtime.ReadAndFree(outText);
    }
}

} // namespace Nd4j.Dsp.Runtime.Llm
