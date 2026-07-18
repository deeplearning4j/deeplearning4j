// ******************************************************************************
//
// This program and the accompanying materials are made available under the
// terms of the Apache License, Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

//! Safe Rust wrapper over the `libsdx_llm` C ABI (`sdxLlm*` / `sdxVlm*` /
//! `sdxAudio*` symbols exported by `libsdx_llm.so`).
//!
//! Follows the same conventions as the parent `sdx_runtime` crate:
//! opaque handles wrapped in `Drop`-implementing structs, a `#[non_exhaustive]`
//! `LlmError` enum, and `Result<_, LlmError>` throughout.
//!
//! The C ABI surface is defined in
//! `nd4j/sdx-aot/include/sdx_llm_c.h`. See ADR 0109.
//!
//! ## Threading
//!
//! A [`LlmRuntime`] handle is **bound to the OS thread that created it**.
//! Create, use, and drop each runtime from one thread.  For concurrent
//! generation create one `LlmRuntime` per thread.
//!
//! ## Library resolution
//!
//! Binaries that link this module resolve `libsdx_llm.so` via a `build.rs`
//! that emits `cargo:rustc-link-search=native=…` only when the `llm` feature
//! is active.  The search order is:
//! 1. `SDX_LLM_LIB_DIR` env var (explicit directory).
//! 2. `SDX_LLM_AOT_HOME/lib` (unpacked AOT SDK package layout).
//! 3. System linker path (pkg-config / LD_LIBRARY_PATH).
//!
//! ## Native side-loading
//!
//! `libsdx_llm.so` resolves its bundled native libraries (ND4J backend,
//! BLAS, tokenizers, …) relative to the host executable, not the `.so`.
//! Call [`ensure_native_lib_dir`] once before the first [`LlmRuntime::new`]
//! call to set `SDX_NATIVE_LIB_DIR` automatically.

#![allow(non_camel_case_types)]
#![allow(dead_code)]  // public API items — used by callers, not internally

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

// ── Status codes ─────────────────────────────────────────────────────────────

pub const SDX_LLM_OK: i32 = 0;

/// Mirror of `sdx_llm_status_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[repr(i32)]
pub enum StatusCode {
    Ok               = 0,
    InvalidArgument  = 1,
    ModelLoadFailed  = 3,
    ExecutionFailed  = 4,
    IoError          = 6,
    Unknown          = -1,
}

impl StatusCode {
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => Self::Ok,
            1 => Self::InvalidArgument,
            3 => Self::ModelLoadFailed,
            4 => Self::ExecutionFailed,
            6 => Self::IoError,
            _ => Self::Unknown,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Ok               => "OK",
            Self::InvalidArgument  => "INVALID_ARGUMENT",
            Self::ModelLoadFailed  => "MODEL_LOAD_FAILED",
            Self::ExecutionFailed  => "EXECUTION_FAILED",
            Self::IoError          => "IO_ERROR",
            Self::Unknown          => "UNKNOWN",
        }
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

/// All errors surfaced by this module.
///
/// Pattern-match to handle specific failure modes:
///
/// ```no_run
/// use sdx_runtime::llm::{LlmRuntime, LlmError, StatusCode};
///
/// match LlmRuntime::new() {
///     Err(LlmError::RuntimeInit { .. }) => eprintln!("init failed"),
///     Err(e) => eprintln!("error: {e}"),
///     Ok(rt) => { /* use rt */ }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum LlmError {
    /// Runtime initialisation failed (`sdxLlmCreateRuntime` returned NULL).
    RuntimeInit { detail: String },

    /// Model load failed.
    ModelLoad { path: String, status: StatusCode, detail: String },

    /// Generation failed.
    Generate { status: StatusCode, detail: String },

    /// Tokenization or detokenization failed.
    Tokenize { status: StatusCode, detail: String },

    /// A C ABI call returned non-zero for the given operation.
    Api { op: &'static str, status: StatusCode, detail: String },

    /// A string argument could not be converted to a C string (embedded NUL).
    InvalidString(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::RuntimeInit { detail } =>
                write!(f, "runtime init failed: {detail}"),
            LlmError::ModelLoad { path, status, detail } =>
                write!(f, "model load failed for '{path}' ({name}): {detail}",
                       name = status.name()),
            LlmError::Generate { status, detail } =>
                write!(f, "generate failed ({name}): {detail}", name = status.name()),
            LlmError::Tokenize { status, detail } =>
                write!(f, "tokenize failed ({name}): {detail}", name = status.name()),
            LlmError::Api { op, status, detail } =>
                write!(f, "{op} failed ({name}): {detail}", name = status.name()),
            LlmError::InvalidString(msg) =>
                write!(f, "invalid string argument: {msg}"),
        }
    }
}

impl std::error::Error for LlmError {}

// ── Opaque C handles ─────────────────────────────────────────────────────────

#[repr(C)]
pub struct sdx_llm_runtime_t { _private: [u8; 0] }

#[repr(C)]
pub struct sdx_llm_model_t { _private: [u8; 0] }

// ── FFI declarations ─────────────────────────────────────────────────────────

#[link(name = "sdx_llm")]
extern "C" {
    fn sdxLlmCreateRuntime() -> *mut sdx_llm_runtime_t;
    fn sdxLlmDestroyRuntime(runtime: *mut sdx_llm_runtime_t) -> c_int;
    fn sdxLlmAbiVersion(runtime: *mut sdx_llm_runtime_t) -> c_int;

    fn sdxLlmLoadModel(
        runtime: *mut sdx_llm_runtime_t,
        model_path: *const c_char,
        tokenizer_path: *const c_char,
        options_json: *const c_char,
    ) -> *mut sdx_llm_model_t;

    fn sdxLlmUnloadModel(
        runtime: *mut sdx_llm_runtime_t,
        model: *mut sdx_llm_model_t,
    ) -> i32;

    fn sdxLlmGenerate(
        runtime: *mut sdx_llm_runtime_t,
        model: *mut sdx_llm_model_t,
        prompt: *const c_char,
        options_json: *const c_char,
        out_text: *mut *mut c_char,
    ) -> i32;

    fn sdxLlmLastResultJson(
        runtime: *mut sdx_llm_runtime_t,
        model: *mut sdx_llm_model_t,
        out_json: *mut *mut c_char,
    ) -> i32;

    fn sdxLlmInfoJson(
        runtime: *mut sdx_llm_runtime_t,
        model: *mut sdx_llm_model_t,
        out_json: *mut *mut c_char,
    ) -> i32;

    fn sdxLlmTokenize(
        runtime: *mut sdx_llm_runtime_t,
        model: *mut sdx_llm_model_t,
        text: *const c_char,
        add_special_tokens: i32,
        out_ids: *mut *mut i32,
        out_count: *mut i32,
    ) -> i32;

    fn sdxLlmDetokenize(
        runtime: *mut sdx_llm_runtime_t,
        model: *mut sdx_llm_model_t,
        ids: *const i32,
        count: i32,
        skip_special_tokens: i32,
        out_text: *mut *mut c_char,
    ) -> i32;

    fn sdxVlmExtract(
        runtime: *mut sdx_llm_runtime_t,
        model_path: *const c_char,
        tokenizer_path: *const c_char,
        input_path: *const c_char,
        options_json: *const c_char,
        out_text: *mut *mut c_char,
    ) -> i32;

    fn sdxAudioTranscribe(
        runtime: *mut sdx_llm_runtime_t,
        model_path: *const c_char,
        audio_path: *const c_char,
        options_json: *const c_char,
        out_text: *mut *mut c_char,
    ) -> i32;

    fn sdxLlmFree(runtime: *mut sdx_llm_runtime_t, pointer: *mut c_void);

    fn sdxLlmGetLastError(
        runtime: *mut sdx_llm_runtime_t,
        buffer: *mut c_char,
        capacity: i32,
    ) -> i32;
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn last_error(runtime: *mut sdx_llm_runtime_t) -> String {
    let mut buf = vec![0u8; 4096];
    let n = unsafe {
        sdxLlmGetLastError(runtime, buf.as_mut_ptr() as *mut c_char, buf.len() as i32)
    };
    if n <= 0 { return String::new(); }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

fn to_cstring(s: &str, ctx: &str) -> Result<CString, LlmError> {
    CString::new(s).map_err(|e| LlmError::InvalidString(format!("{ctx}: {e}")))
}

fn opt_cstring(s: Option<&str>, ctx: &str) -> Result<Option<CString>, LlmError> {
    match s {
        None    => Ok(None),
        Some(v) => to_cstring(v, ctx).map(Some),
    }
}

fn raw_ptr(cs: &Option<CString>) -> *const c_char {
    cs.as_ref().map_or(ptr::null(), |s| s.as_ptr())
}

fn check(status: i32, runtime: *mut sdx_llm_runtime_t, op: &'static str)
    -> Result<(), LlmError>
{
    if status == SDX_LLM_OK { return Ok(()); }
    let sc = StatusCode::from_i32(status);
    Err(LlmError::Api { op, status: sc, detail: last_error(runtime) })
}

/// Collect a NUL-terminated C string returned via `*mut *mut c_char` into a
/// Rust `String` and immediately free the allocation with `sdxLlmFree`.
unsafe fn take_string(
    out_ptr: *mut c_char,
    runtime: *mut sdx_llm_runtime_t,
) -> String {
    if out_ptr.is_null() { return String::new(); }
    let s = CStr::from_ptr(out_ptr).to_string_lossy().into_owned();
    sdxLlmFree(runtime, out_ptr as *mut c_void);
    s
}

// ── Native lib dir ────────────────────────────────────────────────────────────

/// Set `SDX_NATIVE_LIB_DIR` to `$SDX_LLM_AOT_HOME/lib` if not already set.
///
/// Call once before [`LlmRuntime::new`].  `libsdx_llm.so` resolves its
/// bundled native libraries (ND4J backend, BLAS, tokenizers) relative to the
/// host executable, not the `.so`.  This env var overrides that lookup to
/// point at the correct directory.
pub fn ensure_native_lib_dir() {
    if std::env::var_os("SDX_NATIVE_LIB_DIR").is_some() { return; }
    if let Ok(home) = std::env::var("SDX_LLM_AOT_HOME") {
        let lib_dir = Path::new(&home).join("lib");
        // SAFETY: single-threaded init; called before LlmRuntime::new().
        #[allow(deprecated)]
        std::env::set_var("SDX_NATIVE_LIB_DIR", &lib_dir);
    }
}

// ── GenerateStats ─────────────────────────────────────────────────────────────

/// Stats from the most recent [`LlmModel::generate`] call.
///
/// Parsed from the JSON payload of `sdxLlmLastResultJson`.  Unknown fields
/// are silently ignored so future server additions do not break this code.
///
/// JSON field names (camelCase, as emitted by `SdxLlmCore.lastResultJson`):
/// `promptTokens`, `generatedTokens`, `totalTokens`, `generationTimeMs`,
/// `firstTokenLatencyMs`, `tokensPerSecond`, `finishReason`.
#[derive(Debug, Default, Clone)]
pub struct GenerateStats {
    pub prompt_tokens:          i64,
    pub generated_tokens:       i64,
    pub total_tokens:           i64,
    pub generation_time_ms:     f64,
    pub first_token_latency_ms: f64,
    pub tokens_per_sec:         f64,
    pub finish_reason:          String,
}

impl GenerateStats {
    /// Parse a `sdxLlmLastResultJson` payload.
    ///
    /// Uses a simple hand-rolled parser to avoid a `serde_json` dependency.
    pub fn from_json(json: &str) -> Self {
        fn extract_f64(json: &str, key: &str) -> f64 {
            let needle = format!("\"{key}\"");
            if let Some(pos) = json.find(needle.as_str()) {
                let after = &json[pos + needle.len()..];
                if let Some(colon) = after.find(':') {
                    let val = after[colon + 1..].trim();
                    let end = val.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-'
                                       && c != 'e' && c != 'E' && c != '+')
                        .unwrap_or(val.len());
                    if let Ok(v) = val[..end].parse::<f64>() { return v; }
                }
            }
            0.0
        }
        fn extract_str(json: &str, key: &str) -> String {
            let needle = format!("\"{key}\"");
            if let Some(pos) = json.find(needle.as_str()) {
                let after = &json[pos + needle.len()..];
                if let Some(q1) = after.find('"') {
                    let rest = &after[q1 + 1..];
                    if let Some(q2) = rest.find('"') {
                        return rest[..q2].to_string();
                    }
                }
            }
            String::new()
        }
        // Field names match SdxLlmCore.lastResultJson exactly (camelCase).
        Self {
            prompt_tokens:          extract_f64(json, "promptTokens") as i64,
            generated_tokens:       extract_f64(json, "generatedTokens") as i64,
            total_tokens:           extract_f64(json, "totalTokens") as i64,
            generation_time_ms:     extract_f64(json, "generationTimeMs"),
            first_token_latency_ms: extract_f64(json, "firstTokenLatencyMs"),
            tokens_per_sec:         extract_f64(json, "tokensPerSecond"),
            finish_reason:          extract_str(json, "finishReason"),
        }
    }
}

// ── LlmRuntime ────────────────────────────────────────────────────────────────

/// SDX LLM runtime handle — entry point for loading models.
///
/// Created with [`LlmRuntime::new`].  Dropping calls `sdxLlmDestroyRuntime`.
///
/// # Threading
///
/// A runtime is **bound to the OS thread that created it**.  Create, use, and
/// drop from the same thread.  For concurrent generation create one
/// `LlmRuntime` per thread.
#[must_use]
pub struct LlmRuntime {
    ptr: *mut sdx_llm_runtime_t,
}

impl LlmRuntime {
    /// Create a runtime with default options.
    pub fn new() -> Result<Self, LlmError> {
        let ptr = unsafe { sdxLlmCreateRuntime() };
        if ptr.is_null() {
            return Err(LlmError::RuntimeInit {
                detail: "sdxLlmCreateRuntime returned NULL".into(),
            });
        }
        Ok(Self { ptr })
    }

    /// ABI version compiled into the loaded library.
    pub fn abi_version(&self) -> i32 {
        unsafe { sdxLlmAbiVersion(self.ptr) }
    }

    /// The last error string from the runtime, or an empty string.
    pub fn last_error(&self) -> String { last_error(self.ptr) }

    /// Load a model and return an [`LlmModel`] handle.
    ///
    /// # Parameters
    ///
    /// - `model_path`: path to a `.gguf`/`.ggml` or `.sdz`/`.sdnb`/`.fb` file.
    /// - `tokenizer_path`: optional `tokenizer.json` (file or directory);
    ///   `None` tries the model's directory.
    /// - `options_json`: optional JSON, e.g.
    ///   `r#"{"maxNewTokens":128,"sampling":{"preset":"greedy"}}"#`.
    pub fn load_model(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: Option<&str>,
        options_json: Option<&str>,
    ) -> Result<LlmModel, LlmError> {
        let path_str = model_path.as_ref().to_string_lossy();
        let c_model   = to_cstring(&path_str, "model_path")?;
        let c_tok     = opt_cstring(tokenizer_path, "tokenizer_path")?;
        let c_opts    = opt_cstring(options_json, "options_json")?;

        let model_ptr = unsafe {
            sdxLlmLoadModel(
                self.ptr,
                c_model.as_ptr(),
                raw_ptr(&c_tok),
                raw_ptr(&c_opts),
            )
        };
        if model_ptr.is_null() {
            return Err(LlmError::ModelLoad {
                path: path_str.into_owned(),
                status: StatusCode::ModelLoadFailed,
                detail: last_error(self.ptr),
            });
        }
        Ok(LlmModel { runtime: self.ptr, ptr: model_ptr })
    }

    /// VLM document extraction (SmolDocling): image/PDF → doctags/markdown/text.
    ///
    /// Stateless — loads and releases the model per call.
    pub fn vlm_extract(
        &self,
        model_path: impl AsRef<Path>,
        tokenizer_path: Option<&str>,
        input_path: impl AsRef<Path>,
        options_json: Option<&str>,
    ) -> Result<String, LlmError> {
        let c_model  = to_cstring(&model_path.as_ref().to_string_lossy(), "model_path")?;
        let c_tok    = opt_cstring(tokenizer_path, "tokenizer_path")?;
        let c_input  = to_cstring(&input_path.as_ref().to_string_lossy(), "input_path")?;
        let c_opts   = opt_cstring(options_json, "options_json")?;

        let mut out: *mut c_char = ptr::null_mut();
        let status = unsafe {
            sdxVlmExtract(
                self.ptr,
                c_model.as_ptr(),
                raw_ptr(&c_tok),
                c_input.as_ptr(),
                raw_ptr(&c_opts),
                &mut out,
            )
        };
        check(status, self.ptr, "sdxVlmExtract")?;
        Ok(unsafe { take_string(out, self.ptr) })
    }

    /// Whisper speech-to-text.
    ///
    /// Stateless — loads and releases the model per call.
    pub fn audio_transcribe(
        &self,
        model_path: impl AsRef<Path>,
        audio_path: impl AsRef<Path>,
        options_json: Option<&str>,
    ) -> Result<String, LlmError> {
        let c_model  = to_cstring(&model_path.as_ref().to_string_lossy(), "model_path")?;
        let c_audio  = to_cstring(&audio_path.as_ref().to_string_lossy(), "audio_path")?;
        let c_opts   = opt_cstring(options_json, "options_json")?;

        let mut out: *mut c_char = ptr::null_mut();
        let status = unsafe {
            sdxAudioTranscribe(
                self.ptr,
                c_model.as_ptr(),
                c_audio.as_ptr(),
                raw_ptr(&c_opts),
                &mut out,
            )
        };
        check(status, self.ptr, "sdxAudioTranscribe")?;
        Ok(unsafe { take_string(out, self.ptr) })
    }
}

impl Drop for LlmRuntime {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sdxLlmDestroyRuntime(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

// Safety: the underlying GraalVM isolate handle is opaque.
// The ABI contract is single-threaded per isolate; callers must not share
// one LlmRuntime across threads.
unsafe impl Send for LlmRuntime {}

// ── LlmModel ──────────────────────────────────────────────────────────────────

/// A loaded LLM model.
///
/// Returned by [`LlmRuntime::load_model`].  Dropping calls `sdxLlmUnloadModel`.
///
/// The compiled plan and KV state persist across [`LlmModel::generate`] calls,
/// so repeated calls reuse the warm pipeline — no reload cost after the first.
#[must_use]
pub struct LlmModel {
    runtime: *mut sdx_llm_runtime_t,
    ptr:     *mut sdx_llm_model_t,
}

impl LlmModel {
    /// Blocking text generation.
    ///
    /// # Parameters
    ///
    /// - `prompt`: input text (chat template applied when configured).
    /// - `options_json`: per-call override, e.g.
    ///   `r#"{"maxNewTokens":64,"sampling":{"preset":"greedy"}}"#`.
    ///   `None` = load-time defaults.
    pub fn generate(
        &self,
        prompt: &str,
        options_json: Option<&str>,
    ) -> Result<String, LlmError> {
        let c_prompt = to_cstring(prompt, "prompt")?;
        let c_opts   = opt_cstring(options_json, "options_json")?;
        let mut out: *mut c_char = ptr::null_mut();
        let status = unsafe {
            sdxLlmGenerate(
                self.runtime,
                self.ptr,
                c_prompt.as_ptr(),
                raw_ptr(&c_opts),
                &mut out,
            )
        };
        if status != SDX_LLM_OK {
            return Err(LlmError::Generate {
                status: StatusCode::from_i32(status),
                detail: last_error(self.runtime),
            });
        }
        Ok(unsafe { take_string(out, self.runtime) })
    }

    /// Return stats from the most recent `generate` call.
    ///
    /// Call immediately after [`generate`][LlmModel::generate] before
    /// issuing the next call.
    pub fn last_result(&self) -> Result<GenerateStats, LlmError> {
        let mut out: *mut c_char = ptr::null_mut();
        let status = unsafe {
            sdxLlmLastResultJson(self.runtime, self.ptr, &mut out)
        };
        check(status, self.runtime, "sdxLlmLastResultJson")?;
        let json = unsafe { take_string(out, self.runtime) };
        Ok(GenerateStats::from_json(&json))
    }

    /// Model/tokenizer summary as a raw JSON string.
    ///
    /// Keys include vocab size, input/output names, and chat-template flag.
    pub fn info_json(&self) -> Result<String, LlmError> {
        let mut out: *mut c_char = ptr::null_mut();
        let status = unsafe { sdxLlmInfoJson(self.runtime, self.ptr, &mut out) };
        check(status, self.runtime, "sdxLlmInfoJson")?;
        Ok(unsafe { take_string(out, self.runtime) })
    }

    /// Encode UTF-8 text to a `Vec<i32>` of token IDs.
    pub fn tokenize(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, LlmError> {
        let c_text = to_cstring(text, "tokenize text")?;
        let mut out_ids: *mut i32 = ptr::null_mut();
        let mut out_count: i32 = 0;
        let status = unsafe {
            sdxLlmTokenize(
                self.runtime,
                self.ptr,
                c_text.as_ptr(),
                if add_special_tokens { 1 } else { 0 },
                &mut out_ids,
                &mut out_count,
            )
        };
        if status != SDX_LLM_OK {
            return Err(LlmError::Tokenize {
                status: StatusCode::from_i32(status),
                detail: last_error(self.runtime),
            });
        }
        let ids: Vec<i32> = if out_ids.is_null() || out_count <= 0 {
            Vec::new()
        } else {
            // SAFETY: pointer and count are valid per ABI contract.
            let slice = unsafe { std::slice::from_raw_parts(out_ids, out_count as usize) };
            let v = slice.to_vec();
            unsafe { sdxLlmFree(self.runtime, out_ids as *mut c_void) };
            v
        };
        Ok(ids)
    }

    /// Decode a slice of token IDs to UTF-8 text.
    pub fn detokenize(
        &self,
        ids: &[i32],
        skip_special_tokens: bool,
    ) -> Result<String, LlmError> {
        let mut out: *mut c_char = ptr::null_mut();
        let status = unsafe {
            sdxLlmDetokenize(
                self.runtime,
                self.ptr,
                ids.as_ptr(),
                ids.len() as i32,
                if skip_special_tokens { 1 } else { 0 },
                &mut out,
            )
        };
        if status != SDX_LLM_OK {
            return Err(LlmError::Tokenize {
                status: StatusCode::from_i32(status),
                detail: last_error(self.runtime),
            });
        }
        Ok(unsafe { take_string(out, self.runtime) })
    }
}

impl Drop for LlmModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sdxLlmUnloadModel(self.runtime, self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

unsafe impl Send for LlmModel {}
