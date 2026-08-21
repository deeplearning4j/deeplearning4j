/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//! C FFI bindings for HuggingFace tokenizers library
//!
//! This crate provides C-compatible functions that wrap the tokenizers library,
//! allowing it to be called from C/C++/Java via JNI.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use minijinja::{Environment, Error as MiniJinjaError, ErrorKind};
use serde_json::{Map as JsonMap, Value as JsonValue};
use tokenizers::tokenizer::step_decode_stream;
use tokenizers::Tokenizer;

/// Thread-local error storage
thread_local! {
    static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.lock().unwrap() = Some(msg);
    });
}

fn clear_error() {
    LAST_ERROR.with(|e| {
        *e.lock().unwrap() = None;
    });
}

fn required_utf8(value: *const c_char, name: &str) -> Result<String, String> {
    if value.is_null() {
        return Err(format!("{} cannot be null", name));
    }
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(str::to_owned)
        .map_err(|error| format!("{} is not valid UTF-8: {}", name, error))
}

fn normalize_special_token_values(context: &mut JsonMap<String, JsonValue>) {
    for (name, value) in context.iter_mut() {
        if !name.ends_with("_token") {
            continue;
        }
        let content = value
            .as_object()
            .and_then(|object| object.get("content"))
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        if let Some(content) = content {
            *value = JsonValue::String(content);
        }
    }
}

fn validate_chat_messages(messages: &JsonValue) -> Result<(), String> {
    let messages_array = messages
        .as_array()
        .ok_or_else(|| "chat template context.messages must be an array".to_string())?;
    if messages_array.is_empty() {
        return Err("chat template context.messages must contain at least one message".to_string());
    }
    for (index, message) in messages_array.iter().enumerate() {
        let object = message
            .as_object()
            .ok_or_else(|| format!("messages[{}] must be an object", index))?;
        if object
            .get("role")
            .and_then(JsonValue::as_str)
            .filter(|role| !role.trim().is_empty())
            .is_none()
        {
            return Err(format!("messages[{}].role must be a string", index));
        }
        if !object.contains_key("content") && !object.contains_key("tool_calls") {
            return Err(format!(
                "messages[{}] must define content or tool_calls",
                index
            ));
        }
    }
    Ok(())
}

fn render_chat_template_context(
    context_json: &str,
    tokenizer_config_json: &str,
    current_date: &str,
) -> Result<String, String> {
    let runtime: JsonValue = serde_json::from_str(context_json)
        .map_err(|error| format!("chat template context JSON is invalid: {}", error))?;
    let runtime = runtime
        .as_object()
        .cloned()
        .ok_or_else(|| "chat template context JSON must contain an object".to_string())?;
    let messages = runtime
        .get("messages")
        .ok_or_else(|| "chat template context must define messages".to_string())?;
    validate_chat_messages(messages)?;

    let config: JsonValue = serde_json::from_str(tokenizer_config_json)
        .map_err(|error| format!("tokenizer_config.json is invalid: {}", error))?;
    let mut context = config
        .as_object()
        .cloned()
        .ok_or_else(|| "tokenizer_config.json must contain a JSON object".to_string())?;
    let template = context
        .get("chat_template")
        .and_then(JsonValue::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            "tokenizer_config.json must define a non-empty string chat_template".to_string()
        })?;
    normalize_special_token_values(&mut context);
    for (name, value) in runtime {
        if name != "chat_template" {
            context.insert(name, value);
        }
    }
    context
        .entry("tools".to_string())
        .or_insert(JsonValue::Null);
    context
        .entry("documents".to_string())
        .or_insert(JsonValue::Null);
    context
        .entry("add_generation_prompt".to_string())
        .or_insert(JsonValue::Bool(false));
    context
        .entry("continue_final_message".to_string())
        .or_insert(JsonValue::Bool(false));
    context
        .entry("date_string".to_string())
        .or_insert_with(|| JsonValue::String(current_date.to_string()));

    // Hugging Face uses Jinja's default undefined behavior for chat templates.
    // Optional message members such as reasoning_content and tool_calls must be
    // falsey when absent; model-authored raise_exception calls still fail below.
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_function(
        "raise_exception",
        |message: String| -> Result<String, MiniJinjaError> {
            Err(MiniJinjaError::new(ErrorKind::InvalidOperation, message))
        },
    );
    let compiled = environment
        .template_from_str(&template)
        .map_err(|error| format!("chat_template could not be compiled: {:#}", error))?;
    compiled
        .render(JsonValue::Object(context))
        .map_err(|error| format!("chat_template could not be rendered: {:#}", error))
}

fn render_chat_template(
    messages_json: &str,
    tokenizer_config_json: &str,
    current_date: &str,
    add_generation_prompt: bool,
) -> Result<String, String> {
    let messages: JsonValue = serde_json::from_str(messages_json)
        .map_err(|error| format!("messages JSON is invalid: {}", error))?;
    let mut context = JsonMap::new();
    context.insert("messages".to_string(), messages);
    context.insert("tools".to_string(), JsonValue::Null);
    context.insert("documents".to_string(), JsonValue::Null);
    context.insert(
        "add_generation_prompt".to_string(),
        JsonValue::Bool(add_generation_prompt),
    );
    context.insert(
        "continue_final_message".to_string(),
        JsonValue::Bool(false),
    );
    render_chat_template_context(
        &JsonValue::Object(context).to_string(),
        tokenizer_config_json,
        current_date,
    )
}

/// Opaque tokenizer handle
pub struct TokenizerHandle {
    tokenizer: Arc<Tokenizer>,
}

/// Opaque stateful decoder handle exported through the C ABI.
///
/// Keep every piece of incremental decode state owned by the handle. This
/// avoids manufacturing a `'static` reference into the retained tokenizer and
/// makes ordinary Rust drop order sufficient for cleanup.
pub struct DecodeStreamHandle {
    tokenizer: Arc<Tokenizer>,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

/// Opaque encoding handle
pub struct EncodingHandle {
    ids: Vec<u32>,
    tokens: Vec<String>,
    offsets: Vec<(usize, usize)>,
    // Cache for C strings
    token_ptrs: Vec<*const c_char>,
    token_cstrings: Vec<CString>,
}

// ============================================================================
// Core Tokenizer Functions
// ============================================================================

/// Create a tokenizer from a file path (tokenizer.json)
#[no_mangle]
pub extern "C" fn ffi_tokenizer_from_file(path: *const c_char) -> *mut TokenizerHandle {
    clear_error();

    if path.is_null() {
        set_error("Path cannot be null".to_string());
        return ptr::null_mut();
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("Invalid UTF-8 in path: {}", e));
            return ptr::null_mut();
        }
    };

    match Tokenizer::from_file(path_str) {
        Ok(tokenizer) => Box::into_raw(Box::new(TokenizerHandle {
            tokenizer: Arc::new(tokenizer),
        })),
        Err(e) => {
            set_error(format!("Failed to load tokenizer from file: {}", e));
            ptr::null_mut()
        }
    }
}

/// Create a tokenizer from JSON string
#[no_mangle]
pub extern "C" fn ffi_tokenizer_from_json(json: *const c_char) -> *mut TokenizerHandle {
    clear_error();

    if json.is_null() {
        set_error("JSON cannot be null".to_string());
        return ptr::null_mut();
    }

    let json_str = match unsafe { CStr::from_ptr(json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("Invalid UTF-8 in JSON: {}", e));
            return ptr::null_mut();
        }
    };

    match Tokenizer::from_str(json_str) {
        Ok(tokenizer) => Box::into_raw(Box::new(TokenizerHandle {
            tokenizer: Arc::new(tokenizer),
        })),
        Err(e) => {
            set_error(format!("Failed to create tokenizer from JSON: {}", e));
            ptr::null_mut()
        }
    }
}

/// Free a tokenizer handle
#[no_mangle]
pub extern "C" fn ffi_tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Check if a tokenizer handle is valid
#[no_mangle]
pub extern "C" fn ffi_tokenizer_is_valid(handle: *const TokenizerHandle) -> bool {
    !handle.is_null()
}

/// Get vocabulary size
#[no_mangle]
pub extern "C" fn ffi_tokenizer_get_vocab_size(handle: *const TokenizerHandle) -> usize {
    if handle.is_null() {
        return 0;
    }

    let tokenizer = unsafe { &(*handle).tokenizer };
    tokenizer.get_vocab_size(true)
}

/// Resolve an exact vocabulary token to its ID without running normalization,
/// pre-tokenization, or unknown-token substitution.
#[no_mangle]
pub extern "C" fn ffi_tokenizer_token_to_id(
    handle: *const TokenizerHandle,
    token: *const c_char,
    token_id: *mut u32,
) -> bool {
    clear_error();
    if handle.is_null() {
        set_error("Tokenizer handle is null".to_string());
        return false;
    }
    if token.is_null() {
        set_error("Token cannot be null".to_string());
        return false;
    }
    if token_id.is_null() {
        set_error("Token ID output cannot be null".to_string());
        return false;
    }

    let token = match unsafe { CStr::from_ptr(token) }.to_str() {
        Ok(value) => value,
        Err(error) => {
            set_error(format!("Invalid UTF-8 in token: {}", error));
            return false;
        }
    };
    let tokenizer = unsafe { &(*handle).tokenizer };
    match tokenizer.token_to_id(token) {
        Some(id) => {
            unsafe { *token_id = id };
            true
        }
        None => false,
    }
}

/// Render a Hugging Face tokenizer_config.json chat_template with MiniJinja.
///
/// The returned UTF-8 string is owned by the caller and must be released with
/// ffi_tokenizer_free_string. The full tokenizer configuration is used as the
/// template context so model-specific special tokens remain authoritative.
#[no_mangle]
pub extern "C" fn ffi_tokenizer_apply_chat_template(
    messages_json: *const c_char,
    tokenizer_config_json: *const c_char,
    current_date: *const c_char,
    add_generation_prompt: bool,
) -> *mut c_char {
    clear_error();
    let rendered = (|| {
        let messages_json = required_utf8(messages_json, "messages JSON")?;
        let tokenizer_config_json =
            required_utf8(tokenizer_config_json, "tokenizer_config.json")?;
        let current_date = required_utf8(current_date, "current date")?;
        render_chat_template(
            &messages_json,
            &tokenizer_config_json,
            &current_date,
            add_generation_prompt,
        )
    })();

    match rendered {
        Ok(rendered) => match CString::new(rendered) {
            Ok(value) => value.into_raw(),
            Err(error) => {
                set_error(format!("rendered chat prompt contains a NUL byte: {}", error));
                ptr::null_mut()
            }
        },
        Err(error) => {
            set_error(error);
            ptr::null_mut()
        }
    }
}

/// Render a Hugging Face chat template from its complete runtime context.
///
/// Unlike ffi_tokenizer_apply_chat_template, this entry point preserves
/// structured messages, tools, documents, and future template keyword
/// arguments instead of projecting the request down to role/content strings.
#[no_mangle]
pub extern "C" fn ffi_tokenizer_apply_chat_template_context(
    context_json: *const c_char,
    tokenizer_config_json: *const c_char,
    current_date: *const c_char,
) -> *mut c_char {
    clear_error();
    let rendered = (|| {
        let context_json = required_utf8(context_json, "chat template context JSON")?;
        let tokenizer_config_json =
            required_utf8(tokenizer_config_json, "tokenizer_config.json")?;
        let current_date = required_utf8(current_date, "current date")?;
        render_chat_template_context(
            &context_json,
            &tokenizer_config_json,
            &current_date,
        )
    })();

    match rendered {
        Ok(rendered) => match CString::new(rendered) {
            Ok(value) => value.into_raw(),
            Err(error) => {
                set_error(format!("rendered chat prompt contains a NUL byte: {}", error));
                ptr::null_mut()
            }
        },
        Err(error) => {
            set_error(error);
            ptr::null_mut()
        }
    }
}

// ============================================================================
// Encoding Functions
// ============================================================================

/// Encode text into tokens
#[no_mangle]
pub extern "C" fn ffi_tokenizer_encode(
    handle: *const TokenizerHandle,
    text: *const c_char,
    add_special_tokens: bool,
) -> *mut EncodingHandle {
    clear_error();

    if handle.is_null() {
        set_error("Tokenizer handle is null".to_string());
        return ptr::null_mut();
    }

    if text.is_null() {
        set_error("Text cannot be null".to_string());
        return ptr::null_mut();
    }

    let text_str = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("Invalid UTF-8 in text: {}", e));
            return ptr::null_mut();
        }
    };

    let tokenizer = unsafe { &(*handle).tokenizer };

    match tokenizer.encode(text_str, add_special_tokens) {
        Ok(encoding) => {
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            let tokens: Vec<String> = encoding.get_tokens().to_vec();
            let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

            // Pre-convert tokens to C strings for later access
            let token_cstrings: Vec<CString> = tokens
                .iter()
                .map(|t| CString::new(t.as_str()).unwrap_or_else(|_| CString::new("").unwrap()))
                .collect();
            let token_ptrs: Vec<*const c_char> = token_cstrings.iter().map(|cs| cs.as_ptr()).collect();

            Box::into_raw(Box::new(EncodingHandle {
                ids,
                tokens,
                offsets,
                token_ptrs,
                token_cstrings,
            }))
        }
        Err(e) => {
            set_error(format!("Failed to encode text: {}", e));
            ptr::null_mut()
        }
    }
}

/// Free an encoding handle
#[no_mangle]
pub extern "C" fn ffi_encoding_free(handle: *mut EncodingHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Get the number of tokens in an encoding
#[no_mangle]
pub extern "C" fn ffi_encoding_get_length(handle: *const EncodingHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).ids.len() }
}

/// Get token IDs from an encoding
#[no_mangle]
pub extern "C" fn ffi_encoding_get_ids(handle: *const EncodingHandle) -> *const u32 {
    if handle.is_null() {
        return ptr::null();
    }
    unsafe { (*handle).ids.as_ptr() }
}

/// Get token strings from an encoding
#[no_mangle]
pub extern "C" fn ffi_encoding_get_tokens(handle: *const EncodingHandle) -> *const *const c_char {
    if handle.is_null() {
        return ptr::null();
    }
    unsafe { (*handle).token_ptrs.as_ptr() }
}

// ============================================================================
// Decoding Functions
// ============================================================================

/// Decode token IDs back to text
/// Returns a newly allocated string that must be freed with ffi_tokenizer_free_string
#[no_mangle]
pub extern "C" fn ffi_tokenizer_decode(
    handle: *const TokenizerHandle,
    ids: *const u32,
    num_ids: usize,
    skip_special_tokens: bool,
) -> *mut c_char {
    clear_error();

    if handle.is_null() {
        set_error("Tokenizer handle is null".to_string());
        return ptr::null_mut();
    }

    if ids.is_null() || num_ids == 0 {
        set_error("IDs array cannot be null or empty".to_string());
        return ptr::null_mut();
    }

    let ids_slice = unsafe { slice::from_raw_parts(ids, num_ids) };
    let tokenizer = unsafe { &(*handle).tokenizer };

    match tokenizer.decode(ids_slice, skip_special_tokens) {
        Ok(decoded) => match CString::new(decoded) {
            Ok(cs) => cs.into_raw(),
            Err(e) => {
                set_error(format!("Failed to convert decoded string: {}", e));
                ptr::null_mut()
            }
        },
        Err(e) => {
            set_error(format!("Failed to decode: {}", e));
            ptr::null_mut()
        }
    }
}

/// Create a stateful incremental decoder.
///
/// The returned handle owns an Arc reference to the tokenizer, so it remains
/// valid even if the originating tokenizer handle is released first.
#[no_mangle]
pub extern "C" fn ffi_tokenizer_decode_stream_create(
    handle: *const TokenizerHandle,
    skip_special_tokens: bool,
) -> *mut DecodeStreamHandle {
    clear_error();

    if handle.is_null() {
        set_error("Tokenizer handle is null".to_string());
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(DecodeStreamHandle {
        tokenizer: unsafe { Arc::clone(&(*handle).tokenizer) },
        skip_special_tokens,
        ids: Vec::new(),
        prefix: String::new(),
        prefix_index: 0,
    }))
}

/// Decode one token ID and return the next complete text chunk.
///
/// A successful step that cannot emit text yet returns an allocated empty
/// string. The returned string must be released with
/// ffi_tokenizer_free_string.
#[no_mangle]
pub extern "C" fn ffi_tokenizer_decode_stream_step(
    handle: *mut DecodeStreamHandle,
    token_id: u32,
) -> *mut c_char {
    clear_error();

    if handle.is_null() {
        set_error("Decode stream handle is null".to_string());
        return ptr::null_mut();
    }

    let DecodeStreamHandle {
        tokenizer,
        skip_special_tokens,
        ids,
        prefix,
        prefix_index,
    } = unsafe { &mut *handle };
    match step_decode_stream(
        tokenizer.as_ref(),
        token_id,
        *skip_special_tokens,
        ids,
        prefix,
        prefix_index,
    ) {
        Ok(chunk) => {
            let chunk = chunk.unwrap_or_default();
            match CString::new(chunk) {
                Ok(cs) => cs.into_raw(),
                Err(e) => {
                    set_error(format!("Failed to convert decoded chunk: {}", e));
                    ptr::null_mut()
                }
            }
        }
        Err(e) => {
            set_error(format!("Failed to decode stream token: {}", e));
            ptr::null_mut()
        }
    }
}

/// Free a stateful incremental decoder.
#[no_mangle]
pub extern "C" fn ffi_tokenizer_decode_stream_free(handle: *mut DecodeStreamHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Free a string returned by ffi_tokenizer_decode
#[no_mangle]
pub extern "C" fn ffi_tokenizer_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

// ============================================================================
// Error Handling
// ============================================================================

/// Get the last error message
/// Returns null if no error, otherwise a string that must NOT be freed
#[no_mangle]
pub extern "C" fn ffi_tokenizer_get_last_error() -> *const c_char {
    thread_local! {
        static ERROR_CSTRING: Mutex<Option<CString>> = Mutex::new(None);
    }

    LAST_ERROR.with(|e| {
        let error = e.lock().unwrap();
        match &*error {
            Some(msg) => {
                ERROR_CSTRING.with(|ec| {
                    let mut ec = ec.lock().unwrap();
                    *ec = CString::new(msg.as_str()).ok();
                    ec.as_ref().map(|cs| cs.as_ptr()).unwrap_or(ptr::null())
                })
            }
            None => ptr::null(),
        }
    })
}

/// Clear the last error
#[no_mangle]
pub extern "C" fn ffi_tokenizer_clear_error() {
    clear_error();
}

// ============================================================================
// Version Information
// ============================================================================

/// Get version string (do not free)
#[no_mangle]
pub extern "C" fn ffi_tokenizer_get_version() -> *const c_char {
    static VERSION: &[u8] = b"0.21.0\0";
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::{
        ffi_tokenizer_decode_stream_create, ffi_tokenizer_decode_stream_free,
        ffi_tokenizer_decode_stream_step, ffi_tokenizer_free, ffi_tokenizer_free_string,
        render_chat_template, render_chat_template_context, TokenizerHandle,
    };
    use std::ffi::CStr;
    use std::str::FromStr;
    use std::sync::Arc;
    use tokenizers::models::wordpiece::WordPiece;
    use tokenizers::Tokenizer;

    fn word_level_tokenizer_handle() -> *mut TokenizerHandle {
        let vocab = [
            ("hello".to_string(), 0),
            ("world".to_string(), 1),
            ("[UNK]".to_string(), 2),
        ];
        let model = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        Box::into_raw(Box::new(TokenizerHandle {
            tokenizer: Arc::new(Tokenizer::new(model)),
        }))
    }

    #[test]
    fn decode_stream_owns_tokenizer_and_drops_after_steps() {
        for _ in 0..128 {
            let tokenizer = word_level_tokenizer_handle();
            let stream = ffi_tokenizer_decode_stream_create(tokenizer, false);
            assert!(!stream.is_null());

            // The stream's Arc must keep the tokenizer alive after the public
            // tokenizer handle has been released.
            ffi_tokenizer_free(tokenizer);

            for token_id in [0, 1, 0, 1] {
                let chunk = ffi_tokenizer_decode_stream_step(stream, token_id);
                assert!(!chunk.is_null());
                assert!(unsafe { CStr::from_ptr(chunk) }.to_str().is_ok());
                ffi_tokenizer_free_string(chunk);
            }

            ffi_tokenizer_decode_stream_free(stream);
        }
    }

    #[test]
    fn decodes_qwen_byte_level_tokens_to_unicode() {
        // This is the relevant shape of Qwen3.5's tokenizer.json: BPE pieces
        // use GPT-2 byte-level spellings (for example Ġ = leading space and
        // Ã© = the UTF-8 bytes for é), and the decoder must reverse that map.
        let json = r#"{
            "version":"1.0",
            "truncation":null,
            "padding":null,
            "added_tokens":[],
            "normalizer":{"type":"NFC"},
            "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"vocab":{"ĠHello":0,"Ã©":1},"merges":[]}
        }"#;
        let tokenizer = Tokenizer::from_str(json).expect("Qwen ByteLevel tokenizer must load");
        let decoded = tokenizer.decode(&[0, 1], true).expect("Qwen ByteLevel decode");
        assert_eq!(decoded, " Helloé");
    }

    #[test]
    fn renders_qwen_chatml_with_generation_prompt() {
        let template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let rendered = render_chat_template(
            r#"[{"role":"system","content":"Be brief."},{"role":"user","content":"Hello"}]"#,
            &format!(r#"{{"eos_token":"<|im_end|>","chat_template":{}}}"#,
                     serde_json::to_string(template).unwrap()),
            "19 Jul 2026",
            true,
        )
        .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn normalizes_added_token_objects_and_exposes_current_date() {
        let rendered = render_chat_template(
            r#"[{"role":"user","content":"quoted \"value\""}]"#,
            r#"{"bos_token":{"content":"<s>","special":true},"chat_template":"{{ bos_token }}{{ date_string }}: {{ messages[0]['content'] }}"}"#,
            "19 Jul 2026",
            true,
        )
        .unwrap();
        assert_eq!(rendered, "<s>19 Jul 2026: quoted \"value\"");
    }

    #[test]
    fn matches_hugging_face_optional_chat_context_semantics() {
        let template = "{% if tools is defined and tools is none %}no-tools|{% endif %}{% if documents is defined and documents is none %}no-documents|{% endif %}{% for message in messages %}{% if message.reasoning_content %}{{ message.reasoning_content }}|{% endif %}{{ message.content }}{% if message.tool_calls %}|tool-calls{% endif %}{% if not loop.last %}|{% endif %}{% endfor %}";
        let rendered = render_chat_template(
            r#"[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"},{"role":"user","content":"Continue"}]"#,
            &format!(r#"{{"chat_template":{}}}"#, serde_json::to_string(template).unwrap()),
            "19 Jul 2026",
            true,
        )
        .unwrap();
        assert_eq!(rendered, "no-tools|no-documents|Question|Answer|Continue");
    }

    #[test]
    fn preserves_tools_and_structured_multi_turn_messages() {
        let template = "{{ tools | tojson }}|{{ messages[1].tool_calls | tojson }}|{% if messages[1].content is none %}null-content{% endif %}|{{ messages[2].tool_call_id }}|{% if add_generation_prompt %}generate{% endif %}";
        let context = r#"{
            "messages": [
                {"role":"user","content":"Weather?"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call-1","type":"function","function":{"name":"weather","arguments":{"city":"Tokyo"}}}
                ]},
                {"role":"tool","content":"sunny","tool_call_id":"call-1"}
            ],
            "tools":[{
                "type":"function",
                "function":{
                    "name":"weather",
                    "description":"Get weather",
                    "parameters":{
                        "type":"object",
                        "properties":{"city":{"type":"string"}},
                        "required":["city"]
                    }
                }
            }],
            "add_generation_prompt":true
        }"#;
        let rendered = render_chat_template_context(
            context,
            &format!(r#"{{"chat_template":{}}}"#, serde_json::to_string(template).unwrap()),
            "19 Jul 2026",
        )
        .unwrap();

        assert!(rendered.contains(r#""name":"weather""#));
        assert!(rendered.contains(r#""city":"Tokyo""#));
        assert!(rendered.ends_with("|null-content|call-1|generate"));
    }

    #[test]
    fn supports_hugging_face_python_compatible_string_methods() {
        let rendered = render_chat_template(
            r#"[{"role":"user","content":"ignored"}]"#,
            r#"{"chat_template":"{{ 'prefix-value-suffix'.startswith('prefix') }}|{{ 'prefix-value-suffix'.endswith('suffix') }}|{{ 'before</think>\\nafter'.split('</think>')[-1].lstrip('\\n') }}"}"#,
            "19 Jul 2026",
            true,
        )
        .unwrap();
        assert_eq!(rendered, "true|true|after");
    }

    #[test]
    fn propagates_template_rejections() {
        let error = render_chat_template(
            r#"[{"role":"tool","content":"result"}]"#,
            r#"{"chat_template":"{{ raise_exception('unsupported role') }}"}"#,
            "19 Jul 2026",
            true,
        )
        .unwrap_err();
        assert!(error.contains("unsupported role"));
    }
}
