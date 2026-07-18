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

use std::ffi::{c_void, CStr, CString};
use std::mem::ManuallyDrop;
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use tokenizers::tokenizer::{
    DecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper,
    PreTokenizerWrapper,
};
use tokenizers::Tokenizer;

type TokenizerDecodeStream = DecodeStream<
    'static,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

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

/// Opaque tokenizer handle
pub struct TokenizerHandle {
    tokenizer: Arc<Tokenizer>,
}

/// Stateful decoder backed by Hugging Face tokenizers' DecodeStream.
///
/// The stream contains a reference into the Arc allocation retained in
/// `_tokenizer`. The Arc keeps that allocation stable and alive until after
/// the stream is dropped, including when the originating TokenizerHandle has
/// already been freed.
struct DecodeStreamInner {
    stream: ManuallyDrop<TokenizerDecodeStream>,
    _tokenizer: Arc<Tokenizer>,
}

impl Drop for DecodeStreamInner {
    fn drop(&mut self) {
        // Drop the borrowing stream before releasing the Arc that owns the
        // tokenizer it references.
        unsafe {
            ManuallyDrop::drop(&mut self.stream);
        }
    }
}

/// Opaque stateful decoder handle exported through the C ABI.
pub struct DecodeStreamHandle {
    inner: *mut c_void,
}

impl Drop for DecodeStreamHandle {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                drop(Box::from_raw(self.inner as *mut DecodeStreamInner));
            }
            self.inner = ptr::null_mut();
        }
    }
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

    let tokenizer = unsafe { Arc::clone(&(*handle).tokenizer) };
    let tokenizer_ref: &'static Tokenizer = unsafe { &*Arc::as_ptr(&tokenizer) };
    let stream: TokenizerDecodeStream = tokenizer_ref.decode_stream(skip_special_tokens);

    let inner = Box::new(DecodeStreamInner {
        stream: ManuallyDrop::new(stream),
        _tokenizer: tokenizer,
    });
    Box::into_raw(Box::new(DecodeStreamHandle {
        inner: Box::into_raw(inner) as *mut c_void,
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

    let stream_handle = unsafe { &mut *handle };
    if stream_handle.inner.is_null() {
        set_error("Decode stream state is null".to_string());
        return ptr::null_mut();
    }
    let stream = unsafe { &mut *(stream_handle.inner as *mut DecodeStreamInner) };
    match stream.stream.step(token_id) {
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
