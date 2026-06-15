#![allow(non_camel_case_types)]

use std::ffi::{CStr, CString};
use std::fmt::{Display, Formatter};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

pub const SDX_STATUS_OK: i32 = 0;

#[repr(C)]
pub struct sdx_runtime_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct sdx_model_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct sdx_context_t {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sdx_runtime_options_t {
    pub struct_size: u32,
}

impl Default for sdx_runtime_options_t {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sdx_model_options_t {
    pub struct_size: u32,
    pub backend: i32,
    pub strict_backend: i32,
    pub allow_runtime_jit: i32,
    pub gpu_target: i32,
}

impl Default for sdx_model_options_t {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            backend: 0,
            strict_backend: 0,
            allow_runtime_jit: 0,
            gpu_target: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sdx_run_options_t {
    pub struct_size: u32,
    pub backend: i32,
    pub strict_signature: i32,
    pub gpu_target: i32,
}

impl Default for sdx_run_options_t {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            backend: 0,
            strict_signature: 1,
            gpu_target: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sdx_tensor_view_t {
    pub data: *mut c_void,
    pub shape: *const i64,
    pub rank: i32,
    pub dtype: i32,
    pub bytes: usize,
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sdx_execution_report_t {
    pub struct_size: u32,
    pub requested_backend: i32,
    pub applied_backend: i32,
    pub status_code: i32,
    pub used_fallback: i32,
    pub execution_time_ns: u64,
    pub requested_gpu_target: i32,
    pub applied_gpu_target: i32,
}

impl Default for sdx_execution_report_t {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            requested_backend: 0,
            applied_backend: 0,
            status_code: 0,
            used_fallback: 0,
            execution_time_ns: 0,
            requested_gpu_target: 0,
            applied_gpu_target: 0,
        }
    }
}

#[cfg(all(feature = "cuda", feature = "amd"))]
compile_error!("features `cuda` and `amd` are mutually exclusive");

macro_rules! declare_sdx_ffi {
    () => {
        fn sdxGetRuntimeAbiVersion() -> c_int;

        fn sdxCreateRuntime(options: *const sdx_runtime_options_t, out_runtime: *mut *mut sdx_runtime_t) -> c_int;
        fn sdxDestroyRuntime(runtime: *mut sdx_runtime_t);

        fn sdxLoadBundle(
            runtime: *mut sdx_runtime_t,
            bundle_path: *const c_char,
            options: *const sdx_model_options_t,
            out_model: *mut *mut sdx_model_t,
        ) -> c_int;
        fn sdxUnloadModel(model: *mut sdx_model_t);

        fn sdxCreateContext(
            model: *mut sdx_model_t,
            requested_output_names: *const *const c_char,
            num_requested_outputs: i32,
            out_context: *mut *mut sdx_context_t,
        ) -> c_int;
        fn sdxDestroyContext(context: *mut sdx_context_t);

        fn sdxRun(
            context: *mut sdx_context_t,
            inputs: *const sdx_tensor_view_t,
            num_inputs: i32,
            outputs: *const sdx_tensor_view_t,
            num_outputs: i32,
            options: *const sdx_run_options_t,
        ) -> c_int;

        fn sdxGetLastError(runtime: *const sdx_runtime_t) -> *const c_char;
        fn sdxGetExecutionReport(context: *const sdx_context_t, out_report: *mut sdx_execution_report_t) -> c_int;
    };
}

#[cfg(feature = "cuda")]
#[link(name = "nd4jcuda")]
extern "C" {
    declare_sdx_ffi!();
}

#[cfg(feature = "amd")]
#[link(name = "nd4jamd")]
extern "C" {
    declare_sdx_ffi!();
}

#[cfg(not(any(feature = "cuda", feature = "amd")))]
#[link(name = "nd4jcpu")]
extern "C" {
    declare_sdx_ffi!();
}

#[derive(Debug, Clone)]
pub struct SdxError {
    pub status: i32,
    pub message: String,
}

impl Display for SdxError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "status={} message={}", self.status, self.message)
    }
}

impl std::error::Error for SdxError {}

fn last_error_message(runtime: *const sdx_runtime_t) -> String {
    unsafe {
        let ptr = sdxGetLastError(runtime);
        if ptr.is_null() {
            return String::new();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

fn status_to_result(status: i32, runtime: *const sdx_runtime_t, action: &str) -> Result<(), SdxError> {
    if status == SDX_STATUS_OK {
        return Ok(());
    }
    let detail = last_error_message(runtime);
    Err(SdxError {
        status,
        message: format!("{} failed: {}", action, detail),
    })
}

pub fn runtime_abi_version() -> i32 {
    unsafe { sdxGetRuntimeAbiVersion() }
}

pub struct Runtime {
    ptr: *mut sdx_runtime_t,
}

impl Runtime {
    pub fn create(options: Option<sdx_runtime_options_t>) -> Result<Self, SdxError> {
        let mut runtime: *mut sdx_runtime_t = ptr::null_mut();
        let options_ref = options.as_ref().map_or(ptr::null(), |o| o as *const _);
        let status = unsafe { sdxCreateRuntime(options_ref, &mut runtime) };
        if status != SDX_STATUS_OK {
            return Err(SdxError {
                status,
                message: format!("sdxCreateRuntime failed with status={}", status),
            });
        }
        Ok(Self { ptr: runtime })
    }

    pub fn load_model(&self, bundle_path: &str, options: Option<sdx_model_options_t>) -> Result<Model, SdxError> {
        let path = CString::new(bundle_path).map_err(|e| SdxError {
            status: -1,
            message: format!("invalid bundle path: {}", e),
        })?;

        let mut model: *mut sdx_model_t = ptr::null_mut();
        let options_ref = options.as_ref().map_or(ptr::null(), |o| o as *const _);
        let status = unsafe { sdxLoadBundle(self.ptr, path.as_ptr(), options_ref, &mut model) };
        status_to_result(status, self.ptr, "sdxLoadBundle")?;

        Ok(Model {
            runtime: self.ptr,
            ptr: model,
        })
    }

    pub fn last_error(&self) -> String {
        last_error_message(self.ptr)
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sdxDestroyRuntime(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

pub struct Model {
    runtime: *mut sdx_runtime_t,
    ptr: *mut sdx_model_t,
}

impl Model {
    pub fn create_context(&self, requested_outputs: Option<&[&str]>) -> Result<Context, SdxError> {
        let mut c_strings: Vec<CString> = Vec::new();
        let mut c_ptrs: Vec<*const c_char> = Vec::new();

        if let Some(names) = requested_outputs {
            for name in names {
                let c_name = CString::new(*name).map_err(|e| SdxError {
                    status: -1,
                    message: format!("invalid output name: {}", e),
                })?;
                c_ptrs.push(c_name.as_ptr());
                c_strings.push(c_name);
            }
        }

        let mut context: *mut sdx_context_t = ptr::null_mut();
        let status = unsafe {
            sdxCreateContext(
                self.ptr,
                if c_ptrs.is_empty() { ptr::null() } else { c_ptrs.as_ptr() },
                c_ptrs.len() as i32,
                &mut context,
            )
        };
        status_to_result(status, self.runtime, "sdxCreateContext")?;

        Ok(Context {
            runtime: self.runtime,
            ptr: context,
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sdxUnloadModel(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

pub struct Context {
    runtime: *mut sdx_runtime_t,
    ptr: *mut sdx_context_t,
}

impl Context {
    pub fn run(
        &self,
        inputs: &[sdx_tensor_view_t],
        outputs: &[sdx_tensor_view_t],
        options: Option<sdx_run_options_t>,
    ) -> Result<(), SdxError> {
        let options_ref = options.as_ref().map_or(ptr::null(), |o| o as *const _);

        let status = unsafe {
            sdxRun(
                self.ptr,
                if inputs.is_empty() { ptr::null() } else { inputs.as_ptr() },
                inputs.len() as i32,
                if outputs.is_empty() { ptr::null() } else { outputs.as_ptr() },
                outputs.len() as i32,
                options_ref,
            )
        };

        status_to_result(status, self.runtime, "sdxRun")
    }

    pub fn execution_report(&self) -> Result<sdx_execution_report_t, SdxError> {
        let mut report = sdx_execution_report_t::default();
        let status = unsafe { sdxGetExecutionReport(self.ptr, &mut report) };
        status_to_result(status, self.runtime, "sdxGetExecutionReport")?;
        Ok(report)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sdxDestroyContext(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}
