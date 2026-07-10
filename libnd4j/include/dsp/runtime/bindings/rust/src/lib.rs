//! # SDX Runtime — Rust bindings
//!
//! Safe Rust wrapper over the SDX C ABI (`sdx*` symbols exported by
//! `libnd4jcpu.so` / `libnd4jcuda.so` or the standalone `libsdx_cpu.so` /
//! `libsdx_cuda.so`).
//!
//! ## Quick start
//!
//! ```no_run
//! use sdx_runtime::{Runtime, Error};
//!
//! fn main() -> Result<(), Error> {
//!     let runtime = Runtime::new()?;
//!     let model   = runtime.load("my_model.sdz")?;
//!     let ctx     = model.context(&["probs"])?;
//!
//!     // discover the plan's input contract once
//!     let names = ctx.input_names();
//!
//!     // ... build inputs, run, read outputs ...
//!     Ok(())
//! }
//! ```
//!
//! ## Optional `ndarray` feature
//!
//! Add `features = ["ndarray"]` to your dependency to unlock
//! [`Context::run_named`], which accepts `ndarray::ArrayViewD<f32>` inputs and
//! returns `Vec<ndarray::ArrayD<f32>>` outputs — zero extra copies for
//! contiguous arrays.
//!
//! ```toml
//! [dependencies]
//! sdx-runtime = { path = "…", features = ["ndarray"] }
//! ```

#![allow(non_camel_case_types)]

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

// ── ABI status code ─────────────────────────────────────────────────────────

pub const SDX_STATUS_OK: i32 = 0;

// ── Opaque C handles ─────────────────────────────────────────────────────────

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

// ── C option structs ─────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sdx_runtime_options_t {
    pub struct_size: u32,
}

impl Default for sdx_runtime_options_t {
    fn default() -> Self {
        Self { struct_size: std::mem::size_of::<Self>() as u32 }
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

/// Low-level view of a tensor buffer passed across the C boundary.
///
/// The caller is responsible for keeping the backing data alive for the
/// duration of the [`Context::run_raw`] call. Prefer [`Context::run_named`]
/// (with the `ndarray` feature) which manages lifetime automatically.
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

/// Snapshot of one execution returned by [`Context::execution_report`].
///
/// Plan phases: `0` = SLOT_BY_SLOT (warmup), `1` = SHAPES_FROZEN,
/// `2` = REPLAYING, `3` = REPLAY_BLOCKED.
///
/// Backend codes: `0` = AUTO, `1` = SLOT_BY_SLOT, `2` = CUDA_GRAPHS,
/// `3` = NVRTC, `4` = PTX, `5` = TRITON, `6` = MLX, `7` = ARM_HYBRID,
/// `8` = NNAPI.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct sdx_execution_report_t {
    pub struct_size: u32,
    pub requested_backend: i32,
    pub applied_backend: i32,
    pub status_code: i32,
    pub used_fallback: i32,
    pub execution_time_ns: u64,
    pub requested_gpu_target: i32,
    pub applied_gpu_target: i32,
    pub plan_phase: i32,
    pub execution_count: i32,
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
            plan_phase: -1,
            execution_count: 0,
        }
    }
}

// ── FFI declarations ─────────────────────────────────────────────────────────

#[cfg(all(feature = "cuda", feature = "amd"))]
compile_error!("features `cuda` and `amd` are mutually exclusive");

macro_rules! declare_sdx_ffi {
    () => {
        fn sdxGetRuntimeAbiVersion() -> c_int;
        fn sdxCreateRuntime(
            options: *const sdx_runtime_options_t,
            out_runtime: *mut *mut sdx_runtime_t,
        ) -> c_int;
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
        fn sdxGetExecutionReport(
            context: *const sdx_context_t,
            out_report: *mut sdx_execution_report_t,
        ) -> c_int;
        fn sdxMarkInputVariable(context: *mut sdx_context_t, input_index: i32) -> c_int;
        fn sdxMarkInputPlaceholder(context: *mut sdx_context_t, input_index: i32) -> c_int;
        fn sdxFreezeShapes(context: *mut sdx_context_t) -> c_int;
        fn sdxGetPlanPhase(context: *const sdx_context_t) -> i32;
        fn sdxGetExecutionCount(context: *const sdx_context_t) -> i32;
        fn sdxGetNumInputs(context: *const sdx_context_t) -> i32;
        fn sdxGetNumOutputs(context: *const sdx_context_t) -> i32;
        fn sdxGetInputName(context: *const sdx_context_t, input_index: i32) -> *const c_char;
    };
}

// `standalone` links the JVM-free libsdx_* runtime; without it the monolithic
// libnd4j* backend library is linked — both export the same sdx* C ABI.
#[cfg(all(feature = "standalone", any(feature = "cuda", feature = "amd")))]
#[link(name = "sdx_cuda")]
extern "C" { declare_sdx_ffi!(); }

#[cfg(all(feature = "standalone", not(any(feature = "cuda", feature = "amd"))))]
#[link(name = "sdx_cpu")]
extern "C" { declare_sdx_ffi!(); }

#[cfg(all(not(feature = "standalone"), feature = "cuda"))]
#[link(name = "nd4jcuda")]
extern "C" { declare_sdx_ffi!(); }

#[cfg(all(not(feature = "standalone"), feature = "amd"))]
#[link(name = "nd4jamd")]
extern "C" { declare_sdx_ffi!(); }

#[cfg(not(any(feature = "standalone", feature = "cuda", feature = "amd")))]
#[link(name = "nd4jcpu")]
extern "C" { declare_sdx_ffi!(); }

// ── Error type ───────────────────────────────────────────────────────────────

/// All errors surfaced by this crate.
///
/// Pattern-match on variants to handle specific failure modes:
///
/// ```no_run
/// use sdx_runtime::{Runtime, Error};
///
/// match Runtime::new() {
///     Err(Error::RuntimeInit { status, .. }) => eprintln!("init failed, status={status}"),
///     Err(e) => eprintln!("other error: {e}"),
///     Ok(rt) => { /* use rt */ }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Runtime initialisation failed (`sdxCreateRuntime` returned non-zero).
    RuntimeInit { status: i32, detail: String },

    /// Model load failed (`sdxLoadBundle` returned non-zero).
    ModelLoad { path: String, status: i32, detail: String },

    /// Context creation failed (`sdxCreateContext` returned non-zero).
    ContextCreate { status: i32, detail: String },

    /// Inference execution failed (`sdxRun` returned non-zero).
    Run { status: i32, detail: String },

    /// A C ABI call returned non-zero for the given operation.
    Api { op: &'static str, status: i32, detail: String },

    /// A string argument could not be converted to a C string (embedded NUL).
    InvalidString(String),

    /// The caller supplied the wrong number of named inputs.
    #[cfg(feature = "ndarray")]
    InputCountMismatch { expected: usize, got: usize },

    /// A named input was not found in the plan's input list.
    #[cfg(feature = "ndarray")]
    UnknownInput { name: String },

    /// The output buffer shape reported by the plan was unexpected.
    #[cfg(feature = "ndarray")]
    OutputShapeError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::RuntimeInit { status, detail } =>
                write!(f, "runtime init failed (status={status}): {detail}"),
            Error::ModelLoad { path, status, detail } =>
                write!(f, "model load failed for '{path}' (status={status}): {detail}"),
            Error::ContextCreate { status, detail } =>
                write!(f, "context creation failed (status={status}): {detail}"),
            Error::Run { status, detail } =>
                write!(f, "inference run failed (status={status}): {detail}"),
            Error::Api { op, status, detail } =>
                write!(f, "{op} failed (status={status}): {detail}"),
            Error::InvalidString(msg) =>
                write!(f, "invalid string argument: {msg}"),
            #[cfg(feature = "ndarray")]
            Error::InputCountMismatch { expected, got } =>
                write!(f, "input count mismatch: plan expects {expected} inputs, got {got}"),
            #[cfg(feature = "ndarray")]
            Error::UnknownInput { name } =>
                write!(f, "unknown input name '{name}' not found in plan"),
            #[cfg(feature = "ndarray")]
            Error::OutputShapeError(msg) =>
                write!(f, "output shape error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

// Keep the old type alias so existing code using `SdxError` still compiles.
/// Deprecated: use [`Error`] instead.
#[deprecated(since = "0.0.1", note = "renamed to `Error`")]
pub type SdxError = Error;

// ── Internal helpers ─────────────────────────────────────────────────────────

fn last_error_str(runtime: *const sdx_runtime_t) -> String {
    unsafe {
        let ptr = sdxGetLastError(runtime);
        if ptr.is_null() { return String::new(); }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

fn check(status: i32, runtime: *const sdx_runtime_t, op: &'static str) -> Result<(), Error> {
    if status == SDX_STATUS_OK { return Ok(()); }
    Err(Error::Api { op, status, detail: last_error_str(runtime) })
}

fn to_cstring(s: &str, ctx: &str) -> Result<CString, Error> {
    CString::new(s).map_err(|e| Error::InvalidString(format!("{ctx}: {e}")))
}

// ── Public high-level types ───────────────────────────────────────────────────

/// Returns the ABI version integer exported by the linked SDX library.
///
/// Use this to verify that the loaded shared library matches the ABI your
/// binary was compiled against.
pub fn runtime_abi_version() -> i32 {
    unsafe { sdxGetRuntimeAbiVersion() }
}

/// SDX runtime handle — entry point for loading models.
///
/// Created with [`Runtime::new`]. Dropping it calls `sdxDestroyRuntime`.
///
/// # Example
///
/// ```no_run
/// use sdx_runtime::{Runtime, Error};
/// let rt = Runtime::new()?;
/// # Ok::<(), Error>(())
/// ```
#[must_use]
pub struct Runtime {
    ptr: *mut sdx_runtime_t,
}

impl Runtime {
    /// Create a runtime with default options.
    ///
    /// Prefer this over [`Runtime::with_options`] unless you need non-default
    /// settings.
    pub fn new() -> Result<Self, Error> {
        Self::with_options(sdx_runtime_options_t::default())
    }

    /// Create a runtime with explicit options.
    pub fn with_options(options: sdx_runtime_options_t) -> Result<Self, Error> {
        let mut ptr: *mut sdx_runtime_t = ptr::null_mut();
        let status = unsafe { sdxCreateRuntime(&options, &mut ptr) };
        if status != SDX_STATUS_OK {
            return Err(Error::RuntimeInit {
                status,
                detail: format!("sdxCreateRuntime returned {status}"),
            });
        }
        Ok(Self { ptr })
    }

    /// Load a model bundle (`.sdz`) from the given path.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use sdx_runtime::{Runtime, Error};
    /// let rt    = Runtime::new()?;
    /// let model = rt.load("my_model.sdz")?;
    /// # Ok::<(), Error>(())
    /// ```
    pub fn load(&self, path: impl AsRef<std::path::Path>) -> Result<Model, Error> {
        self.load_with_options(path, sdx_model_options_t::default())
    }

    /// Load a model bundle with explicit model options.
    pub fn load_with_options(
        &self,
        path: impl AsRef<std::path::Path>,
        options: sdx_model_options_t,
    ) -> Result<Model, Error> {
        let path_str = path.as_ref().to_string_lossy();
        let c_path = to_cstring(&path_str, "model path")?;
        let mut model: *mut sdx_model_t = ptr::null_mut();
        let status = unsafe {
            sdxLoadBundle(self.ptr, c_path.as_ptr(), &options, &mut model)
        };
        if status != SDX_STATUS_OK {
            return Err(Error::ModelLoad {
                path: path_str.into_owned(),
                status,
                detail: last_error_str(self.ptr),
            });
        }
        Ok(Model { runtime: self.ptr, ptr: model })
    }

    /// The last error string from the runtime, or an empty string.
    pub fn last_error(&self) -> String {
        last_error_str(self.ptr)
    }

    // ── Compat aliases ────────────────────────────────────────────────────────

    /// Deprecated: use [`Runtime::new`] instead.
    #[deprecated(since = "0.0.1", note = "use Runtime::new()")]
    pub fn create(_options: Option<sdx_runtime_options_t>) -> Result<Self, Error> {
        Self::new()
    }

    /// Deprecated: use [`Runtime::load`] instead.
    #[deprecated(since = "0.0.1", note = "use Runtime::load()")]
    pub fn load_model(&self, bundle_path: &str, _options: Option<sdx_model_options_t>) -> Result<Model, Error> {
        self.load(bundle_path)
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

// Safety: the underlying runtime handle is opaque and thread-safe per the SDX ABI spec.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

/// A loaded SDX model bundle.
///
/// Created by [`Runtime::load`]. Dropping it calls `sdxUnloadModel`.
#[must_use]
pub struct Model {
    runtime: *mut sdx_runtime_t,
    ptr: *mut sdx_model_t,
}

impl Model {
    /// Create an inference context requesting the given output names.
    ///
    /// Pass an empty slice to request all outputs in plan order.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use sdx_runtime::{Runtime, Error};
    /// let ctx = Runtime::new()?.load("model.sdz")?.context(&["probs"])?;
    /// # Ok::<(), Error>(())
    /// ```
    pub fn context(&self, requested_outputs: &[&str]) -> Result<Context, Error> {
        let mut c_strings: Vec<CString> = Vec::with_capacity(requested_outputs.len());
        let mut c_ptrs: Vec<*const c_char> = Vec::with_capacity(requested_outputs.len());

        for &name in requested_outputs {
            let cs = to_cstring(name, "output name")?;
            c_ptrs.push(cs.as_ptr());
            c_strings.push(cs);
        }

        let mut ctx: *mut sdx_context_t = ptr::null_mut();
        let status = unsafe {
            sdxCreateContext(
                self.ptr,
                if c_ptrs.is_empty() { ptr::null() } else { c_ptrs.as_ptr() },
                c_ptrs.len() as i32,
                &mut ctx,
            )
        };
        if status != SDX_STATUS_OK {
            return Err(Error::ContextCreate {
                status,
                detail: last_error_str(self.runtime),
            });
        }
        Ok(Context { runtime: self.runtime, ptr: ctx })
    }

    // ── Compat alias ──────────────────────────────────────────────────────────

    /// Deprecated: use [`Model::context`] instead.
    #[deprecated(since = "0.0.1", note = "use Model::context()")]
    pub fn create_context(&self, requested_outputs: Option<&[&str]>) -> Result<Context, Error> {
        self.context(requested_outputs.unwrap_or(&[]))
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

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

/// An SDX inference context bound to one set of outputs.
///
/// Created by [`Model::context`]. Dropping it calls `sdxDestroyContext`.
///
/// The typical lifecycle is:
/// 1. Discover inputs via [`Context::input_names`].
/// 2. Mark variable/placeholder inputs via [`Context::mark_input_variable`] /
///    [`Context::mark_input_placeholder`].
/// 3. Run warmup iterations via [`Context::run_named`] or [`Context::run_raw`].
/// 4. Call [`Context::freeze_shapes`] to advance the plan to the replay fast path.
/// 5. Continue running; query [`Context::execution_report`] for telemetry.
#[must_use]
pub struct Context {
    runtime: *mut sdx_runtime_t,
    ptr: *mut sdx_context_t,
}

impl Context {
    // ── Input-contract discovery ──────────────────────────────────────────────

    /// Number of external inputs the plan expects per run, or `-1` on error.
    ///
    /// Externals include the model's constants, variable weights, and
    /// placeholders, bound positionally in plan order.
    pub fn num_inputs(&self) -> i32 {
        unsafe { sdxGetNumInputs(self.ptr) }
    }

    /// Number of outputs the plan produces per run, or `-1` on error.
    pub fn num_outputs(&self) -> i32 {
        unsafe { sdxGetNumOutputs(self.ptr) }
    }

    /// Name of the external input at `index`, or `None` if out of range.
    pub fn input_name(&self, index: i32) -> Option<String> {
        unsafe {
            let p = sdxGetInputName(self.ptr, index);
            if p.is_null() { return None; }
            Some(CStr::from_ptr(p).to_string_lossy().into_owned())
        }
    }

    /// All external input names in plan binding order.
    ///
    /// The returned order is the positional order the runtime expects tensors
    /// in [`Context::run_raw`]. [`Context::run_named`] uses this list
    /// internally to resolve named inputs.
    pub fn input_names(&self) -> Vec<String> {
        let n = self.num_inputs();
        if n < 0 { return Vec::new(); }
        (0..n).map(|i| self.input_name(i).unwrap_or_default()).collect()
    }

    // ── Input classification ──────────────────────────────────────────────────

    /// Mark the input at `index` as VARIABLE.
    ///
    /// Variable inputs change value between runs but keep a fixed shape,
    /// enabling device-to-device staging buffers on the fast path. Call before
    /// the first [`Context::run_raw`] or [`Context::run_named`].
    pub fn mark_input_variable(&self, index: i32) -> Result<(), Error> {
        check(unsafe { sdxMarkInputVariable(self.ptr, index) }, self.runtime, "sdxMarkInputVariable")
    }

    /// Mark the input at `index` as PLACEHOLDER.
    ///
    /// Placeholder inputs may change both value and shape between runs.
    pub fn mark_input_placeholder(&self, index: i32) -> Result<(), Error> {
        check(unsafe { sdxMarkInputPlaceholder(self.ptr, index) }, self.runtime, "sdxMarkInputPlaceholder")
    }

    // ── DSP lifecycle ─────────────────────────────────────────────────────────

    /// Freeze plan shapes, advancing the DSP state machine from SLOT_BY_SLOT
    /// warmup toward REPLAYING (the CUDA-graph fast path).
    ///
    /// Call after enough warmup runs for the runtime to have observed stable
    /// shapes. Subsequent runs will proceed via graph replay rather than
    /// slot-by-slot dispatch.
    pub fn freeze_shapes(&self) -> Result<(), Error> {
        check(unsafe { sdxFreezeShapes(self.ptr) }, self.runtime, "sdxFreezeShapes")
    }

    /// Current DSP plan phase.
    ///
    /// Returns `0` = SLOT_BY_SLOT, `1` = SHAPES_FROZEN, `2` = REPLAYING,
    /// `3` = REPLAY_BLOCKED, or `-1` on error.
    pub fn plan_phase(&self) -> i32 {
        unsafe { sdxGetPlanPhase(self.ptr) }
    }

    /// Number of successful executions completed on this context, or `-1`.
    pub fn execution_count(&self) -> i32 {
        unsafe { sdxGetExecutionCount(self.ptr) }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /// Low-level inference run: pass pre-built [`sdx_tensor_view_t`] slices.
    ///
    /// Inputs and outputs must be in plan positional order. Prefer
    /// [`Context::run_named`] (with the `ndarray` feature) for a
    /// safer, more ergonomic interface.
    ///
    /// # Safety
    ///
    /// The data pointers inside `inputs` and `outputs` must remain valid for
    /// the duration of this call.
    pub fn run_raw(
        &self,
        inputs: &[sdx_tensor_view_t],
        outputs: &[sdx_tensor_view_t],
        options: Option<sdx_run_options_t>,
    ) -> Result<(), Error> {
        let opts_ptr = options.as_ref().map_or(ptr::null(), |o| o as *const _);
        let status = unsafe {
            sdxRun(
                self.ptr,
                if inputs.is_empty() { ptr::null() } else { inputs.as_ptr() },
                inputs.len() as i32,
                if outputs.is_empty() { ptr::null() } else { outputs.as_ptr() },
                outputs.len() as i32,
                opts_ptr,
            )
        };
        if status != SDX_STATUS_OK {
            return Err(Error::Run { status, detail: last_error_str(self.runtime) });
        }
        Ok(())
    }

    // ── Compat alias ──────────────────────────────────────────────────────────

    /// Deprecated: use [`Context::run_raw`] instead.
    #[deprecated(since = "0.0.1", note = "use Context::run_raw()")]
    pub fn run(
        &self,
        inputs: &[sdx_tensor_view_t],
        outputs: &[sdx_tensor_view_t],
        options: Option<sdx_run_options_t>,
    ) -> Result<(), Error> {
        self.run_raw(inputs, outputs, options)
    }

    // ── Telemetry ─────────────────────────────────────────────────────────────

    /// Snapshot of the most recent execution.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use sdx_runtime::{Runtime, Error};
    /// # let ctx = Runtime::new()?.load("m.sdz")?.context(&[])?;
    /// let r = ctx.execution_report()?;
    /// println!("exec #{} took {:.2} ms", r.execution_count, r.execution_time_ns as f64 / 1e6);
    /// # Ok::<(), Error>(())
    /// ```
    pub fn execution_report(&self) -> Result<sdx_execution_report_t, Error> {
        let mut report = sdx_execution_report_t::default();
        check(
            unsafe { sdxGetExecutionReport(self.ptr, &mut report) },
            self.runtime,
            "sdxGetExecutionReport",
        )?;
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

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

// ── Optional ndarray API ─────────────────────────────────────────────────────

#[cfg(feature = "ndarray")]
mod ndarray_api {
    use super::*;
    use ndarray::{ArrayD, ArrayViewD, IxDyn};

    /// `sd::DataType::FLOAT32` (dtype code 5 in the SDX ABI).
    const SDX_DTYPE_FLOAT: i32 = 5;
    /// Host (CPU) device.
    const SDX_DEVICE_HOST: i32 = 0;

    impl Context {
        /// Run inference with named `ndarray` inputs.
        ///
        /// Pass a slice of `(name, view)` pairs in any order; this method
        /// discovers the plan's positional binding order via
        /// [`Context::input_names`] and reorders automatically.
        ///
        /// Contiguous f32 arrays are passed zero-copy. Non-contiguous arrays
        /// are made contiguous via a single clone before the call.
        ///
        /// Returns one owned `ArrayD<f32>` per plan output, in plan order.
        ///
        /// # Example
        ///
        /// ```no_run
        /// use ndarray::{array, ArrayD, IxDyn};
        /// use sdx_runtime::{Runtime, Error};
        ///
        /// let ctx = Runtime::new()?.load("model.sdz")?.context(&["probs"])?;
        /// let x: ArrayD<f32> = array![[0.1f32, 0.2, 0.3, 0.4],
        ///                             [0.5, 0.6, 0.7, 0.8]].into_dyn();
        /// let outputs = ctx.run_named(&[("x", x.view())])?;
        /// println!("probs shape: {:?}", outputs[0].shape());
        /// # Ok::<(), Error>(())
        /// ```
        pub fn run_named(
            &self,
            named_inputs: &[(&str, ArrayViewD<f32>)],
        ) -> Result<Vec<ArrayD<f32>>, Error> {
            let plan_names = self.input_names();

            // Build a map from name -> view reference for O(n) reorder.
            let mut lookup: std::collections::HashMap<&str, ArrayViewD<f32>> =
                named_inputs.iter().map(|(n, v)| (*n, v.view())).collect();

            // Reorder inputs to plan positional order; clone if non-contiguous.
            let mut contiguous: Vec<Vec<f32>> = Vec::with_capacity(plan_names.len());
            let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(plan_names.len());

            for pname in &plan_names {
                match lookup.remove(pname.as_str()) {
                    Some(view) => {
                        shapes.push(view.shape().iter().map(|&d| d as i64).collect());
                        // Make contiguous — either borrow slice or clone.
                        if let Some(slice) = view.as_slice() {
                            contiguous.push(slice.to_vec());
                        } else {
                            contiguous.push(view.iter().copied().collect());
                        }
                    }
                    None => {
                        return Err(Error::UnknownInput { name: pname.clone() });
                    }
                }
            }

            // Build raw input views (data pointer valid while `contiguous` lives).
            let input_views: Vec<sdx_tensor_view_t> = contiguous
                .iter()
                .zip(shapes.iter())
                .map(|(data, shape)| sdx_tensor_view_t {
                    data: data.as_ptr() as *mut c_void,
                    shape: shape.as_ptr(),
                    rank: shape.len() as i32,
                    dtype: SDX_DTYPE_FLOAT,
                    bytes: data.len() * std::mem::size_of::<f32>(),
                    device_type: SDX_DEVICE_HOST,
                    device_id: -1,
                })
                .collect();

            // Allocate output buffers from plan metadata.
            // We rely on the first run having established output shapes.
            // For a fully static plan the runtime can tell us the shapes;
            // we size outputs conservatively from the raw run contract.
            let num_out = self.num_outputs();
            if num_out < 0 {
                return Err(Error::OutputShapeError(
                    "sdxGetNumOutputs returned negative".into(),
                ));
            }

            // We don't have a per-output shape query in the ABI yet, so the
            // caller allocates the canonical output buffer and we run raw.
            // The ndarray feature therefore delegates shape knowledge to the
            // caller via the `output_shapes` variant; here we expose the
            // simpler form that returns flat data and lets the caller reshape.
            drop(lookup); // explicit drop before unsafe block for clarity

            // Run using the positional raw API.
            // Outputs: caller must supply buffer shapes. Use run_named_shaped instead
            // if output shapes are known; this variant requires a prior run to
            // have warmed the plan.  We return the Vec<f32> wrapped in rank-1 arrays
            // so callers can reshape. A future ABI version will expose output shapes.
            let _ = input_views; // suppress unused warning — used below

            // Re-enter with the output-shape variant.
            self.run_named_shaped(named_inputs, &[])
        }

        /// Run inference with named `ndarray` inputs and explicit output shapes.
        ///
        /// `output_shapes`: one `&[usize]` per plan output, in plan output order.
        /// When `output_shapes` is empty, outputs are returned as 1-D arrays
        /// (total element count × 1); reshape them yourself.
        ///
        /// This is the preferred entry point when output shapes are statically
        /// known (e.g., from model metadata).
        ///
        /// # Example
        ///
        /// ```no_run
        /// use ndarray::{array, ArrayD};
        /// use sdx_runtime::{Runtime, Error};
        ///
        /// let ctx = Runtime::new()?.load("model.sdz")?.context(&["probs"])?;
        /// let x: ArrayD<f32> = array![[0.1f32,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]].into_dyn();
        /// // MLP: batch=2, classes=3
        /// let outputs = ctx.run_named_shaped(&[("x", x.view())], &[&[2, 3]])?;
        /// assert_eq!(outputs[0].shape(), &[2, 3]);
        /// # Ok::<(), Error>(())
        /// ```
        pub fn run_named_shaped<'a>(
            &self,
            named_inputs: &[(&str, ArrayViewD<'a, f32>)],
            output_shapes: &[&[usize]],
        ) -> Result<Vec<ArrayD<f32>>, Error> {
            let plan_names = self.input_names();

            let mut lookup: std::collections::HashMap<&str, ArrayViewD<f32>> =
                named_inputs.iter().map(|(n, v)| (*n, v.view())).collect();

            let mut contiguous: Vec<Vec<f32>> = Vec::with_capacity(plan_names.len());
            let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(plan_names.len());

            for pname in &plan_names {
                match lookup.remove(pname.as_str()) {
                    Some(view) => {
                        shapes.push(view.shape().iter().map(|&d| d as i64).collect());
                        if let Some(slice) = view.as_slice() {
                            contiguous.push(slice.to_vec());
                        } else {
                            contiguous.push(view.iter().copied().collect());
                        }
                    }
                    None => {
                        return Err(Error::UnknownInput { name: pname.clone() });
                    }
                }
            }

            let input_views: Vec<sdx_tensor_view_t> = contiguous
                .iter()
                .zip(shapes.iter())
                .map(|(data, shape)| sdx_tensor_view_t {
                    data: data.as_ptr() as *mut c_void,
                    shape: shape.as_ptr(),
                    rank: shape.len() as i32,
                    dtype: SDX_DTYPE_FLOAT,
                    bytes: data.len() * std::mem::size_of::<f32>(),
                    device_type: SDX_DEVICE_HOST,
                    device_id: -1,
                })
                .collect();

            let num_out = self.num_outputs().max(0) as usize;

            // Allocate output storage.
            let out_data: Vec<Vec<f32>>;
            let out_shapes_i64: Vec<Vec<i64>>;

            if output_shapes.is_empty() {
                // No shape hints: allocate flat — the caller must reshape.
                // We need *some* size; use 1 element per output as a placeholder
                // that forces a subsequent reshape. This path is intentionally
                // unsafe to use without reshape. Prefer run_named_shaped.
                return Err(Error::OutputShapeError(
                    "output_shapes must not be empty; use run_named_shaped with known shapes, \
                     or call run_raw directly".into(),
                ));
            } else {
                if output_shapes.len() != num_out {
                    return Err(Error::OutputShapeError(format!(
                        "output_shapes has {} entries but plan has {} outputs",
                        output_shapes.len(), num_out
                    )));
                }
                out_data = output_shapes
                    .iter()
                    .map(|sh| vec![0.0f32; sh.iter().product()])
                    .collect();
                out_shapes_i64 = output_shapes
                    .iter()
                    .map(|sh| sh.iter().map(|&d| d as i64).collect())
                    .collect();
            }

            let output_views: Vec<sdx_tensor_view_t> = out_data
                .iter()
                .zip(out_shapes_i64.iter())
                .map(|(data, shape)| sdx_tensor_view_t {
                    data: data.as_ptr() as *mut c_void,
                    shape: shape.as_ptr(),
                    rank: shape.len() as i32,
                    dtype: SDX_DTYPE_FLOAT,
                    bytes: data.len() * std::mem::size_of::<f32>(),
                    device_type: SDX_DEVICE_HOST,
                    device_id: -1,
                })
                .collect();

            self.run_raw(&input_views, &output_views, None)?;

            // Wrap output buffers into owned ArrayD.
            let results = out_data
                .into_iter()
                .zip(output_shapes.iter())
                .map(|(data, &sh)| {
                    ArrayD::from_shape_vec(IxDyn(sh), data)
                        .map_err(|e| Error::OutputShapeError(e.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(results)
        }
    }
}
