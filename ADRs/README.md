# Architecture Decision Records

This directory contains all Architecture Decision Records (ADRs) for the
Deeplearning4j project. Each ADR documents a significant technical decision,
its rationale, and its current implementation status.

**Last updated:** 2026-05-23

## Status Key

| Status | Meaning |
|---|---|
| **Implemented** | Decision made and fully integrated into the codebase |
| **Accepted** | Decision made, implementation in progress or complete |
| **Proposed** | Under consideration, not yet decided |
| **Discussion** | Early-stage exploration, may not proceed |
| **Rejected** | Evaluated and explicitly not adopted |
| **Superseded** | Replaced by a later ADR (noted inline) |

## Numbering Notes

- Numbers 0017, 0029, 0040, 0043, 0044 were never assigned
- Pre-existing duplicate numbers (0003, 0024) are kept as-is for history
- Former duplicates (0057×4, 0073×2) have been renumbered to 0085-0088
- Former duplicate 0075 (same as 0056) has been removed
- Root-level stray ADRs have been moved in and numbered 0089-0092

---

## ADR Index by Theme

### File Format & Serialization

| # | Title | Status | Description |
|---|---|---|---|
| [0001](0001-SameDiff_File_Format.md) | SameDiff File Format | Accepted | Zip+FlatBuffers format for SameDiff models: graph structure in FlatBuffers, parameters stored separately, supporting multiple model versions/checkpoints in one file. |
| [0034](0034%20-%20FlatBuffers%20upgrade.md) | FlatBuffers Modernization | Implemented | Upgraded FlatBuffers schemas from legacy `Sequence` patterns to modern `[Type]` vector syntax, fixed CMake integration, standardized namespaces to `graph`. |
| [0035](0035-Samediff-Extended-Storage-Format.md) | SameDiff Unified Container Format (SDNB/SDZ) | Implemented | Section-based binary format (SDNB) and ZIP-wrapped variant (SDZ) with first-class sharding, metadata headers, and backward compatibility for single-file deployment of large models. |

### Model Import

| # | Title | Status | Description |
|---|---|---|---|
| [0002](0002-ONNX_Runtime.md) | ONNX Runtime Module | Implemented | JavaCPP-based ONNX Runtime bindings for interop with ONNX models, exposing a GraphRunner API with INDArrays as interchange format. |
| [0003](0003-Import_IR.md) | Import IR | Implemented | Intermediate representation bridging attribute-based framework formats (TF, ONNX, Keras) to nd4j's list-based op execution format. |
| [0003](0003-NdArray_Strides_ArmCompute.md) | NDArray Padded Strides for ARM Compute | Implemented | Helper functions for non-standard padded strides in NDArray to enable ARM Compute Library integration. |
| [0004](0004-Mapping_IR.md) | Mapping IR | Implemented | MappingRules file format describing op-level transformations between framework protobuf formats and nd4j's OpDescriptor format. |
| [0005](0005-Interpreter.md) | Interpreter | Rejected | Proposed interpreter using Import IR — rejected in favor of direct conversion. |
| [0009](0009%20-%20Import%20node%20pre%20processing.md) | Import Node Pre-Processing | Discussion | Annotation-driven node pre-processor hooks for version-migration rules during graph import. |

### GGML/GGUF Import

| # | Title | Status | Description |
|---|---|---|---|
| [0052](0052%20-%20GGML-GGUF%20Model%20Import.md) | GGML/GGUF Model Import | Implemented | The `nd4j-ggml` module for importing GGML and GGUF model files into SameDiff with memory-mapped I/O and configurable dequantization. Supports 15+ model families. |
| [0053](0053%20-%20GGML%20Quantization%20Handling.md) | GGML Quantization Handling | Implemented | Flexible `Dequantizer` interface supporting FP32, FP16, and preserve-quantization modes with implementations for Q4_0 through IQ4_XS block-quantization schemes. |
| [0054](0054%20-%20GGML%20Architecture%20Detection.md) | GGML Architecture Detection | Implemented | Strategy-pattern `ModelArchitecture` interface with `ArchitectureRegistry` to auto-detect model architectures (LLaMA, Mistral, Gemma, Phi, Whisper, GLM, Granite, etc.) from GGUF metadata. |

### OmniHub / Model Zoo

| # | Title | Status | Description |
|---|---|---|---|
| [0011](0011%20-%20OmniHub-Zoo%20Download.md) | OmniHub Zoo Download | Discussion | Python-backed tooling for downloading models from HuggingFace, ONNX Hub, TF Hub, PyTorch Hub. |
| [0012](0012%20-%20OmniHub-Zoo%20Download%20Implementations.md) | OmniHub Download Implementations | Discussion | Per-ecosystem download implementations for five model sources. |
| [0013](0013%20-%20OmniHub-Zoo%20Consumption.md) | OmniHub Consumption API | Accepted | `Pretrained` namespace API (e.g. `Pretrained.samediff().resnet18().create()`) for instantiating pretrained models. |
| [0014](0014%20-%20OmniHub-%20Replace%20old%20model%20zoo.md) | Replace Old Model Zoo | Accepted | Migration from legacy `deeplearning4j-zoo` to OmniHub with GitHub-hosted model registry. |
| [0015](0015%20-%20Unified%20Resource%20Manager.md) | Unified Resource Manager | Discussion | Consolidation of all download/resource management (Strumpf, legacy zoo, OmniHub, datasets) into a single manager. |
| [0076](0076%20-%20OmniHub%20Model%20Repository%20Abstraction.md) | OmniHub Model Repository Abstraction | Accepted | `ModelRepository` interface with priority-based registry decoupling model source backends (GitHub, HuggingFace) from OmniHubUtils. |

### SameDiff Execution Framework

| # | Title | Status | Description |
|---|---|---|---|
| [0008](0008%20-%20Nd4j%20eager%20%20shape%20computation%20.md) | Eager Shape Computation | Accepted | Dynamic shape computation during model import, resolving shapes as variables are created rather than at full graph execution time. |
| [0018](0018%20-%20SDValue.md) | SDValue | Discussion | Union type in SameDiff for lists, maps, and tensors to flow through graph edges (like ONNX sequences/optionals). |
| [0019](0019%20-%20Invoke.md) | Invoke Op | Discussion | `Invoke` op for executing named SameDiff sub-graphs as nodes in a parent graph. |
| [0020](0020%20-%20New%20Control%20flow.md) | New Control Flow | Discussion | ONNX-compatible `Loop` op support for iteration over sub-graph attributes. |
| [0021](0021%20-%20Create%20View.md) | CreateView Op | Discussion | Live (non-copying) view creation using dynamic index variables for in-place ops. |
| [0022](0022%20-%20Dynamic%20Indexing.md) | Dynamic Indexing | Discussion | Runtime-resolved negative indices in SameDiff (NumPy-style dynamic indexing). |
| [0023](0023%20-%20UDFs.md) | User-Defined Functions | Implemented | First-class UDF mechanism with base class, FlatBuffers serialization, and `sd.udf()` registration for custom ops with custom gradients. |
| [0048](0048%20-%20Improved%20SameDiff%20Execution%20Framework.md) | Improved SameDiff Execution Framework | Accepted | DAG-based cached execution engine replacing the broken `initSubgraph` interpreter, with variable evolution tracking and frame-aware ordering for control flow. |

### DynamicShapePlan (DSP) Execution Engine

| # | Title | Status | Description |
|---|---|---|---|
| [0061](0061%20-%20DynamicShapePlan%20Execution.md) | DynamicShapePlan Execution | Implemented | **Core DSP architecture.** DynamicShapePlan as the sole optimized SameDiff execution path. Lifecycle: compile → warmup → freeze → CUDA graph capture → replay. Shape-keyed plan cache with pin/unpin. |
| [0062](0062%20-%20Java-Side%20Shape%20Inference.md) | Java-Side Shape Inference | Implemented | Java-side shape calculation and per-slot caching to eliminate JNI round-trips and GPU-to-host syncs for stable-shape ops during autoregressive decode. |
| [0066](0066%20-%20InferenceSession%20Autoregressive%20Optimization.md) | InferenceSession Autoregressive Optimization | Implemented | Shape caching, array reuse, suppressed redundant GPU syncs, and pool-trim throttling targeting autoregressive generation throughput. |
| [0078](0078%20-%20DSP%20Diagnostic%20Framework%20Extensions.md) | DSP Diagnostic Framework Extensions | Accepted | Three new diagnostic categories (STREAM_SYNC, MULTI_DEVICE, GRAPH_REPLAY) with programmatic replay readiness and phase-transition tracking via DspDebugger Java API. |
| [0079](0079%20-%20NativeDynamicShapePlan%20Structural%20Refactoring.md) | NativeDynamicShapePlan Structural Refactoring | Accepted | Decomposed 18K-line implementation into separate structs for immutable definition vs. mutable state, removing macro indirection and aliased members that caused silent bugs. |
| [0084](0084%20-%20DSP%20Execution%20State%20Simplification.md) | DSP Execution State Simplification | Accepted | Removed redundant ExecutionPhase enum (collapsed into SegmentLifecycleState) and pruned dead SlotState values to eliminate parallel state machines that diverged silently. |

### DSP Correctness Fixes (Bug-Fix ADRs)

| # | Title | Status | Description |
|---|---|---|---|
| [0080](0080%20-%20Triton%20Fusion%20Replay%20Correctness%20and%20Accuracy%20Validation.md) | Triton Fusion Replay Correctness | Accepted | Fixed stale pinned host copies from incorrect `tl_graphExecutionActive` scoping, op trait misclassification, GELU formula mismatch, and unnecessary per-step recompilation. |
| [0081](0081%20-%20DSP%20View%20Shape%20Correctness%20and%20Execution%20Comparison%20Diagnostics.md) | DSP View Shape Correctness | Accepted | Fixed view builder ignoring ONNX permutation inputs and emitting non-contiguous broadcast strides, both causing silent wrong-number corruption across transformer layers. |
| [0082](0082%20-%20CUDA%20Graph%20Replay%20Pointer%20Stability%20and%20Frozen%20Steady-State.md) | CUDA Graph Replay Pointer Stability | Accepted | Fixed argTableStable being permanently false, external inputs polluting the phase-transition address key, and disabled frozen-constant detection causing stale KV data replay. |
| [0083](0083%20-%20Thread-Local%20Cast%20Cache%20Leak%20Prevention.md) | Thread-Local Cast Cache Leak Prevention | Accepted | Replaced push_back growth with indexed overwrite in tl_castCacheA/B, eliminating ~250 MB/step GPU memory leak in cuBLAS dtype-cast path (OOM around token 100). |
| [0089](0089%20-%20CUDA%20Graph%20Capture%20and%20Replay.md) | CUDA Graph Capture and Replay Orchestration | Accepted | Complete lifecycle specification (warmup → compile → capture → replay) for CUDA graphs with interleaved Triton sub-kernels and native gap ops captured into per-segment graphs. |
| [0090](0090%20-%20Device%20Transfer%20Management%20Framework.md) | Device Transfer Management Framework | Proposed | Five-priority framework for multi-GPU memory: per-variable device pinning, transfer diagnostics, replica leak detection, pointer stability validation, plan destruction under OOM. |

### Triton Graph Backend

| # | Title | Status | Description |
|---|---|---|---|
| [0071](0071%20-%20Triton%20Graph%20Backend.md) | Triton Graph Backend | Implemented | OpenAI Triton as a kernel fusion backend for DSP. Compiles fusible op segments into single kernels where intermediates stay in registers rather than global memory. OpTraitTable.cpp is SSOT for mappability. |

### DSP Deployment & Serving

| # | Title | Status | Description |
|---|---|---|---|
| [0073](0073%20-%20DSP%20Self-Contained%20Runtime%20SDK%20and%20SDZ%20Deployment.md) | DSP Self-Contained Runtime SDK (SDX) | Partially Implemented | Stable native ABI for shipping per-platform DSP runtime binaries that load .sdz/.sdnb models directly without Java graph construction. Multi-language SDK bindings (C#, Java, Kotlin, Python, Rust, Swift). |
| [0074](0074%20-%20SDX%20Runtime%20Serving%20Protocol%20(REST%20%2B%20gRPC).md) | SDX Runtime Serving Protocol | Accepted | gRPC as primary binary protocol, REST as secondary for serving SDX runtime models, with caller-provided output buffers matching the C ABI. |

### Memory Management

| # | Title | Status | Description |
|---|---|---|---|
| [0024](0024%20-%20Workspaces.md) | Workspaces | Implemented | Ring-buffer-backed workspace regions (scoped via try-with-resources) to avoid redundant allocation across cyclic neural network inference patterns. |
| [0028](0028%20-%20Offset%20centralization.md) | Offset Centralization | Proposed | Centralize offset storage into NDArray objects and introduce `OpaqueNDArray` to simplify Java-C++ interop. |
| [0033](0033-shape-buffer-trie.md) | Shape Buffer Trie | Implemented | `DirectShapeTrie` with striped mutex locking replacing `ShapeDescriptor`-based unordered-map cache for shape-buffer allocation. |
| [0060](0060%20-%20CUDA%20Async%20Memory%20Pool.md) | CUDA Async Memory Pool | Implemented | `cudaMallocAsync`-based pooling replacing `cudaMalloc`/`cudaFree` to eliminate per-allocation driver latency, with multi-GPU OOM failover. |
| [0063](0063%20-%20ArrayCacheMemoryMgr%20Buffer%20Reuse.md) | ArrayCacheMemoryMgr Buffer Reuse | Implemented | Capacity-indexed TreeMap with LRU eviction, fixing a closeable-gate leak that permanently lost growth-factor-oversized buffers. |
| [0065](0065%20-%20Multi-GPU%20Memory%20Management.md) | Multi-GPU Memory Management | Implemented | Total-memory-based GPU device selection, multi-stage OOM failover, P2P-aware compute budgeting, and stream-safe cross-device array migration. |
| [0070](0070%20-%20GC%20Pressure%20Optimization.md) | GC Pressure Optimization | Implemented | Heap-pressure-aware conditional GC replacing blind periodic `System.gc()`, fixing a PhantomReference strong-reference cycle that prevented GC-based cleanup. |
| [0086](0086%20-%20Multi-Backend%20Workspace%20System.md) | Multi-Backend Workspace System | Implemented | Extended workspace memory system with multi-device tracking, MSI-style coherence protocol, and cross-device transfer support for hybrid CPU/GPU execution. |

### LLM/VLM Inference

| # | Title | Status | Description |
|---|---|---|---|
| [0064](0064%20-%20VLM%20Inference%20Pipeline.md) | VLM Inference Pipeline | Implemented | Multi-model, multi-GPU VLM pipeline with deferred model release and DSP-backed autoregressive generation for vision-language models (SmolDocling). |
| [0067](0067%20-%20Scaled%20Dot-Product%20Attention%20Optimization.md) | Scaled Dot-Product Attention Optimization | Implemented | Fused Q@K^T, softmax, attn@V into a single kernel via oneDNN/cuDNN backends with compiled partition caching, eliminating intermediate materialization. |
| [0069](0069%20-%20OCR%20Operations.md) | OCR Operations | Implemented | Native OCR engine using SameDiff-executed ONNX model, integrated with VLM image preprocessing pipeline for GPU-accelerated document understanding. |
| [0092](0092%20-%20Op%20Execution%20Timing%20Tracker.md) | Op Execution Timing Tracker | Accepted | Lock-free ring-buffer op timing with phase-level granularity (validation, shape calc, memory alloc, helper exec, native exec), Welford variance, logarithmic histograms, Chrome Trace/CSV export. |

### Training & PEFT

| # | Title | Status | Description |
|---|---|---|---|
| [0057](0057%20-%20Mixed%20Precision%20Training.md) | Mixed Precision Training | Accepted | Dynamic/static loss scaling, gradient accumulation, and TrainingConfig integration for FP16/BF16 mixed-precision training in SameDiff. |
| [0068](0068%20-%20LoRA%20Fused%20MatMul.md) | LoRA Fused MatMul | Implemented | Fused four-step LoRA computation (base GEMM + two low-rank GEMMs + accumulation) into a single op to reduce kernel launches and intermediate allocations. |
| [0077](0077%20-%20PEFT%20and%20Knowledge%20Distillation%20Extensions.md) | PEFT & Knowledge Distillation | Accepted | PiSSA, LoRA+, BitFit, VeRA, DyLoRA PEFT variants plus KL/feature/attention distillation API with DistillationTrainer orchestrator. |

### Kernel Selection & Multi-Backend Execution

| # | Title | Status | Description |
|---|---|---|---|
| [0055](0055-Kernel_Selection_And_Dynamic_Loading.md) | Kernel Selection & Dynamic Loading | Accepted | Runtime auto-tuning (`KernelAutoTuner`), persistent performance caching (`KernelPerformanceRegistry`), and dynamic shared-library plugin loading on top of PlatformHelper. |
| [0058](0058%20-%20Multi-Backend%20Kernel%20Selection%20and%20Management.md) | Multi-Backend Kernel Selection | Accepted | Multi-level (global/category/per-op) kernel selection system with runtime benchmarking and fluent Java API. |
| [0059](0059%20-%20Multi-Backend%20Op%20Execution%20System.md) | Multi-Backend Op Execution | Proposed | Runtime loading of multiple backends simultaneously with automatic op routing based on input data device location. |
| [0091](0091%20-%20LlamaCpp%20OneDNN%20cuDNN%20Backend%20Classifiers.md) | LlamaCpp/OneDNN/cuDNN Backend Classifiers | Accepted | Optional classifier-based Maven/CMake profiles for llama.cpp, OneDNN, and cuDNN backends following the existing avx2/avx512 extension pattern. |

### Alternative Hardware Backends

| # | Title | Status | Description |
|---|---|---|---|
| [0072](0072%20-%20TPU%20Backend.md) | TPU Backend (PJRT) | In Progress | TPU backend via Google's PJRT API, mapping SameDiff graphs to XLA HLO IR with PJRT compilation caching and graph replay. |
| [0085](0085%20-%20MLIR%20JIT%20Compilation%20Backend.md) | MLIR JIT Compilation Backend | Accepted | Optional MLIR JIT backend for libnd4j enabling graph-level op fusion, runtime shape specialization, and cross-platform code generation. |
| [0087](0087%20-%20ZLUDA%20Transpiler%20Support.md) | ZLUDA Transpiler Support | Accepted | ZLUDA runtime transpiler to run existing CUDA codebase on AMD (ROCm) and Intel (oneAPI) GPUs without maintaining separate HIP/SYCL codebases. |
| [0088](0088%20-%20Hexagon%20MLIR%20Backend.md) | Hexagon MLIR NPU Backend | In Progress | Qualcomm Hexagon NPU backend using hexagon-mlir following the DSP graph backend pattern, targeting INT8/INT16 inference on mobile HVX/HTP hardware. |

### Test Architecture

| # | Title | Status | Description |
|---|---|---|---|
| [0006](0006%20-%20Test%20architecture.md) | JUnit 5 Tag Usage | Proposed | JUnit 5 Tags to categorize tests (long/flaky, download-heavy, quick, integration) for selective test execution. |
| [0010](0010%20-%20Test%20module%20consolidation.md) | Test Module Consolidation | Proposed | Consolidate all tests into unit/component/integration/e2e/regression hierarchy. Realized as the `platform-tests` module. |
| [0056](0056%20-%20Libnd4j%20Native%20Test%20Integration.md) | Libnd4j Native Test Integration | Accepted | Run libnd4j GTest suites as JUnit 5 `DynamicTest` instances inside Maven Surefire via `LibNd4jNativeTestRunner`. |

### Build System & Platform

| # | Title | Status | Description |
|---|---|---|---|
| [0007](0007%20-%20Nd4j%20classifiers.md) | Nd4j Classifiers | Accepted | JavaCPP platform classifier extensions (e.g. `avx256-dnnl-2.2`) to expose different native build configurations. |
| [0016](0016%20-%20Java%209%2B%20Support.md) | Java 9+ Module Support | Discussion | `module-info.java` metadata via moditect plugin for Java 9+ module system compatibility. |
| [0030](0030%20-%20Type%20Promotion.md) | Smaller Type-Limited Artifact | Proposed | Publish a second Maven artifact with limited data type support (float-only) to reduce binary size. |
| [0031](0031%20-%20New%20generate%20combinations%20macros.md) | Type Combination Macros | Proposed | Preprocessor macros for automating exhaustive template instantiation for all type combinations. |
| [0039](0039%20-%20Selective%20rendering%20type%20system.md) | Selective Rendering Type System | Implemented | CMake-level semantic filtering engine that automatically determines valid type combinations and generates compile-time macros, avoiding template combinatorial explosion. |
| [0041](0041%20-%20CUDA%20Architecture%20Reduction.md) | CUDA Architecture Target Reduction | Proposed | Drop pre-Ampere compute capabilities (target 8.6+ only) to cut build time ~75% and binary size from ~800MB to ~200MB. |
| [0042](0042%20-%20Android%20NDK%20Migration.md) | Android NDK Migration | Proposed | Upgrade from NDK r21d (2019) to r27d (LLVM 18), minimum API 21, full LLVM toolchain. |
| [0045](0045%20-%20Android%20Cross-Compilation%20Modernization.md) | Android Cross-Compilation Modernization | Proposed | Modernized CMake toolchain files with flexible NDK path detection and explicit LLVM tool specification. |
| [0046](0046%20-%20CUDA%20Macro%20Standardization.md) | CUDA Macro Standardization | Proposed | Replace mixed `__CUDABLAS__`/`__CUDACC__` with consistent `SD_`-prefixed hierarchy (`SD_CUDA`, `SD_HOST`, `SD_DEVICE`, etc.). |
| [0047](0047%20-%20Comprehensive%20Template%20Instantiation%20Migration.md) | Template Instantiation Migration | Implemented | Platform-aware type equivalence classes (e.g., `long`/`int64_t`/`LongType`) ensuring all alias variants are instantiated to eliminate cross-platform linker errors. |

### Namespace Migration

| # | Title | Status | Description |
|---|---|---|---|
| [0036](0036%20-%20Namespace%20refactoring.md) | Namespace Refactoring (OpenRewrite) | Proposed | OpenRewrite recipe to migrate all Java packages from `org.nd4j`/`org.deeplearning4j` to `org.eclipse.deeplearning4j`. |
| [0038](0038%20-%20Namespace%20migration%20to%20Eclipse.md) | Eclipse Namespace Migration | Proposed | Two-phase Eclipse Foundation namespace migration: Phase 1 = Maven groupIds; Phase 2 = full package rename. Driven by OSSRH shutdown. |

### Debugging & Profiling

| # | Title | Status | Description |
|---|---|---|---|
| [0024](0024%20-%20Execution%20Tracing.md) | Graph Execution Trace Collection | Implemented | Capture op-execution metadata (shapes, op names) into a vector replayable as a SameDiff graph for debugging execution order. |
| [0025](0025%20-%20Javacpp%20Pointer%20Tracking%20with%20AspectJ.md) | JavaCPP Pointer Tracking (AspectJ) | Implemented | Compile-time weaving to intercept JavaCPP pointer allocations/deallocations for memory usage reporting. |
| [0026](0026%20-%20LIbnd4j%20method%20backtraces.md) | Libnd4j Function Tracing | Implemented | GCC `-finstrument-functions` via `-Dlibnd4j.functrace=ON` to trace all C++/CUDA function calls. |
| [0027](0027%20-%20Bytebuddy%20op%20execution%20logger) | ByteBuddy Op Execution Logger | Proposed | Java agent using ByteBuddy to record ND4J op executions into an H2 database for cross-version regression detection. |
| [0032](0032-%20CPP%20Debugging.md) | C++ Print Debugging Utilities | Implemented | Three build-time-controlled debugging utilities (print indices, print math, preprocessor output) toggled via Maven/CMake flags. |
| [0037](0037%20-%20Ppstep%20integration%20with%20recording.md) | Ppstep Preprocessor Debugger | Implemented | Interactive macro debugger as an optional CMake target with recording and break-on-error commands. |
| [0049](0049%20-%20AddressSanitizer%20Memory%20Leak%20Detection.md) | AddressSanitizer (ASAN) for JNI | Implemented | ASAN configuration tuned for JNI with mismatch suppression and `ThreadPool` destructor ordering fixes. |
| [0050](0050%20-%20Clang%20Sanitizers%20for%20JNI%20Memory%20Debugging.md) | Clang Sanitizers for JNI | Implemented | CMake `SD_SANITIZERS` flag for ASAN/MSAN/LSAN with embedded sanitizer RPATH in shared libraries. |
| [0051](0051%20-%20NDArray%20and%20DataBuffer%20Lifecycle%20Tracking%20for%20Memory%20Leak%20Detection.md) | NDArray/DataBuffer Lifecycle Tracking | Implemented | Two-level tracker (NDArray + DataBuffer PRIMARY/SPECIAL) with full stack traces, periodic reports, flamegraph output, and JNI API for Java-side leak statistics. |

---

## Numeric Index (complete)

| # | Title | Status |
|---|---|---|
| 0001 | SameDiff File Format | Accepted |
| 0002 | ONNX Runtime Module | Implemented |
| 0003 | Import IR | Implemented |
| 0003 | NDArray Padded Strides for ARM Compute | Implemented |
| 0004 | Mapping IR | Implemented |
| 0005 | Interpreter | Rejected |
| 0006 | JUnit 5 Tag Usage | Proposed |
| 0007 | Nd4j Classifiers | Accepted |
| 0008 | Eager Shape Computation | Accepted |
| 0009 | Import Node Pre-Processing | Discussion |
| 0010 | Test Module Consolidation | Proposed |
| 0011 | OmniHub Zoo Download | Discussion |
| 0012 | OmniHub Download Implementations | Discussion |
| 0013 | OmniHub Consumption API | Accepted |
| 0014 | Replace Old Model Zoo | Accepted |
| 0015 | Unified Resource Manager | Discussion |
| 0016 | Java 9+ Module Support | Discussion |
| 0018 | SDValue | Discussion |
| 0019 | Invoke Op | Discussion |
| 0020 | New Control Flow (Loop) | Discussion |
| 0021 | CreateView Op | Discussion |
| 0022 | Dynamic Indexing | Discussion |
| 0023 | User-Defined Functions | Implemented |
| 0024 | Execution Tracing | Implemented |
| 0024 | Workspaces | Implemented |
| 0025 | JavaCPP Pointer Tracking (AspectJ) | Implemented |
| 0026 | Libnd4j Function Tracing | Implemented |
| 0027 | ByteBuddy Op Execution Logger | Proposed |
| 0028 | Offset Centralization | Proposed |
| 0030 | Smaller Type-Limited Artifact | Proposed |
| 0031 | Type Combination Macros | Proposed |
| 0032 | C++ Print Debugging | Implemented |
| 0033 | Shape Buffer Trie | Implemented |
| 0034 | FlatBuffers Modernization | Implemented |
| 0035 | SameDiff Unified Container (SDNB/SDZ) | Implemented |
| 0036 | Namespace Refactoring (OpenRewrite) | Proposed |
| 0037 | Ppstep Preprocessor Debugger | Implemented |
| 0038 | Eclipse Namespace Migration | Proposed |
| 0039 | Selective Rendering Type System | Implemented |
| 0041 | CUDA Architecture Target Reduction | Proposed |
| 0042 | Android NDK Migration | Proposed |
| 0045 | Android Cross-Compilation Modernization | Proposed |
| 0046 | CUDA Macro Standardization | Proposed |
| 0047 | Template Instantiation Migration | Implemented |
| 0048 | Improved SameDiff Execution Framework | Accepted |
| 0049 | AddressSanitizer for JNI | Implemented |
| 0050 | Clang Sanitizers for JNI | Implemented |
| 0051 | NDArray/DataBuffer Lifecycle Tracking | Implemented |
| 0052 | GGML/GGUF Model Import | Implemented |
| 0053 | GGML Quantization Handling | Implemented |
| 0054 | GGML Architecture Detection | Implemented |
| 0055 | Kernel Selection & Dynamic Loading | Accepted |
| 0056 | Libnd4j Native Test Integration | Accepted |
| 0057 | Mixed Precision Training | Accepted |
| 0058 | Multi-Backend Kernel Selection | Accepted |
| 0059 | Multi-Backend Op Execution | Proposed |
| 0060 | CUDA Async Memory Pool | Implemented |
| 0061 | DynamicShapePlan Execution | Implemented |
| 0062 | Java-Side Shape Inference | Implemented |
| 0063 | ArrayCacheMemoryMgr Buffer Reuse | Implemented |
| 0064 | VLM Inference Pipeline | Implemented |
| 0065 | Multi-GPU Memory Management | Implemented |
| 0066 | InferenceSession Autoregressive Optimization | Implemented |
| 0067 | Scaled Dot-Product Attention Optimization | Implemented |
| 0068 | LoRA Fused MatMul | Implemented |
| 0069 | OCR Operations | Implemented |
| 0070 | GC Pressure Optimization | Implemented |
| 0071 | Triton Graph Backend | Implemented |
| 0072 | TPU Backend (PJRT) | In Progress |
| 0073 | DSP Self-Contained Runtime SDK (SDX) | Partially Implemented |
| 0074 | SDX Runtime Serving Protocol | Accepted |
| 0076 | OmniHub Model Repository Abstraction | Accepted |
| 0077 | PEFT & Knowledge Distillation | Accepted |
| 0078 | DSP Diagnostic Framework Extensions | Accepted |
| 0079 | NativeDynamicShapePlan Structural Refactoring | Accepted |
| 0080 | Triton Fusion Replay Correctness | Accepted |
| 0081 | DSP View Shape Correctness | Accepted |
| 0082 | CUDA Graph Replay Pointer Stability | Accepted |
| 0083 | Thread-Local Cast Cache Leak Prevention | Accepted |
| 0084 | DSP Execution State Simplification | Accepted |
| 0085 | MLIR JIT Compilation Backend | Accepted |
| 0086 | Multi-Backend Workspace System | Implemented |
| 0087 | ZLUDA Transpiler Support | Accepted |
| 0088 | Hexagon MLIR NPU Backend | In Progress |
| 0089 | CUDA Graph Capture and Replay Orchestration | Accepted |
| 0090 | Device Transfer Management Framework | Proposed |
| 0091 | LlamaCpp/OneDNN/cuDNN Backend Classifiers | Accepted |
| 0092 | Op Execution Timing Tracker | Accepted |
