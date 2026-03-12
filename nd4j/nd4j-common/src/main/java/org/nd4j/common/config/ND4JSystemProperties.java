/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.common.config;


import java.io.File;
import java.net.URL;

public class ND4JSystemProperties {


    /**
     * Applicability: always.
     * Description: Nd4j
     * has a DeallocatorService for handling automatic
     * deallocation of ndarrays. This can cause crashes.
     */
    public final static String NO_ARRAY_GC = "org.nd4j.noarraygc";

    /**
     * Applicability: Always<br>
     * Description: Sets the default datatype for ND4J - should be one of "float", "double", "half".
     * ND4J is set to float (32-bit floating point values) by default.
     */
    public static final String DTYPE = "dtype";
    /**
     * Applicability: Always<br>
     * Description: By default, ND4J will log some information when the library has completed initialization, such as the
     * backend (CPU or CUDA), CPU/Devices, memory etc. This system property can be used to disable the logging of this
     * initialization information
     */
    public static final String LOG_INITIALIZATION = "org.nd4j.log.initialization";

    /**
     * Applicability: nd4j-native when running non-AVX binary on an AVX compatible CPU<br>
     * Description: Set to true to avoid logging AVX warnings (i.e., running generic x86 binaries on an AVX2 system)
     */
    public static final String ND4J_IGNORE_AVX = "org.nd4j.avx.ignore";

    /**
     * Applicability: Always<br>
     * Description: This system property defines the maximum amount of off-heap memory that can be used.
     * ND4J uses off-heap memory for storage of all INDArray data. This off-heap memory is a different
     * pool of memory to the on-heap JVM memory (configured using standard Java Xms/Xmx options).
     * Default: 2x Java XMX setting
     *
     * @see #JAVACPP_MEMORY_MAX_PHYSICAL_BYTES
     */
    public static final String JAVACPP_MEMORY_MAX_BYTES = "org.bytedeco.javacpp.maxbytes";
    /**
     * Applicability: Always<br>
     * Description: This system property defines the maximum total amount of memory that the process can use - it is
     * the sum of both off-heap and on-heap memory. This can be used to provide an upper bound on the maximum amount
     * of memory (of all types) that ND4J will use
     *
     * @see #JAVACPP_MEMORY_MAX_BYTES
     */
    public static final String JAVACPP_MEMORY_MAX_PHYSICAL_BYTES = "org.bytedeco.javacpp.maxphysicalbytes";

    /**
     * Applicability: ND4J Temporary file creation/extraction for ClassPathResource, memory mapped workspaces, and  <br>
     * Description: Specify the local directory where temporary files will be written. If not specified, the default
     * Java temporary directory (java.io.tmpdir system property) will generally be used.
     */
    public static final String ND4J_TEMP_DIR_PROPERTY = "org.nd4j.tempdir";

    /**
     * Applicability: always - but only if an ND4J backend cannot be found/loaded via standard ServiceLoader mechanisms<br>
     * Description: Set this property to a set fully qualified JAR files to attempt to load before failing on
     * not loading a backend. JAR files should be semi-colon delimited; i.e., "/some/file.jar;/other/path.jar".
     * This should rarely be required in practice - for example, only in dynamic class loading/dynamic classpath scenarios<br>
     * For equivalent system property, see {@link ND4JEnvironmentVars#BACKEND_DYNAMIC_LOAD_CLASSPATH} for the equivalent
     * system property (the system property will take precidence if both are set)
     */
    public static final String DYNAMIC_LOAD_CLASSPATH_PROPERTY = "org.nd4j.backend.dynamicbackend";
    /**
     * Applicability: Always<br>
     * Description Setting the system property to false will stop ND4J from performing the version check, and logging any
     * warnings/errors. By default, the version check is enabled.<br>
     * Note: the version check is there for a reason! Using incompatible versions of ND4J/DL4J etc is likely to cause
     * issues, and should be avoided.
     */
    public static final String VERSION_CHECK_PROPERTY = "org.nd4j.versioncheck";
    /**
     * Applicability: always<br>
     * Description: Used to specify the maximum number of elements (numbers) to print when using DataBuffer.toString().
     * Use -1 to print all elements (i.e., no limit). This is usually to avoid expensive toString() calls on buffers
     * which may have millions of elements - for example, in a debugger<br>
     * Default: 1000
     */
    public static final String DATABUFFER_TO_STRING_MAX_ELEMENTS = "org.nd4j.databuffer.tostring.maxelements";
    /**
     * Applicability: nd4j-native backend, when multiple BLAS libraries are available<br>
     * Description: This system property can be used to control which BLAS library is loaded and used by ND4J.
     * For example, {@code org.bytedeco.javacpp.openblas.load=mkl_rt} can be used to load a default installation of MKL.
     * However, MKL is liked with by default (when available) so setting this option explicitly is not usually required.
     * For more details, see <a href="https://github.com/bytedeco/javacpp-presets/tree/master/openblas#documentation">https://github.com/bytedeco/javacpp-presets/tree/master/openblas#documentation</a>
     */
    public static final String ND4J_CPU_LOAD_OPENBLAS = "org.bytedeco.openblas.load";
    /**
     * Applicability: nd4j-native backend, when multiple BLAS libraries are available<br>
     * Description: This system property can be used to control which BLAS library is loaded and used by ND4J.
     * Similar to {@link #ND4J_CPU_LOAD_OPENBLAS} but when this is set, LAPACK will not be loaded
     */
    public static final String ND4J_CPU_LOAD_OPENBLAS_NOLAPACK = "org.bytedeco.openblas_nolapack.load";
    /**
     * Applicability: nd4j-parameter-server, dl4j-spark (gradient sharing training master)<br>
     * Description: Aeros in a high-performance communication library used in distributed computing contexts in some
     * places in ND4J and DL4J. This term buffer length determines the maximum message length that can be sent via Aeron
     * in a single message. It can be increased to avoid exceptions such as {@code Encoded message exceeds maxMessageLength of 2097152},
     * at the expense of increased memory consumption (memory consumption is a multiple of this). It is specified in bytes
     * with no unit suffix. Default value: 33554432 (32MB).
     * <b>IMPORTANT</b>: This value must be an exact power of 2.<br>
     * Note also the maximum effective size is 128MB (134217728) (due to Aeron internal limits - beyond which increasing
     * the buffer size will have no effect)
     */
    public static final String AERON_TERM_BUFFER_PROP = "aeron.term.buffer.length";

    /**
     * Applicability: nd4j-common {@link Resources} class (and hence {@link StrumpfResolver})<br>
     * Description: When resolving resources from a Strumpf resource file (Example: {@code Resources.asFile("myFile.txt")}
     * where should the remote files be downloaded to?<br>
     * This is generally used for resolving test resources, but can be used for Strumpf resource files generally.
     */
    public static final String RESOURCES_CACHE_DIR = "org.nd4j.test.resources.cache.dir";

    /**
     * Applicability: nd4j-common {@link Resources} class (and hence {@link StrumpfResolver})<br>
     * Description: When resolving resources from a Strumpf resource file (Example: {@code Resources.asFile("myFile.txt")}
     * what should be the connection timeout, as used by {@link org.apache.commons.io.FileUtils#copyURLToFile(URL, File, int, int)}<br>
     * Default: {@link ResourceFile#DEFAULT_CONNECTION_TIMEOUT}
     */
    public static final String RESOURCES_CONNECTION_TIMEOUT = "org.nd4j.resources.download.connectiontimeout";

    /**
     * Applicability: nd4j-common {@link Resources} class (and hence {@link StrumpfResolver})<br>
     * Description: When resolving resources from a Strumpf resource file (Example: {@code Resources.asFile("myFile.txt")}
     * what should be the connection timeout, as used by {@link org.apache.commons.io.FileUtils#copyURLToFile(URL, File, int, int)}<br>
     * Default: {@link ResourceFile#DEFAULT_READ_TIMEOUT}
     */
    public static final String RESOURCES_READ_TIMEOUT = "org.nd4j.resources.download.readtimeout";

    /**
     * Applicability: nd4j-common {@link Resources} class (and hence {@link StrumpfResolver})<br>
     * Description: When resolving resources, what local directories should be checked (in addition to the classpath) for files?
     * This is optional. Multiple directories may be specified, using comma-separated paths
     */
    public static final String RESOURCES_LOCAL_DIRS = "org.nd4j.strumpf.resource.dirs";

    /**
     * Whether caching should be enabled for samediff memory managers.
     * This ia mainly for the default ArrayCacheMemoryMgr.
     * Sometimes arrays for performance reasons get reused
     * during a samediff inference session. This may have bad side effects (especially involving views)
     * This allows enabling or disabling of that behavior.
     */
    public final static String SAMEDIFF_MEMORY_CACHE_ENABLE = "org.nd4j.autodiff.samediff.cache.enable";

    /**
     * Used to trigger loading the import reflection cache. This allows the user to control the initial scan
     * of the ImportReflectionCache in samediff-import-onnx and samediff-import-tensorflow.
     * Sometimes delayed initialization is favorable for use cases like graalvm AOT.
     */
    public final static String INIT_IMPORT_REFLECTION_CACHE = "org.nd4j.samediff.frameworkimport.initcache";


    /**
     * Used to point to a json resource that contains json for a ClassGraph ScanResult.
     * This may be needed when using AOT. Graalvm can not handle classpath scanning very well.
     * A pre scanned resource option will allow model import that relies on annotation scanning
     * to operate even when using AOT.
     */
    public final static String CLASS_GRAPH_SCAN_RESOURCES = "org.nd4j.samediff.frameworkimport.classgraph.scan.json";

    /**
     * Whether to initialize the native ops holder or not.
     * Depending on whether we are running in native image or not, disabling automatic initialization
     * and setting the relevant native ops elsewhere might be necessary.
     * For more see {@link org.nd4j.nativeblas.NativeOpsHolder }
     */
    public final static String INIT_NATIVEOPS_HOLDER = "org.nd4j.nativeblas.nativeops.init";



    /**
     * Maximum memory fraction to use as cache. For more see:
     * https://github.com/deeplearning4j/deeplearning4j/blob/2f08cc208b3bae1007bbbb001938d17c15926a09/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/memory/ArrayCacheMemoryMgr.java#L156-L157
     */
    public final static String CACHE_MEM_FRACTION = "org.nd4j.cache.cache_mem_fraction";
    /**
     * Below this size (elements), don't apply the
     * "largerArrayMaxMultiple" rule.
     * For more see: https://github.com/deeplearning4j/deeplearning4j/blob/2f08cc208b3bae1007bbbb001938d17c15926a09/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/memory/ArrayCacheMemoryMgr.java#L156-L157
     */
    public final static String SMALL_ARRAY_THRESHOLD = "org.nd4j.cache.cache_small_array_threshold";

    /**
     * Maximum multiple of the requested size to
     * return from the cache. If an array of size
     * 1024 is requested, and largerArrayMaxMultiple
     * is 2.0, then we'll return from the cache
     * the array with the smallest data buffer up to
     * 2.0*1024 elements; otherwise we'll return
     * a new array
     *
     *  For more see: https://github.com/deeplearning4j/deeplearning4j/blob/2f08cc208b3bae1007bbbb001938d17c15926a09/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/memory/ArrayCacheMemoryMgr.java#L156-L157
     */
    public final static String LARGE_ARRAY_MAX_MULTIPLE = "org.nd4j.cache.large_array_max_multiple";

    /**
     * For usage with the EventLogger. When the event logger is enabled, extra information will be tracked
     * including allocations, deallocations and other difficult to track down events.
     * Note that enabling this will add a certain amount of overhead.
     */
    public final static String EVENT_LOGGER_ENABLED = "org.nd4j.linalg.profiler.eventlogger.enabled";

    /**
     * For usage with the EventLogger. Tells the event logger to
     * format its log output as a date instead of the default nanoseconds.
     */
    public final static String EVENT_LOGGER_FORMAT_AS_DATE = "org.nd4j.linalg.profiler.eventlogger.logdate";


    /**
     * Set the package names to be scanned when using udfs.
     * The value should be a comma separated list.
     */
    public final static String UDF_NAME_SPACES = "org.nd4j.linalg.api.ops.udf.packages";


    /**
     * Set the  classes to be used in fully qualified format (org.nd4j.ClassName something for example..)
     * Note this will be checked BEFORE UDF_NAME_SPACES. Pick only 1  to use.
     * The value should be a comma separated list.
     */
    public final static String UDF_CLASSES = "org.nd4j.linalg.api.ops.udf.classes";

    /**
     * Sets the number of threads to be used with the deallocator service.
     */

    public final static String DEALLOCATOR_SERVICE_GC_THREADS = "org.nd4j.deallocator.threads";


    /**
     * Set the priority for the cpu backend.
     */
    public final static String BACKEND_PRIORITY_CPU = "org.nd4j.cpu.priority";

    /**
     * Set the priority for the cuda backend.
     */
    public final static String BACKEND_PRIORITY_GPU = "org.nd4j.gpu.priority";


    /**
     * Set the priority for the aurora backend.
     */
    public final static String BACKEND_PRIORITY_AURORA = "org.nd4j.aurora.priority";


    /**
     * Related to nd4j array events.
     * When determining the point of invocation or point of origin:
     * aka the points where the ndarray event was triggered
     * or the originating call site that kicked off the event
     * These properties represent patterns of regexes to exclude
     * from scanning when detrermining where the ndarray event was triggered.
     */
    public final static String ND4J_EVENT_LOG_POINT_OF_ORIGIN_PATTERNS = "org.nd4j.linalg.profiler.pointoforigin.patterns";

    /**
     * Applicability: Always<br>
     * Description: Controls whether multi-backend support is enabled. When multiple backends
     * (e.g., nd4j-native and nd4j-cuda) are on the classpath, ND4J will automatically discover
     * and initialize all available backends, enabling operations to run on any available device.
     * <p>
     * Default: true (multi-backend is enabled by default when multiple backends are available)
     * <p>
     * When enabled:
     * <ul>
     *   <li>All backends on classpath are discovered and initialized</li>
     *   <li>Device routing allows operations to run on CPU, GPU, or other accelerators</li>
     *   <li>Cross-device data transfers are handled automatically</li>
     * </ul>
     */
    public final static String MULTI_BACKEND_AUTO_ENABLED = "org.nd4j.backend.multi.auto";

    /**
     * Applicability: Always (when multi-backend is enabled)<br>
     * Description: Controls whether automatic device routing is enabled. When enabled,
     * ND4J will automatically select the optimal device for operations based on data location,
     * device availability, and memory constraints.
     * <p>
     * Default: true
     */
    public final static String DEVICE_ROUTING_AUTO_ENABLED = "org.nd4j.device.routing.auto";

    /**
     * Applicability: CUDA backend with nd4j-native also on classpath<br>
     * Description: When set to true, enables the DeviceAwareOpExecutioner with multi-backend
     * support. This allows CPU fallback execution when GPU memory is constrained or data
     * has spilled to CPU memory.
     * <p>
     * Requirements:
     * <ul>
     *   <li>Primary backend must be CUDA (or other GPU backend)</li>
     *   <li>nd4j-native (CPU backend) JAR must be on classpath</li>
     *   <li>Both native libraries must be loadable (no symbol conflicts)</li>
     * </ul>
     * <p>
     * Default: false (single-backend mode)
     * <p>
     * Example usage:
     * <pre>
     * java -Dnd4j.multibackend.enabled=true -jar myapp.jar
     * </pre>
     */
    public final static String MULTI_BACKEND_EXECUTION_ENABLED = "nd4j.multibackend.enabled";

    /**
     * Applicability: Multi-backend configuration<br>
     * Description: When set to true, disables automatic multi-backend discovery and initialization.
     * By default, ND4J will automatically detect and use all available backends on the classpath.
     * Set this to true to force single-backend mode even when multiple backends are available.
     * <p>
     * Default: false (auto-discovery enabled)
     * <p>
     * Example usage:
     * <pre>
     * java -Dnd4j.multibackend.disabled=true -jar myapp.jar
     * </pre>
     */
    public final static String MULTI_BACKEND_DISABLED = "nd4j.multibackend.disabled";

    /**
     * Applicability: When multi-backend execution is enabled<br>
     * Description: Controls whether to log routing decisions when operations are
     * routed to different backends. Useful for debugging and performance analysis.
     * <p>
     * Default: false
     */
    public final static String MULTI_BACKEND_LOG_ROUTING = "nd4j.multibackend.logrouting";

    /**
     * Applicability: When multi-backend execution is enabled<br>
     * Description: Minimum array size (in bytes) for considering CPU fallback.
     * Arrays smaller than this threshold will always stay on the primary device.
     * <p>
     * Default: 1048576 (1 MB)
     */
    public final static String MULTI_BACKEND_MIN_SPILLOVER_SIZE = "nd4j.multibackend.minspilloversize";

    /**
     * Applicability: Multi-backend configuration<br>
     * Description: Comma-separated list of secondary backend classes to load alongside
     * the primary backend. Each backend class must be on the classpath and implement Nd4jBackend.
     * <p>
     * Example usage:
     * <pre>
     * java -Dnd4j.backend.secondary=org.nd4j.linalg.cpu.nativecpu.CpuBackend -jar myapp.jar
     * </pre>
     * <p>
     * This allows loading CPU as a secondary backend when CUDA is primary, enabling
     * CPU fallback execution for spillover data.
     */
    public final static String SECONDARY_BACKEND_CLASSES = "nd4j.backend.secondary";

    /**
     * Applicability: Multi-backend configuration<br>
     * Description: Comma-separated list of secondary backend property files to load.
     * Each property file defines a backend configuration (opexec, native.ops, device.type, etc.)
     * <p>
     * Example usage:
     * <pre>
     * java -Dnd4j.backend.secondary.properties=nd4j-native.properties -jar myapp.jar
     * </pre>
     */
    public final static String SECONDARY_BACKEND_PROPERTIES = "nd4j.backend.secondary.properties";

    /**
     * Applicability: SameDiff workspace mode<br>
     * Description: When true, SameDiff automatically enables workspace-backed memory management
     * for CUDA backends. This uses bump allocation for intermediate arrays, avoiding per-op
     * cudaMalloc/cudaFree calls. Set to "false" to disable auto-enable.
     * Default: true
     */
    public final static String SAMEDIFF_WORKSPACE_AUTO = "nd4j.samediff.workspace.auto";

    /**
     * Applicability: SameDiff workspace mode<br>
     * Description: Initial workspace size in bytes for SameDiff workspace-backed memory management.
     * Default: 268435456 (256 MB)
     */
    public final static String SAMEDIFF_WORKSPACE_SIZE = "nd4j.samediff.workspace.size";

    // ---- DynamicShapePlan (DSP) execution properties ----

    /**
     * Applicability: DynamicShapePlan-based inference (autoregressive decoding)<br>
     * Description: Controls how often the CUDA memory pool is trimmed during DSP execution.
     * During steady-state decode, the pool reuses freed memory without trimming. Trimming every
     * step wastes time on cudaStreamSynchronize + cudaMemPoolTrimTo. Set to N to trim every
     * N steps. Step 0/1 always trims (prefill-to-decode transition).
     * <p>
     * Default: 10
     */
    public static final String DSP_TRIM_INTERVAL = "nd4j.dsp.trimInterval";

    /**
     * Applicability: DynamicShapePlan-based training<br>
     * Description: Enable DSP fast path for training iterations. When enabled, the forward+backward
     * pass is executed via DynamicShapePlanExecutor instead of per-op execution, with updaters
     * applied post-execution.
     * <p>
     * Default: true
     */
    public static final String DSP_TRAINING_ENABLED = "nd4j.dsp.training.enabled";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Per-slot byte threshold for selective cache eviction. When total cached
     * slot memory exceeds 512MB (after prefill), only arrays larger than this threshold are
     * evicted. Small utility arrays (scalars, shapes, small intermediates) survive and serve
     * decode step 1 with O(1) cache hits.
     * <p>
     * Default: 65536 (64KB)
     */
    public static final String DSP_PER_SLOT_EVICTION_THRESHOLD = "nd4j.dsp.perSlotEvictionThreshold";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Byte threshold below which freePendingBuffers uses a fast path that skips
     * the expensive GPU address dedup and live view range check. For decode steps with tiny
     * intermediates (seq_len=1), aliasing is extremely unlikely and the full dedup overhead
     * is unnecessary.
     * <p>
     * Default: 10485760 (10MB)
     */
    public static final String DSP_FAST_CLOSE_THRESHOLD = "nd4j.dsp.fastCloseThreshold";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Flush interval for pending buffer close during execution. Every N ops,
     * dead intermediates are freed to reduce peak GPU memory.
     * <p>
     * Default: 100
     */
    public static final String DSP_FLUSH_INTERVAL = "nd4j.dsp.flushInterval";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Byte threshold for memory-pressure flush during DSP execution.
     * When accumulated dead intermediate bytes exceed this threshold, flush immediately
     * instead of waiting for the op-count interval. Prevents multi-GB intermediate
     * accumulation between flush intervals.
     * <p>
     * Default: 256MB (268435456)
     */
    public static final String DSP_FLUSH_BYTE_THRESHOLD = "nd4j.dsp.flushByteThreshold";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: When true, force single-GPU mode for DSP execution even when multiple
     * CUDA devices are available.
     * <p>
     * Default: false
     */
    public static final String DSP_SINGLE_GPU = "nd4j.dsp.singleGpu";

    /**
     * Applicability: DynamicShapePlan-based inference with non-P2P multi-GPU<br>
     * Description: Fraction of available memory to use as budget for non-P2P secondary GPUs.
     * Non-P2P devices use host-staged transfers which may need extra headroom.
     * <p>
     * Default: 1.0 (use full available memory)
     */
    public static final String DSP_NON_P2P_BUDGET_FRACTION = "nd4j.dsp.nonP2pBudgetFraction";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: When true, serialize parallel worker op execution (only one worker thread
     * executes at a time). For debugging concurrent CUDA issues.
     * <p>
     * Default: false
     */
    public static final String DSP_SERIAL_EXEC = "nd4j.dsp.serialExec";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: When true, enable parallel worker execution across multiple devices
     * even when latent heap corruption is suspected.
     * <p>
     * Default: false
     */
    public static final String DSP_FORCE_PARALLEL = "nd4j.dsp.forceParallel";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Growth factor for intermediate slot cache allocation. When a slot cache miss
     * occurs, the allocated array is this factor times the required size, so it can serve future
     * steps without reallocation (e.g., growing KV cache).
     * <p>
     * Default: 2.0
     */
    public static final String DSP_SLOT_CACHE_GROWTH_FACTOR = "org.nd4j.dsp.slotCacheGrowthFactor";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Maximum size in bytes for the local buffer pool that persists across
     * execute() calls for array reuse.
     * <p>
     * Default: 2147483648 (2GB)
     */
    public static final String DSP_POOL_MAX_BYTES = "org.nd4j.dsp.pool.maxBytes";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Enable detailed per-op timing breakdown for DSP execution.
     * <p>
     * Default: false
     */
    public static final String INFERENCE_TIMING = "org.nd4j.inference.timing";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: When true, tells C++ to skip redundant output shape calculation
     * (Java pre-computes shapes and passes them to C++ via OpContext).
     * <p>
     * Default: true
     */
    public static final String DSP_SHAPE_OVERRIDE = "org.nd4j.inference.dynamicShapePlan.shapeOverride";

    /**
     * Applicability: SameDiff inference<br>
     * Description: Enable or disable DynamicShapePlan-based execution for autoregressive
     * inference with dynamic shapes (e.g., growing KV cache).
     * <p>
     * Default: true
     */
    public static final String DYNAMIC_SHAPE_PLAN_ENABLED = "org.nd4j.inference.dynamicShapePlan";

    /**
     * Applicability: SameDiff inference<br>
     * Description: Enable native C++ graph executor for DynamicShapePlan execution.
     * When enabled, the entire pre-compiled plan is sent to C++ via a single JNI call
     * instead of dispatching each op individually from Java. Falls back to Java executor
     * on any failure.
     * <p>
     * Default: true
     */
    public static final String DSP_NATIVE_EXECUTOR_ENABLED = "nd4j.dsp.nativeExecutor.enabled";

    /**
     * Applicability: CUDA native executor<br>
     * Description: Enable or disable CUDA Graphs for the native C++ plan executor.
     * When enabled, capturable segments of the execution plan are recorded as CUDA graphs
     * and replayed on subsequent calls, reducing kernel launch overhead.
     * <p>
     * Default: true
     */
    public static final String DSP_CUDA_GRAPHS_ENABLED = "nd4j.dsp.cudaGraphs.enabled";

    /**
     * Applicability: DSP execution engine<br>
     * Description: JIT compilation mode for DSP segment execution.<br>
     * Values: "graph" (default, CUDA graph only), "jit" (NVRTC JIT only),
     *         "graph+jit" (try JIT first, fall back to graph capture)
     */
    public static final String DSP_JIT_MODE = "nd4j.dsp.jitMode";

    /**
     * Applicability: DSP execution engine<br>
     * Description: Graph execution mode controlling which backend is used.<br>
     * Values: "AUTO" (default), "SLOT_BY_SLOT", "CUDA_GRAPHS", "NVRTC_JIT", "PTX_JIT", "TRITON",
     *         "MLX", "ARM_HYBRID", "NNAPI"
     */
    public static final String DSP_GRAPH_EXECUTION_MODE = "nd4j.dsp.graphExecutionMode";

    // ---- VLM speculative decoding properties ----

    /**
     * Applicability: VLM batch generation<br>
     * Description: Enable or disable n-gram speculative decoding in VLM batch generation.
     * When enabled, attempts to predict multiple future tokens from n-gram patterns in the
     * generated sequence, then verifies them in a single forward pass.
     * <p>
     * Default: true
     */
    public static final String VLM_SPECULATIVE = "nd4j.vlm.speculative";

    /**
     * Applicability: VLM batch generation (when speculative decoding is enabled)<br>
     * Description: Size of the n-gram to match when predicting future tokens.
     * Larger n-grams are more specific but require more context before matching.
     * <p>
     * Default: 3
     */
    public static final String VLM_SPECULATIVE_NGRAM_SIZE = "nd4j.vlm.speculative.ngramSize";

    /**
     * Applicability: VLM batch generation (when speculative decoding is enabled)<br>
     * Description: Maximum number of tokens to speculate in a single attempt.
     * More tokens can be accepted per step but the verification forward pass is larger.
     * <p>
     * Default: 5
     */
    public static final String VLM_SPECULATIVE_MAX_TOKENS = "nd4j.vlm.speculative.maxTokens";

    /**
     * Applicability: DSP native executor with CUDA graphs<br>
     * Description: Maximum KV cache sequence length for pre-allocation.
     * When set to a positive value, KV cache output slots (present_key / present_value outputs)
     * are pre-allocated at this maximum size on the first decode step and reused for all
     * subsequent steps. This keeps GPU buffer addresses stable across steps, which is required
     * for CUDA graph capture of autoregressive decoder models.
     * <p>
     * Set this to the maximum sequence length you expect (prompt length + max new tokens).
     * Example: -Dnd4j.dsp.maxKvCacheLength=2048
     * <p>
     * Default: 0 (disabled)
     */
    public static final String DSP_MAX_KV_CACHE_LENGTH = "nd4j.dsp.maxKvCacheLength";

    /**
     * When set, enables trace-level logging for DynamicShapePlanExecutor.
     * Presence of the property (any value) enables tracing.
     * Example: -Dnd4j.dsp.trace
     */
    public static final String DSP_TRACE = "nd4j.dsp.trace";

    /**
     * Controls how often CUDA error checks are performed in DynamicShapePlanExecutor.
     * Set to 1 to check after every op (useful for debugging CUDA errors).
     * Default: 50
     * Example: -Dnd4j.dsp.errorCheckInterval=1
     */
    public static final String DSP_ERROR_CHECK_INTERVAL = "nd4j.dsp.errorCheckInterval";

    /**
     * Controls how often the SameDiff workspace is reset during Java-side DSP execution.
     * Set to 1 to reset every op.
     * Default: 25
     * Example: -Dnd4j.dsp.workspaceResetInterval=1
     */
    public static final String DSP_WORKSPACE_RESET_INTERVAL = "nd4j.dsp.workspaceResetInterval";

    /**
     * When set to {@code true}, dumps the first few values of each output to the log
     * after every Java-side DSP execution step. Useful for comparing Java vs native output.
     * Default: false
     * Example: -Dnd4j.dsp.java.dumpOutputs=true
     */
    public static final String DSP_JAVA_DUMP_OUTPUTS = "nd4j.dsp.java.dumpOutputs";

    /**
     * When set to {@code true}, dumps the first few values of each output to the log
     * after every native-side DSP execution step. Useful for comparing Java vs native output.
     * Default: false
     * Example: -Dnd4j.dsp.native.dumpOutputs=true
     */
    public static final String DSP_NATIVE_DUMP_OUTPUTS = "nd4j.dsp.native.dumpOutputs";

    /**
     * When set to {@code true}, enables per-step execution timing in the native DSP executor.
     * Prints a breakdown of time spent in graph-replay vs slot-by-slot segments.
     * Default: false
     * Example: -Dnd4j.dsp.executionTiming=true
     */
    public static final String DSP_EXECUTION_TIMING = "nd4j.dsp.executionTiming";

    // ---- DSP Diagnostics Reporting ----

    /**
     * Enable DSP diagnostic categories. Comma-separated list of category names,
     * or "all" to enable everything, "none" to disable.
     * Categories: COMPILE, JIT, EXECUTE, TIMING, MEMORY, BACKEND, SHAPE,
     *             SEGMENT, FUSION, VERIFY, KV_CACHE, FALLBACK
     * Example: -Dnd4j.dsp.diagnostics=COMPILE,EXECUTE,TIMING
     */
    public static final String DSP_DIAGNOSTICS = "nd4j.dsp.diagnostics";

    /**
     * Set the diagnostic output detail level.
     * Values: summary (default), detailed (per-step), full (every event to stderr)
     * Example: -Dnd4j.dsp.diagnostics.level=detailed
     */
    public static final String DSP_DIAGNOSTICS_LEVEL = "nd4j.dsp.diagnostics.level";

    /**
     * Set the file path for JSON diagnostic report output.
     * When set, a JSON report is written to this path when the plan execution ends.
     * Example: -Dnd4j.dsp.diagnostics.file=/tmp/dsp-report.json
     */
    public static final String DSP_DIAGNOSTICS_FILE = "nd4j.dsp.diagnostics.file";

    // ---- Triton + CUDA Graph Integration ----

    /**
     * Applicability: Triton graph backend with CUDA graphs<br>
     * Description: When true, fallback executor (cuBLAS/native ops) is allowed during
     * CUDA graph capture. This records Triton fused kernels + cuBLAS matmuls + native
     * attention into a single CUDA graph for minimal kernel launch overhead.
     * Similar to how pytorch.compile captures entire subgraphs.
     * <p>
     * Default: true
     */
    public static final String TRITON_ALLOW_FALLBACK_CAPTURE = "nd4j.triton.allowFallbackCapture";

    /**
     * Applicability: Triton + CUDA graph integration<br>
     * Description: Enable CUDA graph capture of Triton execution. When disabled, Triton
     * kernels execute directly each step without capture/replay.
     * <p>
     * Default: true
     */
    public static final String TRITON_GRAPH_CAPTURE = "nd4j.triton.graphCapture";

    /**
     * Applicability: Triton debugging<br>
     * Description: Dump captured Triton CUDA graph to DOT file at /tmp/triton_graph_debug.dot
     * for visualization and debugging.
     * <p>
     * Default: false
     */
    public static final String TRITON_DUMP_GRAPH_DOT = "nd4j.triton.dumpGraphDot";

    /**
     * Applicability: Triton debugging<br>
     * Description: Skip Triton sub-kernel execution and run native slot-by-slot fallback instead.
     * Useful for isolating Triton kernel accuracy issues.
     * <p>
     * Default: false
     */
    public static final String TRITON_SKIP_KERNELS = "nd4j.triton.skipKernels";

    /**
     * Applicability: Triton debugging<br>
     * Description: Run both Triton and native execution for each sub-kernel and compare outputs.
     * Logs mismatches to help identify which specific kernel produces wrong results.
     * <p>
     * Default: false
     */
    public static final String TRITON_VERIFY_KERNELS = "nd4j.triton.verifyKernels";

    /**
     * Applicability: Triton compilation scope<br>
     * Description: When true, Triton compiles ALL section types (matmul, reduction, normalization,
     * attention, gather, etc.) instead of only elementwise/identity. Ops listed in
     * {@link #TRITON_EXCLUDE_OPS} still fall back to cuBLAS/native.
     * <p>
     * Default: false
     */
    public static final String TRITON_COMPILE_ALL = "nd4j.triton.compileAll";

    /**
     * Applicability: Triton compilation scope<br>
     * Description: Comma-separated list of nd4j op names to EXCLUDE from Triton compilation.
     * These ops fall back to cuBLAS/native even when tritonCompileAll=true.
     * Example: "matmul,mmul,tensormmul" keeps GEMMs on cuBLAS (usually faster).
     * <p>
     * Default: "" (empty = no exclusions)
     */
    public static final String TRITON_EXCLUDE_OPS = "nd4j.triton.excludeOps";

    /**
     * Applicability: DSP FusionPass<br>
     * Description: Enable cast elimination pass that removes redundant FP16↔FP32 cast pairs.
     * <p>
     * Default: false
     */
    public static final String DSP_CAST_ELIMINATION = "nd4j.dsp.castElimination";

    /**
     * Applicability: DSP frozen-shapes segment building<br>
     * Description: Break mega-segments at matmul/attention boundaries so element-wise
     * chains between matmuls get separate Triton fusion.
     * <p>
     * Default: false
     */
    public static final String DSP_MATMUL_SEGMENTATION = "nd4j.dsp.matmulSegmentation";

    /**
     * Applicability: CUDA matmul (MmulHelper)<br>
     * Description: Auto-cast FP32 matmul inputs to FP16 for TensorCore GEMM with FP32 accumulation.
     * Provides 2x throughput on GPUs with compute capability >= 6.0.
     * <p>
     * Default: false
     */
    public static final String DSP_FP16_COMPUTE = "nd4j.dsp.fp16Compute";

    /**
     * Applicability: CUDA graph capture<br>
     * Description: Replace per-slot memsets with a single batch-zero kernel during
     * CUDA graph capture. Reduces graph node count by ~800 nodes.
     * Environment variable: ND4J_DSP_BATCH_ZERO
     * <p>
     * Default: false
     */
    public static final String DSP_BATCH_ZERO = "nd4j.dsp.batchZero";

    /**
     * Applicability: CUDA graph capture<br>
     * Description: Log every buffer collected for batch-zero (very verbose).
     * Environment variable: ND4J_DSP_BATCH_ZERO_VERBOSE
     * <p>
     * Default: false
     */
    public static final String DSP_BATCH_ZERO_VERBOSE = "nd4j.dsp.batchZero.verbose";

    /**
     * Applicability: CUDA graph capture<br>
     * Description: When true (default), only zero gap (native fallback) slot outputs.
     * When false, zero ALL slot outputs including Triton sub-kernel outputs.
     * Environment variable: ND4J_DSP_BATCH_ZERO_GAP_ONLY
     * <p>
     * Default: true
     */
    public static final String DSP_BATCH_ZERO_GAP_ONLY = "nd4j.dsp.batchZero.gapOnly";

    /**
     * Applicability: CUDA graph capture<br>
     * Description: Use a single CUDA kernel to zero all buffers instead of N cudaMemsetAsync calls.
     * Reduces graph nodes by ~797 but may have memory ordering differences.<br>
     * <p>
     * Default: false
     */
    public static final String DSP_BATCH_ZERO_KERNEL = "nd4j.dsp.batchZero.kernel";

    /**
     * Applicability: CUDA graph capture<br>
     * Description: Group consecutive same-shape matmul slots into single cublasGemmBatchedEx
     * calls, reducing CUDA graph node count by ~96 nodes for typical transformer models.
     * Environment variable: ND4J_DSP_BATCHED_GEMM
     * <p>
     * Default: false
     */
    public static final String DSP_BATCHED_GEMM = "nd4j.dsp.batchedGemm";

    /**
     * Applicability: CUDA cuBLAS (sm_80+ Ampere and later)<br>
     * Description: Enable TF32 tensor core math mode for cuBLAS FP32 GEMMs.
     * Uses 10-bit mantissa precision for significant compute-bound speedup.
     * Environment variable: ND4J_CUBLAS_TF32
     * <p>
     * Default: false
     */
    public static final String CUBLAS_TF32 = "nd4j.cublas.tf32";

    /**
     * Applicability: DSP FusionPass<br>
     * Description: Sink FP16→FP32 cast ops through matmul boundaries.
     * Marks cast ops as identity when their only consumer is a matmul,
     * since MmulHelper handles mixed-precision internally.
     * Environment variable: ND4J_DSP_CAST_SINK_MATMUL
     * <p>
     * Default: false
     */
    public static final String DSP_CAST_SINK_MATMUL = "nd4j.dsp.castSinkMatmul";

    /**
     * Applicability: Triton GPU backend<br>
     * Description: Consolidate per-kernel arg table H2D copies into a single buffer and copy.
     * Reduces CUDA graph nodes by ~1165.
     * Environment variable: ND4J_TRITON_CONSOLIDATED_ARG_TABLE
     * <p>
     * Default: false
     */
    public static final String TRITON_CONSOLIDATED_ARG_TABLE = "nd4j.triton.consolidatedArgTable";

    /**
     * Applicability: Triton GPU backend<br>
     * Description: Skip arg table refresh for sub-kernels with only static (constant weight) args.
     * Environment variable: ND4J_TRITON_ARG_DIRTY_TRACKING
     * <p>
     * Default: false
     */
    public static final String TRITON_ARG_DIRTY_TRACKING = "nd4j.triton.argDirtyTracking";

    /**
     * Applicability: Triton GPU backend<br>
     * Description: Enable section fusion for Triton mega-kernel merging.
     * When enabled, non-elementwise section types (GATHER, CONCAT, CONST_GEN, SPLIT, STACK)
     * merge with adjacent sections into mega-kernels via buildSectionedModule() DAG analysis.
     * Environment variable: ND4J_TRITON_SECTION_FUSION
     * <p>
     * Default: false
     */
    public static final String TRITON_SECTION_FUSION = "nd4j.triton.sectionFusion";

    /**
     * Applicability: Triton GPU backend<br>
     * Description: Enable cost-model-based fusion scoring for section merge decisions.
     * When enabled, adjacent sections are only merged if the fusion score exceeds the minimum threshold.
     * Environment variable: ND4J_TRITON_FUSION_SCORING
     * <p>
     * Default: false
     */
    public static final String TRITON_FUSION_SCORING = "nd4j.triton.fusionScoring";

    /**
     * Applicability: Triton GPU backend<br>
     * Description: Minimum fusion score required to merge two adjacent sections.
     * Only used when fusion scoring is enabled.
     * Environment variable: ND4J_TRITON_FUSION_MIN_SCORE
     * <p>
     * Default: 1.0
     */
    public static final String TRITON_FUSION_MIN_SCORE = "nd4j.triton.fusionMinScore";

    /**
     * Applicability: DSP execution<br>
     * Description: Enable symbolic shape ranges to avoid recompilation when dimensions
     * change within observed bounds. Dynamic dimensions are hashed by rank/dtype only.
     * Environment variable: ND4J_DSP_SYMBOLIC_SHAPES
     * <p>
     * Default: false
     */
    public static final String DSP_SYMBOLIC_SHAPES = "nd4j.dsp.symbolicShapes";

    /**
     * Applicability: DSP execution<br>
     * Description: Number of observation steps before classifying dimensions as static or dynamic.
     * Environment variable: ND4J_DSP_SYMBOLIC_SHAPE_WARMUP
     * <p>
     * Default: 2
     */
    public static final String DSP_SYMBOLIC_SHAPE_WARMUP = "nd4j.dsp.symbolicShapeWarmup";

    /**
     * Applicability: DSP CUDA execution<br>
     * Description: Route capture buffer allocations through CudaMemoryPool for cross-segment reuse.
     * Environment variable: ND4J_DSP_CAPTURE_POOL_ENABLED
     * <p>
     * Default: false
     */
    public static final String DSP_CAPTURE_POOL_ENABLED = "nd4j.dsp.capturePoolEnabled";

    /**
     * Applicability: DSP CUDA execution<br>
     * Description: Maximum bytes for capture buffer pool allocations.
     * Environment variable: ND4J_DSP_CAPTURE_POOL_MAX_BYTES
     * <p>
     * Default: 1073741824 (1GB)
     */
    public static final String DSP_CAPTURE_POOL_MAX_BYTES = "nd4j.dsp.capturePoolMaxBytes";

    private ND4JSystemProperties() {
    }
}
