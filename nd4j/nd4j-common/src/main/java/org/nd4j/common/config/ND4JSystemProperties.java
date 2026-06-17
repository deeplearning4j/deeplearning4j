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
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: When set to "true", disables DSP freeze (compilation). Models will use
     * the standard InferenceSession execution path instead of the optimized DSP path.
     * Useful for debugging DSP-related issues.
     * <p>
     * Default: false (DSP enabled)
     */
    public static final String DSP_NO_FREEZE = "nd4j.dsp.noFreeze";

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
     * Description: DSP training is implicit — when DSP is built and available,
     * the training path (forward + backward + optimizer + weight-update) executes via
     * DynamicShapePlan automatically. No system property needed to enable it.
     *
     * @deprecated DSP training is always enabled when DSP is available. This constant
     *             is retained only for backward compatibility with tests that reference it.
     */
    @Deprecated
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
     * Description: Fraction of available device memory budgeted for the shape-keyed
     * plan cache. A cached plan counts against this budget by the bytes held in its
     * slot NDArrays. Above the budget, least-recently-used plans are evicted and
     * destroyed. Set to 0 to disable caching (every execute recompiles; never
     * recommended — the slot-immutability contract requires a matching cached plan).
     * Environment variable: ND4J_DSP_PLAN_CACHE_BUDGET_FRACTION
     * <p>
     * Default: 0.05 (5% of free device memory)
     */
    public static final String DSP_PLAN_CACHE_BUDGET_FRACTION = "org.nd4j.dsp.planCache.budgetFraction";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Hard cap on the number of plans cached per SameDiff instance.
     * Lower of {budget-fraction bytes} and {max plans} wins. Guards against
     * pathological models with thousands of distinct input-shape combinations.
     * Environment variable: ND4J_DSP_PLAN_CACHE_MAX_PLANS
     * <p>
     * Default: 64
     */
    public static final String DSP_PLAN_CACHE_MAX_PLANS = "org.nd4j.dsp.planCache.maxPlans";

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
     *         "MLX", "ARM_HYBRID", "NNAPI", "HIP_GRAPHS", "LEVEL_ZERO", "VULKAN", "METAL",
     *         "TPU", "HEXAGON", "OPENVINO", "TVM"
     */
    public static final String DSP_GRAPH_EXECUTION_MODE = "nd4j.dsp.graphExecutionMode";

    /**
     * Applicability: DSP execution engine / slot-by-slot lifecycle coordination<br>
     * Description: Controls how the slot-by-slot (legacy) execution path coordinates with
     * DSP capture/replay state. Gates per-buffer actuality ticks, host/device resync,
     * and deallocation against in-flight DSP plans.<br>
     * Values:
     * <ul>
     *   <li>"LEGACY_UNAWARE" (0) — bisecting regressions: slot-by-slot runs unchanged,
     *       no gating against DSP state. Use only to reproduce pre-fix behavior.</li>
     *   <li>"COEXIST_SAFE" (1, default) — slot-by-slot skips ticks, resyncs, and closes
     *       that would clobber DSP-owned buffers or corrupt in-flight capture/replay.</li>
     *   <li>"STRICT_ISOLATED" (2) — reserved for future use: disallow slot-by-slot
     *       entirely while DSP plan is active.</li>
     * </ul>
     */
    public static final String DSP_EXECUTION_MODE = "nd4j.dsp.executionMode";

    /**
     * Applicability: DSP execution engine<br>
     * Description: When true, merge value-dependent ops into capturable segments after
     * shapes freeze (SHAPES_FROZEN phase). Enables higher capture rate but may cause
     * issues with cross-device migration.
     * <p>
     * Default: false (C++ default may differ; this controls Java-side propagation)
     */
    public static final String DSP_FREEZE_MERGE_SEGMENTS = "nd4j.dsp.freezeMergeSegments";

    /**
     * Applicability: DSP execution engine<br>
     * Description: When true, recompile segments when shapes freeze.
     * <p>
     * Default: false
     */
    public static final String DSP_FREEZE_RECOMPILE = "nd4j.dsp.freezeRecompile";

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

    // ============== Triton Segment Fusion Flags (Temporary Testing) ==============

    /**
     * Applicability: Triton section identification<br>
     * Description: Fuse identity reshape/expand_dims/squeeze into element-wise sections.
     * Reduces section count by ~200-300 for typical decoder models.
     * <p>
     * Default: true
     */
    public static final String TRITON_FUSE_IDENTITY_SHAPES = "nd4j.triton.fuseIdentityShapes";

    /**
     * Applicability: Triton section identification<br>
     * Description: Fuse consecutive cast ops into single cast kernel.
     * Reduces section count by ~50-100 when cast chains present.
     * <p>
     * Default: true
     */
    public static final String TRITON_FUSE_CAST_CHAINS = "nd4j.triton.fuseCastChains";

    /**
     * Applicability: Triton section identification (seq=1 decode)<br>
     * Description: Treat identity permute patterns as no-op for seq=1 decode.
     * Reduces section count by ~60-100 for typical attention patterns.
     * <p>
     * Default: true
     */
    public static final String TRITON_SPECIALIZE_PERMUTE_SEQ1 = "nd4j.triton.specializePermuteSeq1";

    /**
     * Applicability: Triton section identification (HIGH RISK)<br>
     * Description: Fuse matmul→bias→activation patterns. Disabled by default due to accuracy risk.
     * Enable only for testing.
     * <p>
     * Default: false
     */
    public static final String TRITON_FUSED_MATMUL = "nd4j.triton.fusedMatmul";

    // ============== DSP Optimization Flags ==============

    /**
     * Applicability: DSP frozen-shapes segment building<br>
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
     * Applicability: Triton-compiled DotOps (sm_80+ Ampere and later)<br>
     * Description: Enable TF32 precision (10-bit mantissa) for Triton-compiled matmuls
     * and fused attention QK^T / PV dot products. Gives ~2x throughput but compounds
     * precision loss across thousands of ops per transformer decode step.
     * Environment variable: ND4J_TRITON_TF32
     * <p>
     * Default: false
     */
    public static final String TRITON_TF32 = "nd4j.triton.tf32";

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
     * Description: Enable Triton compile-range fusion and compatible post-merges.
     * When enabled, the compiler may coalesce adjacent Triton-compatible sections into
     * larger launch ranges and safely post-merge sections that share a compatible 1D skeleton.
     * Environment variable: ND4J_TRITON_SECTION_FUSION
     * <p>
     * Default: true
     */
    public static final String TRITON_SECTION_FUSION = "nd4j.triton.sectionFusion";

    /**
     * Applicability: Triton GPU backend<br>
     * Description: Enable cost-model-based fusion scoring for section merge decisions.
     * When enabled, adjacent sections are only merged if the fusion score exceeds the minimum threshold.
     * Environment variable: ND4J_TRITON_FUSION_SCORING
     * <p>
     * Default: true
     */
    public static final String TRITON_FUSION_SCORING = "nd4j.triton.fusionScoring";

    /**
     * Applicability: Triton GPU backend<br>
     * Description: Minimum fusion score required to merge two adjacent sections.
     * Only used when fusion scoring is enabled.
     * Environment variable: ND4J_TRITON_FUSION_MIN_SCORE
     * <p>
     * Default: 5.0
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

    // DSP_SYMBOLIC_SHAPE_WARMUP removed — warmup is a compile-time constant (2) in DspConfig::kSymbolicShapeWarmup.

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

    /**
     * Applicability: Multi-device placement<br>
     * Description: Enable automatic device placement planning for DSP execution.
     * When enabled, the planner assigns ops/weights across available devices.
     * Environment variable: ND4J_PLACEMENT_ENABLED
     * <p>
     * Default: false
     */
    public static final String PLACEMENT_ENABLED = "nd4j.placement.enabled";

    /**
     * Applicability: Multi-device placement<br>
     * Description: Device placement strategy. One of: SINGLE_DEVICE, MEMORY_FIT, PIPELINE_PARALLEL, CUSTOM.
     * Environment variable: ND4J_PLACEMENT_STRATEGY
     * <p>
     * Default: SINGLE_DEVICE
     */
    public static final String PLACEMENT_STRATEGY = "nd4j.placement.strategy";

    /**
     * Applicability: Multi-device placement<br>
     * Description: Default device ID for placement when using SINGLE_DEVICE strategy.
     * Environment variable: ND4J_PLACEMENT_DEFAULT_DEVICE
     * <p>
     * Default: 0
     */
    public static final String PLACEMENT_DEFAULT_DEVICE = "nd4j.placement.defaultDevice";

    /**
     * Applicability: Multi-device placement<br>
     * Description: Fraction of device memory to reserve as headroom (0.0-1.0).
     * The planner will not assign ops that would use more than (1 - headroom) of device memory.
     * Environment variable: ND4J_PLACEMENT_MEMORY_HEADROOM
     * <p>
     * Default: 0.1
     */
    public static final String PLACEMENT_MEMORY_HEADROOM = "nd4j.placement.memoryHeadroom";

    /**
     * Applicability: Multi-device placement<br>
     * Description: Log placement decisions for debugging.
     * Environment variable: ND4J_PLACEMENT_LOG_DECISIONS
     * <p>
     * Default: false
     */
    public static final String PLACEMENT_LOG_DECISIONS = "nd4j.placement.logDecisions";

    // ---- VLM/DSP Optimizer & Execution Flags ----

    /**
     * Applicability: SameDiff graph optimizer<br>
     * Description: Enable or disable the SameDiff graph optimizer.
     * <p>
     * Default: false
     */
    public static final String OPTIMIZER_ENABLED = "nd4j.optimizer.enabled";

    /**
     * Applicability: SameDiff graph optimizer<br>
     * Description: Enable FP16 optimizations in the graph optimizer.
     * <p>
     * Default: false
     */
    public static final String OPTIMIZER_FP16 = "nd4j.optimizer.fp16";

    /**
     * Applicability: SameDiff graph optimizer<br>
     * Description: Enable BF16 optimizations in the graph optimizer. Takes precedence over FP16 if both are set.
     * <p>
     * Default: false
     */
    public static final String OPTIMIZER_BF16 = "nd4j.optimizer.bf16";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: When true, disables attention override optimization in DSP execution.
     * <p>
     * Default: false
     */
    public static final String DSP_NO_ATTN_OVERRIDE = "nd4j.dsp.noAttnOverride";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: When true, disables direct buffer access in DSP execution.
     * <p>
     * Default: false
     */
    public static final String DSP_NO_DIRECT = "nd4j.dsp.noDirect";

    /**
     * Applicability: CUDA cuBLAS<br>
     * Description: Controls cuBLAS workspace capture behavior.
     * Set to "0" to disable workspace capture during CUDA graph recording.
     * <p>
     * Default: unset
     */
    public static final String CUBLAS_CAPTURE_WORKSPACE = "nd4j.cublas.captureWorkspace";

    /**
     * Applicability: CUDA DSP capture<br>
     * Description: Size in MB of the pre-allocated workspace used during CUDA graph capture.
     * Ops that allocate GPU memory during capture use this workspace instead of cudaMallocAsync
     * (which cannot be used during capture). If the workspace is too small, capture fails with
     * an OOM error. Increase this value for larger models.
     * <p>
     * Default: 256
     */
    public static final String DSP_CAPTURE_WORKSPACE_MB = "nd4j.dsp.captureWorkspaceMb";

    /**
     * Applicability: CUDA DSP capture<br>
     * Description: Size in MB of the host-side workspace used during CUDA graph capture
     * for pinned host memory allocations (e.g., shape info replication).
     * <p>
     * Default: 32
     */
    public static final String DSP_CAPTURE_HOST_WORKSPACE_MB = "nd4j.dsp.captureHostWorkspaceMb";

    /**
     * Applicability: SameDiff execution<br>
     * Description: Enable per-op timing instrumentation.
     * <p>
     * Default: false
     */
    public static final String OP_TIMING = "nd4j.op.timing";

    /**
     * Applicability: Framework diagnostics<br>
     * Description: Enable health monitoring for framework subsystems.
     * <p>
     * Default: false
     */
    public static final String HEALTH_MONITORING_ENABLE = "org.nd4j.framework.health.monitoring.enable";

    /**
     * Applicability: Framework diagnostics<br>
     * Description: Health check interval in milliseconds.
     * <p>
     * Default: 5000
     */
    public static final String HEALTH_CHECK_INTERVAL = "org.nd4j.framework.health.check.interval.ms";

    /**
     * Applicability: Profiling subsystem<br>
     * Description: Enable operation profiling.
     * <p>
     * Default: false
     */
    public static final String PROFILING_ENABLE = "org.nd4j.profiling.enable";

    /**
     * Applicability: Profiling subsystem<br>
     * Description: Profiling sample frequency in Hz.
     * <p>
     * Default: 1000
     */
    public static final String PROFILING_FREQUENCY = "org.nd4j.profiling.frequency";

    /**
     * Applicability: Workspace subsystem<br>
     * Description: Default workspace size in bytes.
     * <p>
     * Default: 134217728 (128 MB)
     */
    public static final String WORKSPACE_DEFAULT_SIZE = "org.nd4j.workspace.default.size";

    /**
     * Applicability: Workspace subsystem<br>
     * Description: Initial workspace size in bytes.
     * <p>
     * Default: 134217728 (128 MB)
     */
    public static final String WORKSPACE_INITIAL_SIZE = "org.nd4j.workspace.initial.size";

    /**
     * Applicability: Workspace subsystem<br>
     * Description: Workspace learning mode (NONE, AVERAGE, MAX).
     * <p>
     * Default: NONE
     */
    public static final String WORKSPACE_LEARNING_MODE = "org.nd4j.workspace.learning.mode";

    /**
     * Applicability: Workspace subsystem<br>
     * Description: Enable workspace debug mode.
     * <p>
     * Default: false
     */
    public static final String WORKSPACE_DEBUG_MODE = "org.nd4j.workspace.debug.mode";

    /**
     * Applicability: Execution subsystem<br>
     * Description: Enable kernel selection debug mode.
     * <p>
     * Default: false
     */
    public static final String KERNEL_SELECTION_DEBUG = "org.nd4j.execution.kernel.selection.debug";

    /**
     * Applicability: Execution subsystem<br>
     * Description: Preferred backend (native, cuda, etc.).
     * <p>
     * Default: auto
     */
    public static final String PREFERRED_BACKEND = "org.nd4j.execution.preferred.backend";

    /**
     * Applicability: Framework diagnostics<br>
     * Description: Enable diagnostics for framework subsystems.
     * <p>
     * Default: false
     */
    public static final String DIAGNOSTICS_ENABLE = "org.nd4j.framework.diagnostics.enable";

    /**
     * Applicability: Framework diagnostics<br>
     * Description: Diagnostics level (INFO, DEBUG, VERBOSE).
     * <p>
     * Default: INFO
     */
    public static final String DIAGNOSTICS_LEVEL = "org.nd4j.framework.diagnostics.level";

    /**
     * Applicability: Framework diagnostics<br>
     * Description: Enable verbose diagnostics output.
     * <p>
     * Default: false
     */
    public static final String DIAGNOSTICS_VERBOSE = "org.nd4j.framework.diagnostics.verbose";

    /**
     * Applicability: Framework diagnostics<br>
     * Description: Diagnostics output file path.
     * <p>
     * Default: stderr
     */
    public static final String DIAGNOSTICS_OUTPUT_FILE = "org.nd4j.framework.diagnostics.output.file";

    /**
     * Applicability: Function tracing<br>
     * Description: Enable function tracing.
     * <p>
     * Default: false
     */
    public static final String FUNCTRACE_ENABLE = "org.nd4j.functrace.enable";

    /**
     * Applicability: Function tracing<br>
     * Description: Print allocation events.
     * <p>
     * Default: false
     */
    public static final String FUNCTRACE_PRINT_ALLOCATE = "org.nd4j.functrace.print.allocate";

    /**
     * Applicability: Function tracing<br>
     * Description: Print free events.
     * <p>
     * Default: false
     */
    public static final String FUNCTRACE_PRINT_FREE = "org.nd4j.functrace.print.free";

    /**
     * Applicability: Memory environment<br>
     * Description: Workspace metadata size.
     * <p>
     * Default: 1024
     */
    public static final String WORKSPACE_META_SIZE = "org.nd4j.workspace.meta.size";

    /**
     * Applicability: Memory environment<br>
     * Description: Heap pressure threshold (0.0-1.0).
     * <p>
     * Default: 0.9
     */
    public static final String HEAP_PRESSURE_THRESHOLD = "org.nd4j.memory.heap.pressure.threshold";

    /**
     * Applicability: Profiling environment<br>
     * Description: Enable verbose profiling.
     * <p>
     * Default: false
     */
    public static final String PROFILING_VERBOSE = "org.nd4j.profiling.verbose";

    /**
     * Applicability: Profiling environment<br>
     * Description: Profiling output file path.
     * <p>
     * Default: stdout
     */
    public static final String PROFILING_OUTPUT_FILE = "org.nd4j.profiling.output.file";

    /**
     * Applicability: Profiling environment<br>
     * Description: Enable bandwidth profiling.
     * <p>
     * Default: false
     */
    public static final String PROFILING_BANDWIDTH = "org.nd4j.profiling.bandwidth";

    /**
     * Applicability: Workspace environment<br>
     * Description: Maximum workspace size.
     * <p>
     * Default: 2147483648 (2 GB)
     */
    public static final String WORKSPACE_MAX_SIZE = "org.nd4j.workspace.max.size";

    /**
     * Applicability: Workspace environment<br>
     * Description: Enable workspace preallocation.
     * <p>
     * Default: true
     */
    public static final String WORKSPACE_PREALLOCATE = "org.nd4j.workspace.preallocate";

    /**
     * Applicability: Workspace environment<br>
     * Description: Workspace overallocation limit.
     * <p>
     * Default: 0.5
     */
    public static final String WORKSPACE_OVERALLOCATION_LIMIT = "org.nd4j.workspace.overallocation.limit";

    /**
     * Applicability: Device environment<br>
     * Description: Enable CUDA context per thread.
     * <p>
     * Default: false
     */
    public static final String CUDA_CONTEXT_PER_THREAD = "org.nd4j.cuda.context.per.thread";

    /**
     * Applicability: Device environment<br>
     * Description: Use CUDA stream memory.
     * <p>
     * Default: true
     */
    public static final String CUDA_USE_STREAM_MEMORY = "org.nd4j.cuda.use.stream.memory";

    /**
     * Applicability: Lifecycle tracking<br>
     * Description: Enable lifecycle tracking.
     * <p>
     * Default: false
     */
    public static final String LIFECYCLE_TRACKING_ENABLE = "org.nd4j.lifecycle.tracking.enable";

    /**
     * Applicability: Lifecycle tracking<br>
     * Description: Enable stack trace capture.
     * <p>
     * Default: false
     */
    public static final String LIFECYCLE_STACK_TRACE_CAPTURE = "org.nd4j.lifecycle.stacktrace.capture";

    /**
     * Applicability: Lifecycle tracking<br>
     * Description: Event retention count.
     * <p>
     * Default: 10000
     */
    public static final String LIFECYCLE_EVENT_RETENTION = "org.nd4j.lifecycle.event.retention";

    /**
     * Applicability: Leak detection<br>
     * Description: Enable leak detection.
     * <p>
     * Default: false
     */
    public static final String LEAK_DETECTION_ENABLE = "org.nd4j.leak.detection.enable";

    /**
     * Applicability: Leak detection<br>
     * Description: Age threshold in milliseconds.
     * <p>
     * Default: 60000
     */
    public static final String LEAK_DETECTION_AGE_THRESHOLD = "org.nd4j.leak.detection.age.threshold";

    /**
     * Applicability: Leak detection<br>
     * Description: Size threshold in bytes.
     * <p>
     * Default: 104857600 (100 MB)
     */
    public static final String LEAK_DETECTION_SIZE_THRESHOLD = "org.nd4j.leak.detection.size.threshold";

    // ==== Device Transfer Management Framework ====

    /**
     * Applicability: Device transfer management<br>
     * Description: Enable device transfer tracking and diagnostics.
     * When enabled, all H2D, D2H, and D2D transfers are recorded with timing and byte counts.
     * Access via Nd4j.framework.device().transfers().
     * <p>
     * Default: false (disabled for zero overhead)
     */
    public static final String DEVICE_TRANSFER_TRACKING = "nd4j.device.transfer.tracking";

    /**
     * Applicability: Device transfer management<br>
     * Description: Enable per-variable device pinning.
     * Allows pinning variables to specific devices or policies (STICKY, FOLLOW_THREAD, EXPLICIT).
     * Access via Nd4j.framework.device().pinning().
     * <p>
     * Default: false
     */
    public static final String DEVICE_PINNING_ENABLED = "nd4j.device.pinning.enabled";

    /**
     * Applicability: Device transfer management<br>
     * Description: Enable replica leak detection.
     * Tracks replicated arrays across devices and detects leaks when replicas are not properly cleaned up.
     * Access via Nd4j.framework.device().replicaLeaks().
     * <p>
     * Default: false
     */
    public static final String DEVICE_REPLICA_LEAK_DETECTION = "nd4j.device.replica.leak.detection";

    /**
     * Applicability: Device transfer management<br>
     * Description: Enable pointer stability validation for CUDA graph replay.
     * Validates that GPU buffer addresses remain stable across graph capture and replay.
     * Access via Nd4j.framework.device().pointerStability().
     * <p>
     * Default: false
     */
    public static final String DEVICE_POINTER_STABILITY_CHECK = "nd4j.device.pointerStability.check";

    // ========================================================================
    // Environment forwarding properties (orchestrator → subprocess pattern).
    // These are set as system properties by an orchestrator process that reads
    // persisted ND4J config JSON but does NOT initialize the ND4J backend.
    // Subprocesses read these on startup to apply environment settings.
    // ========================================================================

    // --- BLAS / Threading ---
    public static final String ENV_ENABLE_BLAS = "nd4j.environment.enableBlas";
    public static final String ENV_HELPERS_ALLOWED = "nd4j.environment.helpersAllowed";
    public static final String ENV_MAX_THREADS = "nd4j.environment.maxThreads";
    public static final String ENV_MAX_MASTER_THREADS = "nd4j.environment.maxMasterThreads";
    public static final String ENV_OMP_NUM_THREADS = "nd4j.environment.ompNumThreads";

    // --- Debug / Profiling ---
    public static final String ENV_DEBUG = "nd4j.environment.debug";
    public static final String ENV_VERBOSE = "nd4j.environment.verbose";
    public static final String ENV_PROFILING = "nd4j.environment.profiling";
    public static final String ENV_DETECTING_LEAKS = "nd4j.environment.detectingLeaks";

    // --- Parallelism thresholds ---
    public static final String ENV_TAD_THRESHOLD = "nd4j.environment.tadThreshold";
    public static final String ENV_ELEMENTWISE_THRESHOLD = "nd4j.environment.elementwiseThreshold";

    // --- Memory limits ---
    public static final String ENV_MAX_PRIMARY_MEMORY = "nd4j.environment.maxPrimaryMemory";
    public static final String ENV_MAX_SPECIAL_MEMORY = "nd4j.environment.maxSpecialMemory";
    public static final String ENV_MAX_DEVICE_MEMORY = "nd4j.environment.maxDeviceMemory";

    // --- Lifecycle tracking ---
    public static final String ENV_LIFECYCLE_TRACKING = "nd4j.environment.lifecycleTracking";
    public static final String ENV_TRACK_VIEWS = "nd4j.environment.trackViews";
    public static final String ENV_TRACK_DELETIONS = "nd4j.environment.trackDeletions";
    public static final String ENV_SNAPSHOT_FILES = "nd4j.environment.snapshotFiles";
    public static final String ENV_TRACK_OPERATIONS = "nd4j.environment.trackOperations";
    public static final String ENV_STACK_DEPTH = "nd4j.environment.stackDepth";
    public static final String ENV_REPORT_INTERVAL = "nd4j.environment.reportInterval";
    public static final String ENV_MAX_DELETION_HISTORY = "nd4j.environment.maxDeletionHistory";

    // --- Subsystem tracking ---
    public static final String ENV_NDARRAY_TRACKING = "nd4j.environment.ndArrayTracking";
    public static final String ENV_DATA_BUFFER_TRACKING = "nd4j.environment.dataBufferTracking";
    public static final String ENV_TAD_CACHE_TRACKING = "nd4j.environment.tadCacheTracking";
    public static final String ENV_SHAPE_CACHE_TRACKING = "nd4j.environment.shapeCacheTracking";
    public static final String ENV_OP_CONTEXT_TRACKING = "nd4j.environment.opContextTracking";

    // --- Function trace ---
    public static final String ENV_FUNC_TRACE_PRINT_ALLOCATE = "nd4j.environment.funcTracePrintAllocate";
    public static final String ENV_FUNC_TRACE_PRINT_DEALLOCATE = "nd4j.environment.funcTracePrintDeallocate";
    public static final String ENV_FUNC_TRACE_PRINT_JAVA_ONLY = "nd4j.environment.funcTracePrintJavaOnly";

    // --- Logging ---
    public static final String ENV_LOG_NATIVE_NDARRAY_CREATION = "nd4j.environment.logNativeNDArrayCreation";
    public static final String ENV_LOG_NDARRAY_EVENTS = "nd4j.environment.logNDArrayEvents";
    public static final String ENV_TRUNCATE_NDARRAY_LOG_STRINGS = "nd4j.environment.truncateNDArrayLogStrings";
    public static final String ENV_CHECK_INPUT_CHANGE = "nd4j.environment.checkInputChange";
    public static final String ENV_CHECK_OUTPUT_CHANGE = "nd4j.environment.checkOutputChange";
    public static final String ENV_TRACK_WORKSPACE_OPEN_CLOSE = "nd4j.environment.trackWorkspaceOpenClose";
    public static final String ENV_DELETE_SHAPE_INFO = "nd4j.environment.deleteShapeInfo";
    public static final String ENV_DELETE_PRIMARY = "nd4j.environment.deletePrimary";
    public static final String ENV_DELETE_SPECIAL = "nd4j.environment.deleteSpecial";
    public static final String ENV_VARIABLE_TRACING_ENABLED = "nd4j.environment.variableTracingEnabled";

    // --- BLAS serialization ---
    public static final String ENV_BLAS_SERIALIZATION_ENABLED = "nd4j.environment.blasSerializationEnabled";
    public static final String ENV_OPENBLAS_THREADS = "nd4j.environment.openBlasThreads";

    // --- CUDA device settings ---
    public static final String ENV_CUDA_CURRENT_DEVICE = "nd4j.environment.cudaCurrentDevice";
    public static final String ENV_CUDA_MEMORY_PINNED = "nd4j.environment.cudaMemoryPinned";
    public static final String ENV_CUDA_USE_MANAGED_MEMORY = "nd4j.environment.cudaUseManagedMemory";
    public static final String ENV_CUDA_MEMORY_POOL_SIZE = "nd4j.environment.cudaMemoryPoolSize";
    public static final String ENV_CUDA_FORCE_P2P = "nd4j.environment.cudaForceP2P";
    public static final String ENV_CUDA_ALLOCATOR_ENABLED = "nd4j.environment.cudaAllocatorEnabled";
    public static final String ENV_CUDA_MAX_BLOCKS = "nd4j.environment.cudaMaxBlocks";
    public static final String ENV_CUDA_MAX_THREADS_PER_BLOCK = "nd4j.environment.cudaMaxThreadsPerBlock";
    public static final String ENV_CUDA_ASYNC_EXECUTION = "nd4j.environment.cudaAsyncExecution";
    public static final String ENV_CUDA_STREAM_LIMIT = "nd4j.environment.cudaStreamLimit";
    public static final String ENV_CUDA_USE_DEVICE_HOST = "nd4j.environment.cudaUseDeviceHost";
    public static final String ENV_CUDA_EVENT_LIMIT = "nd4j.environment.cudaEventLimit";
    public static final String ENV_CUDA_CACHING_ALLOCATOR_LIMIT = "nd4j.environment.cudaCachingAllocatorLimit";
    public static final String ENV_CUDA_USE_UNIFIED_MEMORY = "nd4j.environment.cudaUseUnifiedMemory";
    public static final String ENV_CUDA_PREFETCH_SIZE = "nd4j.environment.cudaPrefetchSize";
    public static final String ENV_CUDA_GRAPH_OPTIMIZATION = "nd4j.environment.cudaGraphOptimization";
    public static final String ENV_CUDA_TENSOR_CORE_ENABLED = "nd4j.environment.cudaTensorCoreEnabled";
    public static final String ENV_CUDA_BLOCKING_SYNC = "nd4j.environment.cudaBlockingSync";
    public static final String ENV_CUDA_DEVICE_SCHEDULE = "nd4j.environment.cudaDeviceSchedule";
    public static final String ENV_CUDA_STACK_SIZE = "nd4j.environment.cudaStackSize";
    public static final String ENV_CUDA_MALLOC_HEAP_SIZE = "nd4j.environment.cudaMallocHeapSize";
    public static final String ENV_CUDA_PRINTF_FIFO_SIZE = "nd4j.environment.cudaPrintfFifoSize";
    public static final String ENV_CUDA_DEV_RUNTIME_SYNC_DEPTH = "nd4j.environment.cudaDevRuntimeSyncDepth";
    public static final String ENV_CUDA_DEV_RUNTIME_PENDING_LAUNCH_COUNT = "nd4j.environment.cudaDevRuntimePendingLaunchCount";
    public static final String ENV_CUDA_MAX_L2_FETCH_GRANULARITY = "nd4j.environment.cudaMaxL2FetchGranularity";
    public static final String ENV_CUDA_PERSISTING_L2_CACHE_SIZE = "nd4j.environment.cudaPersistingL2CacheSize";

    // --- DSP plan disk cache ---

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Enable or disable the DSP plan disk cache. When enabled,
     * serialized plan bytes are persisted to disk and reloaded on subsequent
     * JVM starts, avoiding recompilation for unchanged graph structures.
     * Environment variable: ND4J_DSP_PLAN_CACHE_DISK_ENABLED
     * <p>
     * Default: true
     */
    public static final String DSP_PLAN_CACHE_DISK_ENABLED = "nd4j.dsp.planCache.diskEnabled";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Override directory for the DSP plan disk cache.
     * When not set, defaults to ~/.kompile/cache/dsp/dsp_plan_cache/.
     * Environment variable: ND4J_DSP_PLAN_CACHE_DIR
     */
    public static final String DSP_PLAN_CACHE_DISK_DIR = "nd4j.dsp.planCache.diskDir";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: Override directory for pre-seeded DSP plans (highest priority lookup).
     * Plans placed here are never overwritten by the cache writer.
     * Environment variable: ND4J_DSP_PLAN_CACHE_OVERRIDE_DIR
     */
    public static final String DSP_PLAN_CACHE_OVERRIDE_DIR = "nd4j.dsp.planCache.overrideDir";

    /**
     * Applicability: DynamicShapePlan-based inference<br>
     * Description: When true, ignore existing cached plans and force recompilation.
     * New plans are still written to the cache. Useful for rebuilding after
     * a suspected cache corruption.
     * Environment variable: ND4J_DSP_PLAN_CACHE_FORCE_RECOMPILE
     * <p>
     * Default: false
     */
    public static final String DSP_PLAN_CACHE_FORCE_RECOMPILE = "nd4j.dsp.planCache.forceRecompile";

    // --- Triton environment ---
    public static final String ENV_TRITON_BUILD_THREADS = "nd4j.environment.tritonBuildThreads";
    public static final String ENV_TRITON_CACHE_ENABLED = "nd4j.environment.tritonCacheEnabled";
    public static final String ENV_TRITON_VERBOSE = "nd4j.environment.tritonVerbose";
    public static final String ENV_TRITON_ALWAYS_COMPILE = "nd4j.environment.tritonAlwaysCompile";
    public static final String ENV_TRITON_NUM_WARPS = "nd4j.environment.tritonNumWarps";
    public static final String ENV_TRITON_NUM_STAGES = "nd4j.environment.tritonNumStages";
    public static final String ENV_TRITON_NUM_CTAS = "nd4j.environment.tritonNumCTAs";
    public static final String ENV_TRITON_ENABLE_FP_FUSION = "nd4j.environment.tritonEnableFpFusion";
    public static final String ENV_TRITON_CACHE_DIR = "nd4j.environment.tritonCacheDir";
    public static final String ENV_TRITON_DUMP_DIR = "nd4j.environment.tritonDumpDir";
    public static final String ENV_TRITON_OVERRIDE_ARCH = "nd4j.environment.tritonOverrideArch";

    // ---- Array cache properties ----

    /**
     * Applicability: ArrayCacheMemoryMgr<br>
     * Description: Growth factor for over-allocation on cache miss. Buffers are allocated
     * with this multiplier so that the next step's slightly larger request (e.g. growing KV cache)
     * can reuse the buffer via capacity matching.
     * Set to 1.0 to disable over-allocation on memory-constrained systems.
     * <p>
     * Default: 1.05
     */
    public static final String CACHE_GROWTH_FACTOR = "org.nd4j.cache.growthFactor";

    // ---- Graph optimizer properties ----

    /**
     * Applicability: SameDiff GraphOptimizer<br>
     * Description: Comma-separated list of optimizer class simple names to skip during graph optimization.
     * For example: {@code -Dnd4j.optimizer.skip=NormalizationFusionOptimizations,QuantizationOptimizations}
     * <p>
     * Default: "" (empty - no optimizers are skipped)
     */
    public static final String OPTIMIZER_SKIP = "nd4j.optimizer.skip";

    /**
     * Applicability: SameDiff GraphOptimizer<br>
     * Description: Maximum number of optimization iterations to run on a graph.
     * Increasing this value allows more aggressive optimization at the cost of compile time.
     * <p>
     * Default: 3
     */
    public static final String OPTIMIZER_MAX_ITERATIONS = "nd4j.optimizer.maxIterations";

    /**
     * Applicability: SameDiff GraphOptimizer<br>
     * Description: When set to "true", logs each optimization pass that is applied to the graph.
     * Useful for debugging optimizer behavior.
     * <p>
     * Default: false
     */
    public static final String OPTIMIZER_LOG_APPLIED = "nd4j.optimizer.logApplied";

    // ---- SDZ (SameDiff ZIP archive) properties ----

    /**
     * Applicability: SDZSerializer<br>
     * Description: Maximum total decompressed size in bytes allowed when extracting SDZ ZIP archives.
     * This limit protects against zip bomb attacks that could exhaust disk space.
     * <p>
     * Default: 10737418240 (10 GB)
     */
    public static final String SDZ_MAX_ZIP_SIZE = "nd4j.sdz.maxZipSize";

    /**
     * Applicability: SDZSerializer<br>
     * Description: Maximum allowed compression ratio (uncompressed / compressed size) for SDZ ZIP entries.
     * Protects against zip bomb attacks with highly compressed data.
     * <p>
     * Default: 100.0
     */
    public static final String SDZ_MAX_ZIP_SIZE_RATIO = "nd4j.sdz.maxCompressionRatio";

    /**
     * Applicability: SDZSerializer<br>
     * Description: Maximum number of entries allowed in a model SDZ ZIP archive.
     * Protects against zip-based denial-of-service attacks that create many entries.
     * <p>
     * Default: 1000
     */
    public static final String SDZ_MAX_ZIP_ENTRIES = "nd4j.sdz.maxZipEntries";

    // ---- CPU device descriptor properties ----

    /**
     * Applicability: CpuDeviceDescriptor (nd4j-native backend)<br>
     * Description: Override for CPU AVX capability detection. When set, the value is used
     * instead of auto-detecting AVX support from the CPU. Also checks environment variable ND4J_CPU_AVX.
     * <p>
     * Default: auto-detected
     */
    public static final String CPU_AVX = "nd4j.cpu.avx";

    /**
     * Applicability: CpuDeviceDescriptor (nd4j-native backend)<br>
     * Description: Override for CPU SVE (Scalable Vector Extension) capability detection.
     * When set, the value is used instead of auto-detecting SVE support from the CPU.
     * Also checks environment variable ND4J_CPU_SVE.
     * <p>
     * Default: auto-detected
     */
    public static final String CPU_SVE = "nd4j.cpu.sve";

    // ---- Memory configuration properties ----

    /**
     * Applicability: MemoryConfig<br>
     * Description: Percentage of total device memory the CUDA memory pool is allowed to keep reserved
     * during pool release. Range: 1-100. Also checks environment variable SD_POOL_RELEASE_THRESHOLD_PERCENT.
     * <p>
     * Default: 75
     */
    public static final String MEMORY_POOL_RELEASE_THRESHOLD_PERCENT = "nd4j.memory.poolReleaseThresholdPercent";

    /**
     * Applicability: MemoryConfig<br>
     * Description: Minimum percentage of total GPU memory that must remain free AFTER a managed-memory
     * allocation for it to use the fast cudaMallocManaged path on non-peer devices. Range: 1-100.
     * Also checks environment variable SD_NON_PEER_HEADROOM_PERCENT.
     * <p>
     * Default: 50
     */
    public static final String MEMORY_NON_PEER_HEADROOM_PERCENT = "nd4j.memory.nonPeerHeadroomPercent";

    /**
     * Applicability: MemoryEnvironmentAccess<br>
     * Description: Default fraction of device memory to use for ND4J operations.
     * Accepts a float value in the range [0.0, 1.0].
     * <p>
     * Default: 0.9
     */
    public static final String MEMORY_FRACTION = "org.nd4j.memory.fraction";

    /**
     * Applicability: OpaqueDataBuffer<br>
     * Description: When set to "false", disables CPU fallback when CUDA/GPU allocation fails.
     * By default, if GPU allocation fails, ND4J will fall back to allocating on CPU host memory.
     * <p>
     * Default: true
     */
    public static final String MEMORY_FALLBACK_ENABLED = "nd4j.memory.fallback.enabled";

    /**
     * Applicability: CUDA backend (CudaMemoryManager)<br>
     * Description: When set to "false", disables CPU host-memory fallback when CUDA device allocation
     * fails. By default, failed CUDA allocations fall back to host memory so execution can continue.
     * <p>
     * Default: true
     */
    public static final String CUDA_MEMORY_FALLBACK_ENABLED = "nd4j.cuda.memory.fallback.enabled";

    // ---- Opaque native buffer diagnostics ----

    /**
     * Applicability: OpaqueNDArray<br>
     * Description: When set to "true", enables Java stack trace capture at OpaqueNDArray allocation time.
     * Useful for diagnosing native memory leaks. Stack trace capture is expensive and should only be
     * enabled for debugging.
     * <p>
     * Default: false
     */
    public static final String OPAQUE_STACKTRACE = "nd4j.opaque.stacktrace";

    // ---- Multi-GPU debug ----

    /**
     * Applicability: Multi-GPU execution (MultiGpuTracer)<br>
     * Description: When set to "true", enables detailed trace logging for multi-GPU device
     * decisions, data transfers, stream usage, and buffer lifecycle. All output is logged at
     * debug level with the [MultiGpu] prefix. When disabled (default), all trace methods are
     * no-ops with zero overhead.
     * <p>
     * Default: false
     */
    public static final String MULTI_GPU_DEBUG = "org.nd4j.multiGpu.debug";

    // ---- BackendManager configuration ----

    /**
     * Applicability: BackendManager<br>
     * Description: Comma-separated list of device types in priority order for backend selection.
     * Supported values: CUDA_GPU, ROCM_GPU, METAL_GPU, TPU, CPU (and aliases cuda, gpu, rocm, cpu).
     * Example: {@code -Dnd4j.backend.priority=CUDA_GPU,CPU}
     * <p>
     * Default: CUDA_GPU, ROCM_GPU, METAL_GPU, TPU, CPU
     */
    public static final String BACKEND_PRIORITY = "nd4j.backend.priority";

    /**
     * Applicability: BackendManager<br>
     * Description: When set to "true", enables automatic memory fallback from GPU to CPU
     * when GPU memory is exhausted.
     * <p>
     * Default: true
     */
    public static final String BACKEND_MEMORY_FALLBACK = "nd4j.backend.memory.fallback";

    /**
     * Applicability: BackendManager<br>
     * Description: Fraction of GPU memory to use (0.0-1.0).
     * <p>
     * Default: 0.9
     */
    public static final String BACKEND_GPU_MEMORY_FRACTION = "nd4j.backend.gpu.memory.fraction";

    /**
     * Applicability: BackendManager<br>
     * Description: Fraction of CPU memory to use (0.0-1.0).
     * <p>
     * Default: 0.8
     */
    public static final String BACKEND_CPU_MEMORY_FRACTION = "nd4j.backend.cpu.memory.fraction";

    /**
     * Applicability: BackendManager<br>
     * Description: When set to "true", enables automatic cross-device data transfer when
     * an operation requires data on a device where it is not currently resident.
     * <p>
     * Default: true
     */
    public static final String BACKEND_AUTO_TRANSFER = "nd4j.backend.auto.transfer";

    /**
     * Applicability: BackendManager<br>
     * Description: When set to "true", enables automatic initialization of BackendManager
     * during Nd4j startup. Set to "false" to defer or suppress auto-initialization.
     * <p>
     * Default: true
     */
    public static final String BACKEND_AUTO_INIT = "nd4j.backend.auto.init";

    // ---- Parallel executor service ----

    /**
     * Applicability: ExecutorServiceProvider<br>
     * Description: Number of threads in the global ND4J parallel executor service.
     * Default is the number of available processors.
     * <p>
     * Default: Runtime.getRuntime().availableProcessors()
     */
    public static final String PARALLEL_THREADS = "org.nd4j.parallel.threads";

    /**
     * Applicability: ExecutorServiceProvider<br>
     * Description: When set to "false", disables the global ND4J parallel executor service
     * and forces single-threaded execution (nThreads = 1).
     * <p>
     * Default: true
     */
    public static final String PARALLEL_ENABLED = "org.nd4j.parallel.enabled";

    // ---- Native plugin loading ----

    /**
     * Applicability: NativePluginLoader<br>
     * Description: Filesystem path from which to load native ND4J plugin libraries (.so/.dll/.dylib).
     * When not set, the loader falls back to the default search path under the user home directory.
     * <p>
     * Default: unset
     */
    public static final String NATIVE_PLUGIN_PATH = "nd4j.native.plugin.path";

    // ---- FileBatch ZIP security properties ----

    /**
     * Applicability: FileBatch ZIP reader<br>
     * Description: Maximum total decompressed size in bytes allowed when reading FileBatch ZIP files.
     * Protects against zip bomb attacks that could exhaust memory or disk space.
     * <p>
     * Default: 1073741824 (1 GB)
     */
    public static final String ND4J_FILEBATCH_MAX_ZIP_SIZE = "nd4j.filebatch.maxZipSize";

    /**
     * Applicability: FileBatch ZIP reader<br>
     * Description: Maximum allowed compression ratio (uncompressed / compressed size) for FileBatch ZIP entries.
     * Protects against zip bomb attacks with highly compressed data.
     * <p>
     * Default: 100.0
     */
    public static final String ND4J_FILEBATCH_MAX_COMPRESSION_RATIO = "nd4j.filebatch.maxCompressionRatio";

    /**
     * Applicability: FileBatch ZIP reader<br>
     * Description: Maximum number of entries allowed in a FileBatch ZIP archive.
     * Protects against zip-based denial-of-service attacks that create many entries.
     * <p>
     * Default: 10000
     */
    public static final String ND4J_FILEBATCH_MAX_ZIP_ENTRIES = "nd4j.filebatch.maxZipEntries";

    // ---- ArchiveUtils ZIP/TAR security properties ----

    /**
     * Applicability: ArchiveUtils (ZIP/TAR extraction)<br>
     * Description: Maximum total decompressed size in bytes allowed when extracting archives.
     * Protects against zip bomb attacks that could exhaust disk space.
     * <p>
     * Default: 10737418240 (10 GB)
     */
    public static final String ND4J_ARCHIVE_MAX_UNCOMPRESSED_SIZE = "nd4j.archive.maxUncompressedSize";

    /**
     * Applicability: ArchiveUtils (ZIP/TAR extraction)<br>
     * Description: Maximum allowed compression ratio (uncompressed / compressed size) for archive entries.
     * Protects against zip bomb attacks with highly compressed data.
     * <p>
     * Default: 100.0
     */
    public static final String ND4J_ARCHIVE_MAX_COMPRESSION_RATIO = "nd4j.archive.maxCompressionRatio";

    /**
     * Applicability: ArchiveUtils (ZIP/TAR extraction)<br>
     * Description: Maximum number of entries allowed in an archive.
     * Protects against zip-based denial-of-service attacks that create many entries.
     * <p>
     * Default: 100000
     */
    public static final String ND4J_ARCHIVE_MAX_ENTRIES = "nd4j.archive.maxEntries";

    // ---- Frozen decode step diagnostics ----

    /**
     * Applicability: FrozenDecodeStep (speculative decoding)<br>
     * Description: When set to "true", dumps the graph summary to the log after
     * FrozenDecodeStep compilation. Useful for inspecting graph structure for
     * seqLen&gt;1 edge cases.
     * <p>
     * Default: false
     */
    public static final String ND4J_FROZEN_SUMMARY = "nd4j.frozen.summary";

    /**
     * Applicability: FrozenDecodeStep (speculative decoding)<br>
     * Description: When set to "true", enables debug+verbose ND4J environment logging
     * during the first (warmup) execution of a FrozenDecodeStep. Useful for tracing
     * all op shapes during warmup.
     * <p>
     * Default: false
     */
    public static final String ND4J_FROZEN_DEBUG = "nd4j.frozen.debug";

    // ---- Decode loop diagnostics ----

    /**
     * Applicability: StaticKvCacheDecodeLoop<br>
     * Description: When set to "true", enables per-step GPU free-memory diagnostics
     * during autoregressive decoding. Logs free memory and delta for the first 10 steps.
     * <p>
     * Default: false
     */
    public static final String ND4J_DECODE_MEMORY_DIAG = "nd4j.decode.memoryDiag";

    /**
     * Applicability: StaticKvCacheDecodeLoop / DSP padded-shape decode path<br>
     * Description: When set to "true", disables the DSP padded-shape (static KV) decode
     * path. Forces the fallback dynamic-shape path for each decode step.
     * <p>
     * Default: false
     */
    public static final String DSP_NO_PADDED = "nd4j.dsp.noPadded";

    // ---- VLM image preprocessing ----

    /**
     * Applicability: ImageTiler (VLM image preprocessing)<br>
     * Description: When set to "true", applies a mild sharpening kernel to resized
     * images before tiling. Helps preserve character edge clarity for OCR-heavy VLM tasks.
     * <p>
     * Default: false
     */
    public static final String VLM_IMAGE_SHARPEN = "nd4j.vlm.image.sharpen";

    private ND4JSystemProperties() {
    }
}
