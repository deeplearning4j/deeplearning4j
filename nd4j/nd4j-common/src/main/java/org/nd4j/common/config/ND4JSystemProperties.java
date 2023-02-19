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
    public final static String SAMEDIFF_MEMORY_CACHE_ENABLE = "org.nd4j.autodiff.samediff.cache";

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

    private ND4JSystemProperties() {
    }
}
