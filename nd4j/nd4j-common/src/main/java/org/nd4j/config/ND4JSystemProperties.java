/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.config;

public class ND4JSystemProperties {

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
    public static final String ND4J_CPU_LOAD_OPENBLAS = "org.bytedeco.javacpp.openblas.load";
    /**
     * Applicability: nd4j-native backend, when multiple BLAS libraries are available<br>
     * Description: This system property can be used to control which BLAS library is loaded and used by ND4J.
     * Similar to {@link #ND4J_CPU_LOAD_OPENBLAS} but when this is set, LAPACK will not be loaded
     */
    public static final String ND4J_CPU_LOAD_OPENBLAS_NOLAPACK = "org.bytedeco.javacpp.openblas_nolapack.load";
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

    private ND4JSystemProperties() {
    }
}
