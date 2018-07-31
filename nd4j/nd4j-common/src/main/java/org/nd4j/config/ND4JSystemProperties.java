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

    private ND4JSystemProperties(){ }

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


    //TODO ADD JAVACCP MEMORY OPTIONS HERE
    //(Technically they aren't ND4J system properties, but they are very important)

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


    public static final String ND4J_CPU_LOAD_OPENBLAS = "org.bytedeco.javacpp.openblas.load";

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
}
