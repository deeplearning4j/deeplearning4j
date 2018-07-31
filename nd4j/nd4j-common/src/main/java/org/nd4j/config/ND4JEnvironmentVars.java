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

public class ND4JEnvironmentVars {

    private ND4JEnvironmentVars(){ }

    /**
     * Applicability: nd4j-native, when multiple backends are on classpath<br>
     * Description: Defines the priority that the CPU/Native backend should be loaded (or attempt to be loaded). If this
     * is set to a higher value than {@link #BACKEND_PRIORITY_GPU} (which has default value 100) the native backend
     * will be loaded in preference to the CUDA backend, when both are on the classpath. Default value: 0
     */
    public static final String BACKEND_PRIORITY_CPU = "BACKEND_PRIORITY_CPU";

    /**
     * Applicability: nd4j-cuda-xx, when multiple backends are on classpath<br>
     * Description: Defines the priority that the CUDA (GPU) backend should be loaded (or attempt to be loaded). If this
     * is set to a higher value than {@link #BACKEND_PRIORITY_CPU} (which has default value 0) the GPU backend
     * will be loaded in preference to the CUDA backend, when both are on the classpath. Default value: 100 - hence
     * by default, the CUDA backend will be loaded when both it and the CPU/native backend are on the classpath
     */
    public static final String BACKEND_PRIORITY_GPU = "BACKEND_PRIORITY_GPU";

    /**
     * Applicability: always - but only if an ND4J backend cannot be found/loaded via standard ServiceLoader mechanisms<br>
     * Description: Set this environment variable to a set fully qualified JAR files to attempt to load before failing on
     * not loading a backend. JAR files should be semi-colon delimited; i.e., "/some/file.jar;/other/path.jar".
     * This should rarely be required in practice - for example, only in dynamic class loading/dynamic classpath scenarios<br>
     * For equivalent system property, see {@link ND4JSystemProperties#DYNAMIC_LOAD_CLASSPATH_PROPERTY} for the equivalent
     * system property (that will take precidence if both are set)
     */
    public static final String BACKEND_DYNAMIC_LOAD_CLASSPATH = "ND4J_DYNAMIC_LOAD_CLASSPATH";

    /**
     * Applicability: nd4j-native backend<br>
     * Description: Sets the number of OpenMP parallel threads for ND4J native operations (and also native BLAS libraries
     * such as Intel MKL and OpenBLAS).
     * By default, this will be set to the number of physical cores (i.e., excluding hyperthreading cores), which usually
     * provides optimal performance. Setting this to a larger value than the number of physical cores (for example, equal
     * to number of logical cores - i.e., setting to 16 on an 8-core + hypethreading processor) - can result in reduced
     * performance<br>
     * Note that if you have a significant number of parallel Java threads (for example, Spark or ParallelWrapper), or
     * you want to keep some cores free for other programs - you may want to reduce this value.
     * @see #ND4J_SKIP_BLAS_THREADS
     */
    public static final String OMP_NUM_THREADS = "OMP_NUM_THREADS";

    /**
     * Applicability: nd4j-native backend<br>
     * Description: Skips the setting of the {@link #OMP_NUM_THREADS} property for ND4J ops. Note that this property
     * will usually still take effect for native BLAS libraries (MKL, OpenBLAS) even if this property is set
     */
    public static final String ND4J_SKIP_BLAS_THREADS = "ND4J_SKIP_BLAS_THREADS";
}
