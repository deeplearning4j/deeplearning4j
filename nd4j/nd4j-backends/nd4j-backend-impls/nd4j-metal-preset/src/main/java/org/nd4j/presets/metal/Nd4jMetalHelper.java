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

package org.nd4j.presets.metal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Helper class for the ND4J Metal/MLX JavaCPP preset.
 *
 * <p>Mirrors {@code Nd4jTpuHelper}: checks for the presence of the Metal native
 * library and exposes device-level metadata without requiring the full preset
 * to be generated.</p>
 *
 * <p>At build time (when preset generation runs on macOS arm64) this class is
 * passed as the {@code helper} attribute of {@code @Properties}. It is invoked
 * after the preset class loads to perform any additional JNI/framework setup.</p>
 */
public class Nd4jMetalHelper {

    private static final Logger log = LoggerFactory.getLogger(Nd4jMetalHelper.class);

    /**
     * The environment variable or system property that specifies the path to
     * the libnd4j Metal dylib or MLX framework, if not on the default dylib
     * search path. Mirrors {@code Nd4jTpuHelper.PJRT_PATH_ENV}.
     */
    public static final String METAL_LIBRARY_PATH_PROP = "nd4j.metal.library.path";
    public static final String METAL_LIBRARY_PATH_ENV  = "ND4J_METAL_LIBRARY_PATH";

    private Nd4jMetalHelper() {}

    /**
     * Returns true when:
     * <ol>
     *   <li>The JVM is running on macOS arm64, AND</li>
     *   <li>A loadable libnd4j_metal.dylib is on the native library path
     *       (or pointed to by {@code METAL_LIBRARY_PATH_ENV} / system property).</li>
     * </ol>
     * Stub returns false until native bindings are generated.
     */
    public static boolean isMetalAvailable() {
        // TODO: attempt Class.forName("org.nd4j.linalg.metal.bindings.Nd4jMetal")
        // and call getMetalDeviceCount() once the preset is generated.
        return false;
    }

    /**
     * Returns the number of Metal-capable GPUs visible to this process.
     * Stub returns 0 until native bindings are generated.
     */
    public static int getDeviceCount() {
        // TODO: delegate to Nd4jMetal.getMetalDeviceCount()
        return 0;
    }
}
