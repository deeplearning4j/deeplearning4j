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

package org.nd4j.autodiff.samediff.execution;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.nio.file.Path;

/**
 * Manages shareable Triton kernel cache bundles (.tkcache files).
 *
 * <p>Triton kernel compilation takes 10-60+ seconds per sub-kernel. This manager
 * provides export/import capabilities to share pre-compiled PTX kernels across
 * machines with compatible GPU architectures.</p>
 *
 * <p>Exported bundles are STORED-only ZIP archives containing PTX binaries and
 * their metadata. Imported bundles are placed in the Triton override directory,
 * which has highest priority in the kernel lookup chain.</p>
 *
 * <h3>Usage:</h3>
 * <pre>{@code
 * // After running a model to populate the cache:
 * TritonCacheManager.exportCache(Path.of("/tmp/model.tkcache"));
 *
 * // On a new machine:
 * TritonCacheManager.importCache(Path.of("/tmp/model.tkcache"));
 *
 * // Inspect without importing:
 * String manifest = TritonCacheManager.inspectBundle(Path.of("/tmp/model.tkcache"));
 * }</pre>
 */
public class TritonCacheManager {

    private TritonCacheManager() {
        // Static utility class
    }

    /**
     * Export the current Triton kernel cache to a shareable bundle file.
     *
     * @param outputPath path to write the .tkcache bundle
     * @return number of kernels exported
     * @throws IllegalStateException if export fails
     */
    public static int exportCache(Path outputPath) {
        NativeOps ops = Nd4j.getNativeOps();
        int result = ops.exportTritonCacheBundle(outputPath.toAbsolutePath().toString());
        if (result < 0) {
            throw new IllegalStateException("Failed to export Triton cache bundle to: " + outputPath
                    + " (error code: " + result + ")");
        }
        return result;
    }

    /**
     * Import a .tkcache bundle into the Triton override directory with architecture validation.
     *
     * @param bundlePath path to the .tkcache bundle
     * @return number of kernels imported
     * @throws IllegalStateException if import fails
     * @throws IllegalArgumentException if bundle architecture is incompatible
     */
    public static int importCache(Path bundlePath) {
        return importCache(bundlePath, true);
    }

    /**
     * Import a .tkcache bundle into the Triton override directory.
     *
     * @param bundlePath path to the .tkcache bundle
     * @param validateArch if true, reject bundles compiled for incompatible GPU architectures
     * @return number of kernels imported
     * @throws IllegalStateException if import fails
     * @throws IllegalArgumentException if bundle architecture is incompatible (when validateArch=true)
     */
    public static int importCache(Path bundlePath, boolean validateArch) {
        NativeOps ops = Nd4j.getNativeOps();
        int result = ops.importTritonCacheBundle(bundlePath.toAbsolutePath().toString(), validateArch);
        if (result == -2) {
            throw new IllegalArgumentException("Bundle architecture is incompatible with this GPU: " + bundlePath);
        }
        if (result < 0) {
            throw new IllegalStateException("Failed to import Triton cache bundle from: " + bundlePath
                    + " (error code: " + result + ")");
        }
        return result;
    }

    /**
     * Read and return the manifest from a .tkcache bundle without importing it.
     *
     * @param bundlePath path to the .tkcache bundle
     * @return JSON string containing the bundle manifest
     */
    public static String inspectBundle(Path bundlePath) {
        NativeOps ops = Nd4j.getNativeOps();
        return ops.inspectTritonCacheBundle(bundlePath.toAbsolutePath().toString());
    }

    /**
     * Check if Triton is available on this backend.
     *
     * @return true if Triton backend support is compiled and available
     */
    public static boolean isTritonAvailable() {
        return Nd4j.getNativeOps().isTritonAvailable();
    }
}
