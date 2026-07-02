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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    private static final Logger log = LoggerFactory.getLogger(TritonCacheManager.class);

    private TritonCacheManager() {
        // Static utility class
    }

    /**
     * Check if Triton is active: available AND has compiled or cached kernels in this session.
     *
     * <p>A backend may report {@link #isTritonAvailable()} = {@code true} (the Triton binary is
     * present) while having no active GPU context or no compiled kernels yet. This happens on
     * the CPU backend, which detects the Triton binary but cannot compile or launch GPU kernels.
     * Calling {@link #exportCache(Path)} in that state would return error code -1 from the
     * native layer, triggering a spurious {@link IllegalStateException}.</p>
     *
     * <p>This method additionally checks whether any Triton kernels have been compiled or served
     * from cache in the current JVM session. Use it before {@link #exportCache(Path)} when you
     * need to distinguish "no session yet" from "export error".</p>
     *
     * <pre>{@code
     * if (TritonCacheManager.isTritonActive()) {
     *     int exported = TritonCacheManager.exportCache(outputPath);
     * }
     * }</pre>
     *
     * @return {@code true} if Triton is available and has compiled or cached kernels
     */
    public static boolean isTritonActive() {
        if (!isTritonAvailable()) {
            return false;
        }
        NativeOps ops = Nd4j.getNativeOps();
        return ops.getTritonKernelLaunchCount() > 0 || ops.getTritonCacheHitCount() > 0;
    }

    /**
     * Export the current Triton kernel cache to a shareable bundle file.
     *
     * <p>Returns {@code 0} (with a warning log) when Triton is available but no kernels have
     * been compiled or served from cache in this session (e.g. the CPU backend has the Triton
     * binary present but no GPU context). Use {@link #isTritonActive()} to distinguish
     * "empty session" from an actual export error when a GPU backend is active.</p>
     *
     * @param outputPath path to write the .tkcache bundle
     * @return number of kernels exported; {@code 0} if no kernels exist in this session
     * @throws IllegalStateException if export fails on an active GPU backend
     */
    public static int exportCache(Path outputPath) {
        NativeOps ops = Nd4j.getNativeOps();
        int result = ops.exportTritonCacheBundle(outputPath.toAbsolutePath().toString());
        if (result < 0) {
            if (!isTritonActive()) {
                log.warn("TritonCacheManager.exportCache: Triton binary is present but no GPU context "
                        + "or compiled kernels exist in this session — returning 0 (nothing to export). "
                        + "Use isTritonActive() before calling exportCache() to avoid this warning.");
                return 0;
            }
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
     * Check if Triton support is compiled into this backend.
     *
     * <p><b>Important:</b> returning {@code true} means the Triton binary is present, but does
     * <em>not</em> mean a GPU context exists or that any kernels have been compiled. On the CPU
     * backend this can return {@code true} while kernel compilation and cache export are not
     * possible. Use {@link #isTritonActive()} when you need to confirm that Triton has actually
     * been used in this session.</p>
     *
     * @return {@code true} if Triton backend support is compiled and the binary is present
     */
    public static boolean isTritonAvailable() {
        return Nd4j.getNativeOps().isTritonAvailable();
    }
}
