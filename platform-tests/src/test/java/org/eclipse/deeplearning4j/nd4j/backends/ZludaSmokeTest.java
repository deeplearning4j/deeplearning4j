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
package org.eclipse.deeplearning4j.nd4j.backends;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.util.ServiceLoader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Smoke tests for running the CUDA backend on non-NVIDIA GPUs through ZLUDA
 * (https://github.com/vosen/ZLUDA). See ADR 0087 (ZLUDA Transpiler Support)
 * and ADR 0102 (Accelerator and CPU-Architecture CI Test Tiers).
 *
 * Selected by {@code -Ptest-zluda} (groups zluda,rocm,amd-gpu), which activates
 * automatically when the {@code ZLUDA_PATH} environment variable is set. On a
 * machine without ZLUDA these tests skip via assumptions, so they are safe in
 * unfiltered runs.
 *
 * Scope is deliberately limited to what ZLUDA supports as of v6 (2026):
 * core driver/runtime API, PTX JIT, and cuBLAS GEMM paths. cuDNN, cuSPARSE
 * depth, and CUDA graph capture/replay are NOT exercised here — ZLUDA does not
 * reliably support stream capture, so DSP CUDA_GRAPHS mode is out of scope.
 *
 * All backend-specific classes are accessed reflectively so this class compiles
 * without the nd4j-zluda dependency on the classpath.
 */
@Slf4j
@Tag(TagNames.ZLUDA)
@Tag(TagNames.ROCM)
@Tag(TagNames.AMD_GPU)
@DisplayName("ZLUDA smoke tests (CUDA backend on AMD/Intel GPUs)")
public class ZludaSmokeTest {

    private static String zludaPath;

    @BeforeAll
    public static void requireZludaEnvironment() {
        zludaPath = System.getenv("ZLUDA_PATH");
        assumeTrue(zludaPath != null && !zludaPath.isEmpty(),
                "ZLUDA_PATH not set — skipping ZLUDA smoke tests");
        assumeTrue(new File(zludaPath).isDirectory(),
                "ZLUDA_PATH does not point to a directory: " + zludaPath);
    }

    @Test
    @DisplayName("ZLUDA environment and backend discovery report")
    public void zludaEnvironmentReport() {
        log.info("ZLUDA_PATH          = {}", zludaPath);
        log.info("ZLUDA_TARGET        = {}", System.getenv("ZLUDA_TARGET"));
        log.info("LD_LIBRARY_PATH     = {}", System.getenv("LD_LIBRARY_PATH"));
        log.info("java.library.path   = {}", System.getProperty("java.library.path"));
        log.info("javacpp pathsFirst  = {}", System.getProperty("org.bytedeco.javacpp.pathsFirst"));

        File rootLibCuda = new File(zludaPath, "libcuda.so");
        File libDirLibCuda = new File(zludaPath, "lib/libcuda.so");
        log.info("libcuda.so at ZLUDA_PATH root: {}", rootLibCuda.isFile());
        log.info("libcuda.so at ZLUDA_PATH/lib : {}", libDirLibCuda.isFile());
        if (rootLibCuda.isFile() && !libDirLibCuda.isFile()) {
            // JZludaBackend.checkZludaAvailable() only looks in ZLUDA_PATH/lib — a root-level
            // layout means the SPI backend will report unavailable even though interception works.
            log.warn("libcuda.so found only at ZLUDA_PATH root; JZludaBackend expects ZLUDA_PATH/lib/libcuda.so. "
                    + "Create a lib/ symlink or JZludaBackend.isAvailable() will be false.");
        }

        boolean zludaSpiOnClasspath;
        try {
            Class.forName("org.nd4j.linalg.jzluda.JZludaBackend");
            zludaSpiOnClasspath = true;
        } catch (ClassNotFoundException e) {
            zludaSpiOnClasspath = false;
        }
        log.info("nd4j-zluda SPI backend on classpath: {}", zludaSpiOnClasspath);

        for (Nd4jBackend backend : ServiceLoader.load(Nd4jBackend.class)) {
            String available;
            String canRun;
            try {
                available = String.valueOf(backend.isAvailable());
            } catch (Throwable t) {
                available = "threw: " + t.getMessage();
            }
            try {
                canRun = String.valueOf(backend.canRun());
            } catch (Throwable t) {
                canRun = "threw: " + t.getMessage();
            }
            log.info("Discovered backend {} (priority {}): isAvailable={}, canRun={}",
                    backend.getClass().getName(), backend.getPriority(), available, canRun);
        }

        assertTrue(rootLibCuda.isFile() || libDirLibCuda.isFile(),
                "No libcuda.so found under ZLUDA_PATH (checked " + rootLibCuda + " and " + libDirLibCuda + ")");
    }

    @Test
    @DisplayName("Backend initializes and allocates through ZLUDA")
    public void backendInitializes() {
        INDArray probe = Nd4j.create(DataType.FLOAT, 2, 2);
        assertNotNull(probe);
        assertNotNull(Nd4j.getBackend());
        log.info("Active backend: {}", Nd4j.getBackend().getClass().getName());
        try {
            log.info("Backend build info:\n{}", Nd4j.getBackend().buildInfo());
        } catch (Throwable t) {
            log.warn("buildInfo() failed: {}", t.getMessage());
        }
    }

    @Test
    @DisplayName("Elementwise arithmetic produces correct values")
    public void elementwiseArithmetic() {
        int len = 64;
        float[] aData = new float[len];
        float[] bData = new float[len];
        for (int i = 0; i < len; i++) {
            aData[i] = ((i * 31 + 7) % 13) - 6.0f;
            bData[i] = ((i * 17 + 3) % 11) - 5.0f;
        }
        INDArray a = Nd4j.create(aData, new long[]{len});
        INDArray b = Nd4j.create(bData, new long[]{len});

        float[] sum = a.add(b).toFloatVector();
        float[] diff = a.sub(b).toFloatVector();
        float[] scaled = a.mul(2.5f).toFloatVector();
        for (int i = 0; i < len; i++) {
            assertEquals(aData[i] + bData[i], sum[i], 1e-5f, "add mismatch at " + i);
            assertEquals(aData[i] - bData[i], diff[i], 1e-5f, "sub mismatch at " + i);
            assertEquals(aData[i] * 2.5f, scaled[i], 1e-5f, "mul mismatch at " + i);
        }
    }

    @Test
    @DisplayName("GEMM (cuBLAS path) matches pure-Java reference")
    public void matrixMultiplyMatchesJavaReference() {
        int m = 32, k = 48, n = 16;
        float[] aData = new float[m * k];
        float[] bData = new float[k * n];
        for (int i = 0; i < aData.length; i++)
            aData[i] = (((i * 31 + 11) % 13) - 6) / 3.0f;
        for (int i = 0; i < bData.length; i++)
            bData[i] = (((i * 23 + 5) % 9) - 4) / 2.0f;

        INDArray a = Nd4j.create(aData, new long[]{m, k});
        INDArray b = Nd4j.create(bData, new long[]{k, n});
        float[] actual = a.mmul(b).ravel().toFloatVector();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double expected = 0;
                for (int x = 0; x < k; x++)
                    expected += (double) aData[i * k + x] * bData[x * n + j];
                float act = actual[i * n + j];
                double tol = Math.max(1e-3, Math.abs(expected) * 1e-3);
                assertEquals(expected, act, tol, "gemm mismatch at (" + i + "," + j + ")");
            }
        }
    }

    @Test
    @DisplayName("Reductions and transforms produce correct values")
    public void reductionsAndTransforms() {
        int len = 128;
        float[] data = new float[len];
        double sum = 0;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < len; i++) {
            data[i] = (((i * 29 + 13) % 17) - 8) / 4.0f;
            sum += data[i];
            max = Math.max(max, data[i]);
        }
        INDArray v = Nd4j.create(data, new long[]{len});

        assertEquals(sum, v.sumNumber().doubleValue(), 1e-3, "sum mismatch");
        assertEquals(sum / len, v.meanNumber().doubleValue(), 1e-4, "mean mismatch");
        assertEquals(max, v.maxNumber().doubleValue(), 1e-5, "max mismatch");

        float[] exp = Transforms.exp(v, true).toFloatVector();
        float[] tanh = Transforms.tanh(v, true).toFloatVector();
        for (int i = 0; i < len; i++) {
            double expExpected = Math.exp(data[i]);
            assertEquals(expExpected, exp[i], Math.max(1e-4, expExpected * 1e-4), "exp mismatch at " + i);
            assertEquals(Math.tanh(data[i]), tanh[i], 1e-4, "tanh mismatch at " + i);
        }

        // 2d reduction along a dimension exercises TAD paths
        INDArray mat = Nd4j.create(data, new long[]{8, 16});
        float[] rowSums = mat.sum(1).toFloatVector();
        for (int r = 0; r < 8; r++) {
            double expected = 0;
            for (int c = 0; c < 16; c++)
                expected += data[r * 16 + c];
            assertEquals(expected, rowSums[r], 1e-3, "row-sum mismatch at row " + r);
        }
    }
}
