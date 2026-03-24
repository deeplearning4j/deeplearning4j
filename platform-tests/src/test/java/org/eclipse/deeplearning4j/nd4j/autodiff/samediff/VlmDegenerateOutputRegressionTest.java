/*
 *  ******************************************************************************
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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for degenerate VLM output bugs.
 *
 * Each test isolates ONE specific behavior that caused the VLM benchmark
 * to produce repeating garbage instead of coherent English text.
 *
 * Root causes found:
 * 1. S1: Frozen fast path mask off-by-one (NativeDynamicShapePlan_cuda.cu)
 * 2. S2: Triton TF32 precision compounding (TritonIRBuilder_kernels.cpp)
 * 3. S3: Cast cache stale entries surviving warmup→capture (NativeDynamicShapePlan_gpubackend.cpp)
 *
 * Run with:
 *   cd platform-tests && mvn test \
 *     -Dtest=VlmDegenerateOutputRegressionTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
@DisplayName("VlmDegenerateOutputRegressionTest")
public class VlmDegenerateOutputRegressionTest {

    // ═══════════════════════════════════════════════════════════════════════
    // Test: tritonTf32Enabled flag defaults to false
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Regression: Triton TF32 was hardcoded ON in TritonIRBuilder_kernels.cpp,
     * causing 10-bit mantissa precision to compound across ~3840 ops per decode
     * step. After hundreds of steps, accumulated error caused degenerate output.
     *
     * Fix: tritonTf32Enabled defaults to false (IEEE precision). Can be
     * explicitly enabled via Environment flag when accuracy is acceptable.
     */
    @Test
    @DisplayName("tritonTf32Enabled defaults to false (IEEE precision)")
    public void testTritonTf32DefaultsFalse() {
        Environment env = Nd4j.getEnvironment();
        assertFalse(env.tritonTf32Enabled(),
                "tritonTf32Enabled must default to false — TF32 compounds precision loss "
                + "across thousands of transformer decode ops, causing degenerate output");
    }

    /**
     * Test that tritonTf32Enabled can be toggled and read back.
     */
    @Test
    @DisplayName("tritonTf32Enabled setter/getter round-trip")
    public void testTritonTf32SetterGetter() {
        Environment env = Nd4j.getEnvironment();
        boolean original = env.tritonTf32Enabled();
        try {
            env.setTritonTf32Enabled(true);
            assertTrue(env.tritonTf32Enabled(), "tritonTf32Enabled should be true after setting to true");

            env.setTritonTf32Enabled(false);
            assertFalse(env.tritonTf32Enabled(), "tritonTf32Enabled should be false after setting to false");
        } finally {
            env.setTritonTf32Enabled(original);
        }
    }

    /**
     * Test that cublasTf32Enabled setter exists and works (was read-only before).
     */
    @Test
    @DisplayName("cublasTf32Enabled setter/getter round-trip")
    public void testCublasTf32SetterGetter() {
        Environment env = Nd4j.getEnvironment();
        boolean original = env.cublasTf32Enabled();
        try {
            env.setCublasTf32Enabled(true);
            assertTrue(env.cublasTf32Enabled(), "cublasTf32Enabled should be true after setting to true");

            env.setCublasTf32Enabled(false);
            assertFalse(env.cublasTf32Enabled(), "cublasTf32Enabled should be false after setting to false");
        } finally {
            env.setCublasTf32Enabled(original);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Test: tritonTf32 and cublasTf32 are independent
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Verify that Triton TF32 and cuBLAS TF32 are independent flags.
     * Setting one should not affect the other.
     */
    @Test
    @DisplayName("Triton TF32 and cuBLAS TF32 are independent")
    public void testTritonAndCublasTf32Independent() {
        Environment env = Nd4j.getEnvironment();
        boolean origTriton = env.tritonTf32Enabled();
        boolean origCublas = env.cublasTf32Enabled();
        try {
            // Set Triton ON, cuBLAS OFF
            env.setTritonTf32Enabled(true);
            env.setCublasTf32Enabled(false);
            assertTrue(env.tritonTf32Enabled());
            assertFalse(env.cublasTf32Enabled());

            // Set cuBLAS ON, verify Triton still ON
            env.setCublasTf32Enabled(true);
            assertTrue(env.tritonTf32Enabled(), "Triton TF32 should not change when cuBLAS TF32 is set");
            assertTrue(env.cublasTf32Enabled());

            // Set Triton OFF, verify cuBLAS still ON
            env.setTritonTf32Enabled(false);
            assertFalse(env.tritonTf32Enabled());
            assertTrue(env.cublasTf32Enabled(), "cuBLAS TF32 should not change when Triton TF32 is set");
        } finally {
            env.setTritonTf32Enabled(origTriton);
            env.setCublasTf32Enabled(origCublas);
        }
    }
}
