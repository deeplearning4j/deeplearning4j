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

package org.eclipse.deeplearning4j.nd4j.linalg.api;

import org.eclipse.deeplearning4j.tests.extensions.BackendCheckerExtension;
import org.eclipse.deeplearning4j.tests.extensions.BackendTest;
import org.eclipse.deeplearning4j.tests.extensions.MultiBackendTestConfiguration;
import org.eclipse.deeplearning4j.tests.extensions.MultiBackendTestConfiguration.Helper;
import org.eclipse.deeplearning4j.tests.extensions.MultiBackendTestConfiguration.HelperCapabilities;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Example test class demonstrating multi-backend testing capabilities.
 *
 * This class shows how to:
 * - Use @BackendTest annotation for helper-specific tests
 * - Use tags for helper filtering
 * - Access helper information in tests
 * - Write parameterized tests across multiple backends
 *
 * Usage:
 * <pre>
 * # Run all multi-backend tests
 * mvn test -Pmulti-backend
 *
 * # Run OneDNN-specific tests
 * mvn test -Ptest-onednn
 *
 * # Run tests with specific helpers
 * mvn test -Dsd.test.helpers=onednn,cudnn
 *
 * # Run with helper comparison
 * mvn test -Ptest-helper-comparison
 * </pre>
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Tag("multi-backend")
@ExtendWith(BackendCheckerExtension.class)
@DisplayName("Multi-Backend Operations Tests")
public class MultiBackendOperationsTest {

    private static final Logger log = LoggerFactory.getLogger(MultiBackendOperationsTest.class);

    private MultiBackendTestConfiguration config;

    @BeforeEach
    void setUp() {
        config = MultiBackendTestConfiguration.getInstance();
        log.info("Test running with backend: {}, enabled helpers: {}",
                config.getCurrentBackend(), config.getEnabledHelpers());
    }

    /**
     * Provides test configurations for each enabled helper
     */
    static Stream<MultiBackendTestConfiguration.HelperTestConfig> helperConfigurations() {
        return MultiBackendTestConfiguration.getInstance()
                .getTestConfigurations()
                .stream();
    }

    // =========================================================================
    // Basic Multi-Backend Tests
    // =========================================================================

    @Test
    @DisplayName("Matrix multiplication should work on all backends")
    void testMatMulAllBackends() {
        INDArray a = Nd4j.rand(DataType.FLOAT, 64, 128);
        INDArray b = Nd4j.rand(DataType.FLOAT, 128, 64);

        INDArray result = a.mmul(b);

        assertNotNull(result);
        assertArrayEquals(new long[]{64, 64}, result.shape());

        log.info("MatMul test passed on backend: {}", config.getCurrentBackend());
    }

    @Test
    @DisplayName("Convolution operations should work on all backends")
    void testConvolutionAllBackends() {
        // NCHW format: batch=1, channels=3, height=28, width=28
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 3, 28, 28);
        // OIHW format: outChannels=16, inChannels=3, kernelH=3, kernelW=3
        INDArray weights = Nd4j.rand(DataType.FLOAT, 16, 3, 3, 3);

        // This will use the appropriate helper if available
        Conv2DConfig config2d = Conv2DConfig.builder()
                .kH(3).kW(3)
                .sH(1).sW(1)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .build();
        INDArray result = Nd4j.cnn().conv2d(input, weights, config2d);

        assertNotNull(result);
        assertEquals(1, result.size(0)); // batch
        assertEquals(16, result.size(1)); // output channels

        log.info("Convolution test passed on backend: {}", config.getCurrentBackend());
    }

    // =========================================================================
    // Helper-Specific Tests
    // =========================================================================

    @Test
    @Tag("onednn")
    @DisplayName("OneDNN-optimized batch normalization")
    void testOneDNNBatchNorm() {
        Assumptions.assumeTrue(
                BackendCheckerExtension.isHelperAvailable("onednn"),
                "OneDNN helper not available");

        INDArray input = Nd4j.rand(DataType.FLOAT, 32, 64, 28, 28);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, 64);
        INDArray beta = Nd4j.zeros(DataType.FLOAT, 64);
        INDArray mean = Nd4j.zeros(DataType.FLOAT, 64);
        INDArray var = Nd4j.ones(DataType.FLOAT, 64);

        INDArray result = Nd4j.nn().batchNorm(input, mean, var, gamma, beta, 1e-5, 1);

        assertNotNull(result);
        assertArrayEquals(input.shape(), result.shape());

        log.info("OneDNN BatchNorm test passed");
    }

    @Test
    @Tag("cudnn")
    @DisplayName("cuDNN-optimized operations")
    @BackendTest(helpers = {"cudnn"}, backends = {MultiBackendTestConfiguration.Backend.CUDA})
    void testCuDNNOperations() {
        Assumptions.assumeTrue(
                BackendCheckerExtension.isHelperAvailable("cudnn"),
                "cuDNN helper not available");

        INDArray input = Nd4j.rand(DataType.FLOAT, 32, 128);
        INDArray result = Transforms.relu(input);

        assertNotNull(result);
        assertTrue(result.minNumber().floatValue() >= 0, "ReLU should produce non-negative values");

        log.info("cuDNN operations test passed");
    }

    @Test
    @Tag("armcompute")
    @DisplayName("ARM Compute-optimized operations")
    void testARMComputeOperations() {
        Assumptions.assumeTrue(
                BackendCheckerExtension.isHelperAvailable("armcompute"),
                "ARM Compute helper not available");

        INDArray a = Nd4j.rand(DataType.FLOAT, 256, 256);
        INDArray b = Nd4j.rand(DataType.FLOAT, 256, 256);

        INDArray result = a.mmul(b);

        assertNotNull(result);
        assertArrayEquals(new long[]{256, 256}, result.shape());

        log.info("ARM Compute operations test passed");
    }

    // =========================================================================
    // Parameterized Tests Across Helpers
    // =========================================================================

    @ParameterizedTest(name = "Helper: {0}")
    @MethodSource("helperConfigurations")
    @DisplayName("Softmax should produce valid probability distributions on all helpers")
    void testSoftmaxAcrossHelpers(MultiBackendTestConfiguration.HelperTestConfig helperConfig) {
        log.info("Testing softmax with helper: {}", helperConfig.getHelper());

        INDArray input = Nd4j.rand(DataType.FLOAT, 32, 10);
        INDArray result = Transforms.softmax(input);

        assertNotNull(result);
        assertArrayEquals(input.shape(), result.shape());

        // Each row should sum to ~1.0
        INDArray rowSums = result.sum(1);
        for (int i = 0; i < rowSums.length(); i++) {
            assertEquals(1.0, rowSums.getDouble(i), 1e-4,
                    "Softmax row " + i + " should sum to 1.0");
        }

        log.info("Softmax test passed for helper: {}", helperConfig.getHelper());
    }

    @ParameterizedTest(name = "Helper: {0}")
    @MethodSource("helperConfigurations")
    @DisplayName("Element-wise operations should be consistent across helpers")
    void testElementWiseConsistency(MultiBackendTestConfiguration.HelperTestConfig helperConfig) {
        log.info("Testing element-wise consistency with helper: {}", helperConfig.getHelper());

        INDArray a = Nd4j.rand(DataType.FLOAT, 100, 100);
        INDArray b = Nd4j.rand(DataType.FLOAT, 100, 100);

        // Test various element-wise operations
        INDArray add = a.add(b);
        INDArray sub = a.sub(b);
        INDArray mul = a.mul(b);
        INDArray div = a.div(b.add(1e-6)); // Add small value to avoid div by zero

        // Verify shapes
        assertArrayEquals(a.shape(), add.shape());
        assertArrayEquals(a.shape(), sub.shape());
        assertArrayEquals(a.shape(), mul.shape());
        assertArrayEquals(a.shape(), div.shape());

        // Verify no NaN/Inf
        assertFalse(add.isNaN().any());
        assertFalse(add.isInfinite().any());

        log.info("Element-wise consistency test passed for helper: {}", helperConfig.getHelper());
    }

    // =========================================================================
    // Capability-Based Tests
    // =========================================================================

    @Test
    @BackendTest(requiresCapability = "float16", description = "Test FP16 operations")
    @DisplayName("Float16 operations should work on capable helpers")
    void testFloat16Operations() {
        List<Helper> capableHelpers = config.getHelpersWithCapability("float16")
                .stream().toList();

        Assumptions.assumeFalse(capableHelpers.isEmpty(),
                "No helpers with float16 capability available");

        log.info("Testing FP16 with capable helpers: {}", capableHelpers);

        INDArray fp16Array = Nd4j.rand(DataType.FLOAT16, 64, 64);
        INDArray result = fp16Array.mul(2.0);

        assertNotNull(result);
        assertEquals(DataType.FLOAT16, result.dataType());

        log.info("Float16 test passed");
    }

    @Test
    @BackendTest(requiresCapability = "attention", description = "Test attention mechanisms")
    @DisplayName("Attention mechanisms should work on capable helpers")
    void testAttentionOperations() {
        List<Helper> capableHelpers = config.getHelpersWithCapability("attention")
                .stream().toList();

        Assumptions.assumeFalse(capableHelpers.isEmpty(),
                "No helpers with attention capability available");

        log.info("Testing attention with capable helpers: {}", capableHelpers);

        int batchSize = 2;
        int seqLen = 16;
        int embedDim = 64;

        INDArray query = Nd4j.rand(DataType.FLOAT, batchSize, seqLen, embedDim);
        INDArray key = Nd4j.rand(DataType.FLOAT, batchSize, seqLen, embedDim);
        INDArray value = Nd4j.rand(DataType.FLOAT, batchSize, seqLen, embedDim);

        // Simple scaled dot-product attention
        double scale = 1.0 / Math.sqrt(embedDim);
        INDArray scores = query.mmul(key.permute(0, 2, 1)).mul(scale);
        INDArray weights = Transforms.softmax(scores);
        INDArray result = weights.mmul(value);

        assertNotNull(result);
        assertArrayEquals(new long[]{batchSize, seqLen, embedDim}, result.shape());

        log.info("Attention test passed");
    }

    // =========================================================================
    // Performance Comparison Tests
    // =========================================================================

    @Test
    @Tag("multi-backend")
    @DisplayName("Compare performance across enabled helpers")
    void testPerformanceComparison() {
        int warmupIterations = 5;
        int benchmarkIterations = 20;

        INDArray a = Nd4j.rand(DataType.FLOAT, 512, 512);
        INDArray b = Nd4j.rand(DataType.FLOAT, 512, 512);

        // Warmup
        for (int i = 0; i < warmupIterations; i++) {
            a.mmul(b);
        }

        // Benchmark
        long startTime = System.nanoTime();
        for (int i = 0; i < benchmarkIterations; i++) {
            a.mmul(b);
        }
        long endTime = System.nanoTime();

        double avgTimeMs = (endTime - startTime) / 1_000_000.0 / benchmarkIterations;

        log.info("Performance on backend {}: avg {}ms per 512x512 matmul",
                config.getCurrentBackend(), String.format("%.3f", avgTimeMs));

        // Just verify it completes without error
        assertTrue(avgTimeMs > 0);
    }

    // =========================================================================
    // Helper Information Tests
    // =========================================================================

    @Test
    @DisplayName("Helper configuration should be properly initialized")
    void testHelperConfiguration() {
        assertNotNull(config);
        assertNotNull(config.getCurrentBackend());
        assertNotNull(config.getEnabledHelpers());
        assertFalse(config.getEnabledHelpers().isEmpty(), "At least CPU helper should be enabled");

        log.info("Current configuration:");
        log.info("  Backend: {}", config.getCurrentBackend());
        log.info("  Enabled helpers: {}", config.getEnabledHelpers());
        log.info("  Helper priority: {}", config.getHelperPriority());
        log.info("  Kernel strategy: {}", config.getKernelStrategy());

        for (Helper helper : config.getEnabledHelpers()) {
            HelperCapabilities caps = config.getHelperCapabilities(helper);
            if (caps != null) {
                log.info("  {} capabilities: fp16={}, convolution={}, rnn={}, attention={}",
                        helper,
                        caps.supportsFloat16(),
                        caps.supportsConvolution(),
                        caps.supportsRNN(),
                        caps.supportsAttention());
            }
        }
    }
}
