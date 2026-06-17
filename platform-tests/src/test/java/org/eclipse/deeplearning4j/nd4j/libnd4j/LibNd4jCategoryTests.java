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
package org.eclipse.deeplearning4j.nd4j.libnd4j;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

/**
 * Categorized test runner for libnd4j C++ tests.
 * This class provides separate test methods for different categories of native tests,
 * allowing more granular control over which tests to run.
 *
 * Categories:
 * - Operations: DeclarableOps, BroadcastableOps, LegacyOps, ParityOps
 * - Arrays: NDArray, ArrayOptions, MultiDataType
 * - Convolutions: Convolution, Pooling
 * - Attention: Attention mechanisms and NLP
 * - Helpers: Various helper utilities
 * - Memory: Workspace, Memory management
 * - Graphs: Graph execution and serialization
 *
 * Usage:
 * mvn test -Dtest=LibNd4jCategoryTests#testDeclarableOps
 * mvn test -Dtest=LibNd4jCategoryTests#testConvolutions
 */
@Tag("native")
@Tag("libnd4j")
@DisplayName("libnd4j Native Tests by Category")
public class LibNd4jCategoryTests {
    private static final Logger log = LoggerFactory.getLogger(LibNd4jCategoryTests.class);

    private static Path tempXmlDir;

    // Test suite categories
    private static final List<String> DECLARABLE_OPS_SUITES = Arrays.asList(
        "DeclarableOpsTests1", "DeclarableOpsTests2", "DeclarableOpsTests3",
        "DeclarableOpsTests4", "DeclarableOpsTests5", "DeclarableOpsTests6",
        "DeclarableOpsTests7", "DeclarableOpsTests8", "DeclarableOpsTests9",
        "DeclarableOpsTests10", "DeclarableOpsTests11", "DeclarableOpsTests12",
        "DeclarableOpsTests13", "DeclarableOpsTests14", "DeclarableOpsTests15",
        "DeclarableOpsTests16", "DeclarableOpsTests17", "DeclarableOpsTests18",
        "DeclarableOpsTests19"
    );

    private static final List<String> ARRAY_SUITES = Arrays.asList(
        "NDArrayTest", "NDArrayTest2", "NDArrayListTests",
        "ArrayOptionsTests", "MultiDataTypeTests", "DataTypesValidationTests"
    );

    private static final List<String> CONVOLUTION_SUITES = Arrays.asList(
        "ConvolutionTests1", "ConvolutionTests2",
        "TypedConvolutionTests1", "TypedConvolutionTests2"
    );

    private static final List<String> ATTENTION_NLP_SUITES = Arrays.asList(
        "AttentionTests", "NlpTests"
    );

    private static final List<String> HELPER_SUITES = Arrays.asList(
        "HelpersTests1", "HelpersTests2", "ShapeTests", "ShapeTests2",
        "ShapeUtilsTests", "ConstantShapeHelperTests", "ConstantTadHelperTests"
    );

    private static final List<String> MEMORY_SUITES = Arrays.asList(
        "WorkspaceTests", "MemoryUtilsTests", "DataBufferTests"
    );

    private static final List<String> GRAPH_SUITES = Arrays.asList(
        "GraphTests", "GraphStateTests", "GraphHolderTests",
        "FlatBuffersTest", "FlatUtilsTests"
    );

    private static final List<String> BROADCAST_SUITES = Arrays.asList(
        "BroadcastableOpsTests", "BroadcastMultiDimTest"
    );

    private static final List<String> LEGACY_SUITES = Arrays.asList(
        "LegacyOpsTests", "ParityOpsTests", "BooleanOpsTests"
    );

    private static final List<String> RNG_SUITES = Arrays.asList(
        "RNGTests", "GraphRandomGeneratorTests"
    );

    private static final List<String> BACKPROP_SUITES = Arrays.asList(
        "BackpropTests"
    );

    @BeforeAll
    static void setUp() throws Exception {
        tempXmlDir = Files.createTempDirectory("libnd4j-category-tests");
        log.info("XML results will be written to: {}", tempXmlDir);
    }

    @AfterAll
    static void tearDown() throws Exception {
        if (tempXmlDir != null && Files.exists(tempXmlDir)) {
            try {
                Files.walk(tempXmlDir)
                    .sorted((a, b) -> -a.compareTo(b))
                    .forEach(path -> {
                        try { Files.deleteIfExists(path); } catch (Exception e) { }
                    });
            } catch (Exception e) {
                log.debug("Error cleaning up temp directory", e);
            }
        }
    }

    static boolean nativeTestsAreAvailable() {
        return Libnd4jTestHelper.isNativeTestingEnabled() &&
               Libnd4jTestHelper.isNativeTestsAvailable();
    }

    // ==================== Category Test Methods ====================

    @Nested
    @DisplayName("Declarable Operations Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class DeclarableOpsTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#declarableOpsSuites")
        void testDeclarableOpsSuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> declarableOpsSuites() {
        return DECLARABLE_OPS_SUITES.stream();
    }

    @Nested
    @DisplayName("Array Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class ArrayTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#arraySuites")
        void testArraySuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> arraySuites() {
        return ARRAY_SUITES.stream();
    }

    @Nested
    @DisplayName("Convolution Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class ConvolutionTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#convolutionSuites")
        void testConvolutionSuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> convolutionSuites() {
        return CONVOLUTION_SUITES.stream();
    }

    @Nested
    @DisplayName("Attention & NLP Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class AttentionNlpTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#attentionNlpSuites")
        void testAttentionNlpSuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> attentionNlpSuites() {
        return ATTENTION_NLP_SUITES.stream();
    }

    @Nested
    @DisplayName("Helper Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class HelperTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#helperSuites")
        void testHelperSuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> helperSuites() {
        return HELPER_SUITES.stream();
    }

    @Nested
    @DisplayName("Memory & Workspace Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class MemoryTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#memorySuites")
        void testMemorySuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> memorySuites() {
        return MEMORY_SUITES.stream();
    }

    @Nested
    @DisplayName("Graph Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class GraphTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#graphSuites")
        void testGraphSuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> graphSuites() {
        return GRAPH_SUITES.stream();
    }

    @Nested
    @DisplayName("Broadcast Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class BroadcastTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#broadcastSuites")
        void testBroadcastSuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> broadcastSuites() {
        return BROADCAST_SUITES.stream();
    }

    @Nested
    @DisplayName("Legacy & Parity Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class LegacyTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#legacySuites")
        void testLegacySuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> legacySuites() {
        return LEGACY_SUITES.stream();
    }

    @Nested
    @DisplayName("RNG Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class RngTests {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#rngSuites")
        void testRngSuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> rngSuites() {
        return RNG_SUITES.stream();
    }

    @Nested
    @DisplayName("Backprop Tests")
    @EnabledIf("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#nativeTestsAreAvailable")
    class BackpropTestsSuite {

        @ParameterizedTest(name = "{0}")
        @MethodSource("org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests#backpropSuites")
        void testBackpropSuite(String suiteName) {
            runTestSuite(suiteName);
        }
    }

    static Stream<String> backpropSuites() {
        return BACKPROP_SUITES.stream();
    }

    // ==================== Utility Methods ====================

    /**
     * Run a single test suite and assert it passes.
     */
    private void runTestSuite(String suiteName) {
        assumeTrue(Libnd4jTestHelper.isNativeTestsAvailable(),
            "Native tests not available");

        Path xmlPath = tempXmlDir.resolve(suiteName + "-results.xml");
        String filter = suiteName + ".*";

        log.info("Running test suite: {}", suiteName);
        long startTime = System.currentTimeMillis();

        Libnd4jTestHelper.TestResult result = Libnd4jTestHelper.runTest(filter, xmlPath);

        long duration = System.currentTimeMillis() - startTime;

        // Log results
        if (Files.exists(xmlPath)) {
            GTestResults gtestResults = GTestResultParser.parseXmlFile(xmlPath);
            int total = gtestResults.getTotalTestCount();
            int failures = gtestResults.getTotalFailureCount();
            log.info("{}: {} tests, {} passed, {} failed in {}ms",
                suiteName, total, total - failures, failures, duration);

            if (gtestResults.hasFailures()) {
                for (GTestCase failure : gtestResults.getAllFailedTests()) {
                    log.error("  FAILED: {} - {}",
                        failure.getFullName(),
                        failure.getFailureMessage());
                }
            }
        } else {
            log.info("{}: completed in {}ms (exit code: {})",
                suiteName, duration, result.getExitCode());
        }

        assertTrue(result.isPassed(),
            () -> buildFailureMessage(suiteName, result, xmlPath));
    }

    /**
     * Build detailed failure message.
     */
    private String buildFailureMessage(String suiteName, Libnd4jTestHelper.TestResult result, Path xmlPath) {
        StringBuilder msg = new StringBuilder();
        msg.append("Native test suite failed: ").append(suiteName).append("\n");
        msg.append("Exit code: ").append(result.getExitCode()).append("\n");

        if (Files.exists(xmlPath)) {
            GTestResults gtestResults = GTestResultParser.parseXmlFile(xmlPath);
            List<GTestCase> failures = gtestResults.getAllFailedTests();
            if (!failures.isEmpty()) {
                msg.append("\nFailed tests:\n");
                for (GTestCase failure : failures) {
                    msg.append("  - ").append(failure.getFullName()).append("\n");
                    if (failure.getFailureMessage() != null) {
                        msg.append("    ").append(failure.getFailureMessage()).append("\n");
                    }
                }
            }
        }

        // Append truncated output
        String stdout = result.getStdout();
        if (stdout != null && !stdout.isEmpty()) {
            String[] lines = stdout.split("\n");
            int start = Math.max(0, lines.length - 30);
            msg.append("\nOutput (last ").append(lines.length - start).append(" lines):\n");
            for (int i = start; i < lines.length; i++) {
                msg.append(lines[i]).append("\n");
            }
        }

        return msg.toString();
    }

    // ==================== Individual Quick Tests ====================

    @Test
    @DisplayName("Quick Smoke Test - ArrayOptions")
    @EnabledIf("nativeTestsAreAvailable")
    void quickSmokeTestArrayOptions() {
        runTestSuite("ArrayOptionsTests");
    }

    @Test
    @DisplayName("Quick Smoke Test - ShapeUtils")
    @EnabledIf("nativeTestsAreAvailable")
    void quickSmokeTestShapeUtils() {
        runTestSuite("ShapeUtilsTests");
    }

    @Test
    @DisplayName("Quick Smoke Test - Attention")
    @EnabledIf("nativeTestsAreAvailable")
    void quickSmokeTestAttention() {
        runTestSuite("AttentionTests");
    }
}
