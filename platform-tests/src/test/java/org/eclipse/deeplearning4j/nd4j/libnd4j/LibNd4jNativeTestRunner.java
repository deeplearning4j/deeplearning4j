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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;
import static org.junit.jupiter.api.DynamicContainer.dynamicContainer;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

/**
 * JUnit 5 test runner for libnd4j C++ tests.
 * This class integrates the Google Test-based C++ test suite with JUnit 5,
 * allowing both test suites to be run together via Maven.
 *
 * Configuration:
 * - org.nd4j.libnd4j.home: Path to libnd4j directory
 * - org.nd4j.libnd4j.chip: Target chip (cpu/cuda), default: cpu
 * - org.nd4j.libnd4j.test.filter: GTest filter pattern, default: *
 * - org.nd4j.libnd4j.test.timeout: Timeout in minutes, default: 30
 * - org.nd4j.libnd4j.test.enabled: Enable/disable native tests, default: true
 * - org.nd4j.libnd4j.autobuild: Auto-build if executable not found, default: false
 *
 * Usage:
 * mvn test -Dtest=LibNd4jNativeTestRunner
 * mvn test -Dtest=LibNd4jNativeTestRunner -Dorg.nd4j.libnd4j.test.filter=ArrayOptionsTests.*
 */
@Tag("native")
@Tag("libnd4j")
@DisplayName("libnd4j Native C++ Tests")
public class LibNd4jNativeTestRunner {
    private static final Logger log = LoggerFactory.getLogger(LibNd4jNativeTestRunner.class);

    private static Path tempXmlDir;
    private static boolean nativeTestsAvailable;

    @BeforeAll
    static void setUp() throws Exception {
        // Check if native testing is enabled
        if (!Libnd4jTestHelper.isNativeTestingEnabled()) {
            log.info("Native testing is disabled via system property");
            nativeTestsAvailable = false;
            return;
        }

        // Try to build if auto-build is enabled
        nativeTestsAvailable = Libnd4jTestHelper.buildTestsIfNeeded();

        if (!nativeTestsAvailable) {
            log.warn("libnd4j native tests are not available. To enable:");
            log.warn("  1. Build libnd4j with tests: cd libnd4j && ./buildnativeoperations.sh -t");
            log.warn("  2. Or set -D{}=<path> to the build directory",
                Libnd4jTestHelper.LIBND4J_BUILD_DIR_PROPERTY);
        }

        // Create temp directory for XML output
        tempXmlDir = Files.createTempDirectory("libnd4j-test-results");
        log.info("XML results will be written to: {}", tempXmlDir);
    }

    @AfterAll
    static void tearDown() throws Exception {
        // Clean up temp directory
        if (tempXmlDir != null && Files.exists(tempXmlDir)) {
            try {
                Files.walk(tempXmlDir)
                    .sorted((a, b) -> -a.compareTo(b))
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (Exception e) {
                            log.debug("Could not delete: {}", path);
                        }
                    });
            } catch (Exception e) {
                log.debug("Error cleaning up temp directory", e);
            }
        }
    }

    /**
     * Condition method for enabling tests only when native tests are available.
     */
    static boolean nativeTestsAreAvailable() {
        return Libnd4jTestHelper.isNativeTestingEnabled() &&
               Libnd4jTestHelper.isNativeTestsAvailable();
    }

    /**
     * Run all native tests as a single test (useful for quick validation).
     */
    @Test
    @DisplayName("All Native Tests")
    @EnabledIf("nativeTestsAreAvailable")
    void runAllNativeTests() {
        assumeTrue(nativeTestsAvailable, "Native tests not available");

        String filter = Libnd4jTestHelper.getTestFilter();
        Path xmlPath = tempXmlDir.resolve("all-tests-results.xml");

        log.info("Running all native tests with filter: {}", filter);
        Libnd4jTestHelper.TestResult result = Libnd4jTestHelper.runTest(filter, xmlPath);

        // Parse and log results
        if (Files.exists(xmlPath)) {
            GTestResultParser.GTestResults gtestResults = GTestResultParser.parseXmlFile(xmlPath);
            log.info("Test results: {} total, {} passed, {} failed",
                gtestResults.getTotalTestCount(),
                gtestResults.getTotalTestCount() - gtestResults.getTotalFailureCount(),
                gtestResults.getTotalFailureCount());

            if (gtestResults.hasFailures()) {
                List<GTestResultParser.GTestCase> failures = gtestResults.getAllFailedTests();
                for (GTestResultParser.GTestCase failure : failures) {
                    log.error("FAILED: {} - {}", failure.getFullName(), failure.getFailureMessage());
                }
            }
        }

        assertTrue(result.isPassed(),
            () -> "Native tests failed:\n" + result.getFailureMessage() + "\n\nFull output:\n" + result.getStdout());
    }

    /**
     * Generate dynamic tests for each test suite.
     * This provides better granularity in test reporting.
     */
    @TestFactory
    @DisplayName("Native Test Suites")
    @EnabledIf("nativeTestsAreAvailable")
    Stream<DynamicNode> generateNativeTestSuites() {
        assumeTrue(nativeTestsAvailable, "Native tests not available");

        List<String> suites = Libnd4jTestHelper.discoverTestSuites();
        if (suites.isEmpty()) {
            log.warn("No test suites discovered");
            return Stream.of(dynamicTest("No tests discovered", () ->
                fail("Could not discover any test suites")));
        }

        // Apply filter if specified
        String filter = Libnd4jTestHelper.getTestFilter();
        if (!"*".equals(filter) && !filter.contains("*")) {
            // Exact suite filter
            suites = suites.stream()
                .filter(s -> s.equals(filter) || filter.startsWith(s + "."))
                .collect(Collectors.toList());
        }

        log.info("Generating tests for {} test suites", suites.size());

        return suites.stream()
            .map(suite -> dynamicContainer(
                suite,
                generateTestsForSuite(suite)
            ));
    }

    /**
     * Generate dynamic tests for individual test cases within a suite.
     */
    private Stream<DynamicTest> generateTestsForSuite(String suiteName) {
        List<String> testCases = Libnd4jTestHelper.discoverTestCases(suiteName);

        if (testCases.isEmpty()) {
            // Fall back to running the entire suite as one test
            return Stream.of(dynamicTest(
                suiteName + " (all)",
                () -> runTestSuite(suiteName)
            ));
        }

        return testCases.stream()
            .map(testCase -> dynamicTest(
                testCase,
                () -> runSingleTest(suiteName, testCase)
            ));
    }

    /**
     * Run an entire test suite.
     */
    private void runTestSuite(String suiteName) {
        Path xmlPath = tempXmlDir.resolve(suiteName + "-results.xml");
        String filter = suiteName + ".*";

        log.info("Running test suite: {}", suiteName);
        Libnd4jTestHelper.TestResult result = Libnd4jTestHelper.runTest(filter, xmlPath);

        // Parse results for detailed reporting
        if (Files.exists(xmlPath)) {
            GTestResultParser.GTestResults gtestResults = GTestResultParser.parseXmlFile(xmlPath);
            log.info("{}: {} tests, {} passed, {} failed in {:.3f}s",
                suiteName,
                gtestResults.getTotalTestCount(),
                gtestResults.getTotalTestCount() - gtestResults.getTotalFailureCount(),
                gtestResults.getTotalFailureCount(),
                result.getDurationMs() / 1000.0);
        }

        assertTrue(result.isPassed(),
            () -> buildFailureMessage(suiteName, result, xmlPath));
    }

    /**
     * Run a single test case.
     */
    private void runSingleTest(String suiteName, String testCase) {
        String filter = suiteName + "." + testCase;
        Path xmlPath = tempXmlDir.resolve(suiteName + "_" + testCase + "-results.xml");

        log.debug("Running test: {}", filter);
        Libnd4jTestHelper.TestResult result = Libnd4jTestHelper.runTest(filter, xmlPath);

        log.debug("{}: {} in {}ms", filter, result.isPassed() ? "PASSED" : "FAILED", result.getDurationMs());

        assertTrue(result.isPassed(),
            () -> buildFailureMessage(filter, result, xmlPath));
    }

    /**
     * Build a detailed failure message from test results.
     */
    private String buildFailureMessage(String testName, Libnd4jTestHelper.TestResult result, Path xmlPath) {
        StringBuilder msg = new StringBuilder();
        msg.append("Native test failed: ").append(testName).append("\n");
        msg.append("Exit code: ").append(result.getExitCode()).append("\n");
        msg.append("Duration: ").append(result.getDurationMs()).append("ms\n");

        // Try to get details from XML
        if (Files.exists(xmlPath)) {
            GTestResultParser.GTestResults gtestResults = GTestResultParser.parseXmlFile(xmlPath);
            List<GTestResultParser.GTestCase> failures = gtestResults.getAllFailedTests();
            if (!failures.isEmpty()) {
                msg.append("\nFailure details:\n");
                for (GTestResultParser.GTestCase failure : failures) {
                    msg.append("  - ").append(failure.getFullName()).append("\n");
                    if (failure.getFile() != null && !failure.getFile().isEmpty()) {
                        msg.append("    at ").append(failure.getFile())
                           .append(":").append(failure.getLine()).append("\n");
                    }
                    if (failure.getFailureMessage() != null) {
                        msg.append("    Message: ").append(failure.getFailureMessage()).append("\n");
                    }
                    if (failure.getFailureDetails() != null && !failure.getFailureDetails().isEmpty()) {
                        msg.append("    Details: ").append(failure.getFailureDetails().trim()).append("\n");
                    }
                }
            }
        }

        // Add stdout excerpt (last 50 lines)
        String stdout = result.getStdout();
        if (stdout != null && !stdout.isEmpty()) {
            String[] lines = stdout.split("\n");
            int startLine = Math.max(0, lines.length - 50);
            msg.append("\nOutput (last ").append(lines.length - startLine).append(" lines):\n");
            for (int i = startLine; i < lines.length; i++) {
                msg.append(lines[i]).append("\n");
            }
        }

        return msg.toString();
    }

    /**
     * Utility test to verify test infrastructure setup.
     */
    @Test
    @DisplayName("Verify Test Infrastructure")
    void verifyTestInfrastructure() {
        Path home = Libnd4jTestHelper.getLibnd4jHome();
        log.info("libnd4j home: {}", home);
        log.info("Chip: {}", Libnd4jTestHelper.getChip());
        log.info("Test executable: {}", Libnd4jTestHelper.getTestExecutable());
        log.info("Tests available: {}", nativeTestsAvailable);
        log.info("Testing enabled: {}", Libnd4jTestHelper.isNativeTestingEnabled());

        if (home != null) {
            assertTrue(Files.exists(home), "libnd4j home should exist");
        }

        if (nativeTestsAvailable) {
            Path executable = Libnd4jTestHelper.getTestExecutable();
            assertNotNull(executable, "Executable should not be null when tests are available");
            assertTrue(Files.exists(executable), "Executable should exist");
            assertTrue(Files.isExecutable(executable), "Should be executable");
        }
    }

    /**
     * Test that lists all available test suites (useful for debugging).
     */
    @Test
    @DisplayName("List Available Test Suites")
    @EnabledIf("nativeTestsAreAvailable")
    void listAvailableTestSuites() {
        List<String> suites = Libnd4jTestHelper.discoverTestSuites();
        assertFalse(suites.isEmpty(), "Should discover at least one test suite");

        log.info("Discovered {} test suites:", suites.size());
        for (String suite : suites) {
            List<String> cases = Libnd4jTestHelper.discoverTestCases(suite);
            log.info("  {} ({} tests)", suite, cases.size());
        }
    }
}
