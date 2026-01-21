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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Helper class for discovering, building, and executing libnd4j C++ tests.
 * This integrates the libnd4j Google Test suite with Java JUnit tests.
 */
public class Libnd4jTestHelper {
    private static final Logger log = LoggerFactory.getLogger(Libnd4jTestHelper.class);

    // System property names for configuration
    public static final String LIBND4J_HOME_PROPERTY = "org.nd4j.libnd4j.home";
    public static final String LIBND4J_CHIP_PROPERTY = "org.nd4j.libnd4j.chip";
    public static final String LIBND4J_BUILD_DIR_PROPERTY = "org.nd4j.libnd4j.builddir";
    public static final String LIBND4J_TEST_FILTER_PROPERTY = "org.nd4j.libnd4j.test.filter";
    public static final String LIBND4J_TEST_TIMEOUT_PROPERTY = "org.nd4j.libnd4j.test.timeout";
    public static final String LIBND4J_AUTO_BUILD_PROPERTY = "org.nd4j.libnd4j.autobuild";
    public static final String LIBND4J_TEST_ENABLED_PROPERTY = "org.nd4j.libnd4j.test.enabled";

    // Default values
    private static final String DEFAULT_CHIP = "cpu";
    private static final long DEFAULT_TEST_TIMEOUT_MINUTES = 30;

    // Cached paths
    private static volatile Path libnd4jHome;
    private static volatile Path testExecutable;
    private static volatile List<String> availableTestSuites;

    /**
     * Check if native tests are enabled.
     */
    public static boolean isNativeTestingEnabled() {
        return Boolean.parseBoolean(System.getProperty(LIBND4J_TEST_ENABLED_PROPERTY, "true"));
    }

    /**
     * Get the libnd4j home directory.
     */
    public static Path getLibnd4jHome() {
        if (libnd4jHome == null) {
            synchronized (Libnd4jTestHelper.class) {
                if (libnd4jHome == null) {
                    libnd4jHome = resolveLibnd4jHome();
                }
            }
        }
        return libnd4jHome;
    }

    /**
     * Resolve libnd4j home from various sources.
     */
    private static Path resolveLibnd4jHome() {
        // First, check system property
        String homeFromProperty = System.getProperty(LIBND4J_HOME_PROPERTY);
        if (homeFromProperty != null && !homeFromProperty.isEmpty()) {
            Path path = Paths.get(homeFromProperty);
            if (Files.exists(path)) {
                log.info("Using libnd4j home from system property: {}", path);
                return path;
            }
        }

        // Second, check environment variable
        String homeFromEnv = System.getenv("LIBND4J_HOME");
        if (homeFromEnv != null && !homeFromEnv.isEmpty()) {
            Path path = Paths.get(homeFromEnv);
            if (Files.exists(path)) {
                log.info("Using libnd4j home from environment variable: {}", path);
                return path;
            }
        }

        // Third, try to find relative to current working directory (project structure)
        Path currentDir = Paths.get(System.getProperty("user.dir"));

        // Try various relative locations
        List<String> relativePathsToTry = Arrays.asList(
            "libnd4j",
            "../libnd4j",
            "../../libnd4j",
            "../../../libnd4j"
        );

        for (String relativePath : relativePathsToTry) {
            Path candidatePath = currentDir.resolve(relativePath).normalize();
            if (isValidLibnd4jHome(candidatePath)) {
                log.info("Found libnd4j home at: {}", candidatePath);
                return candidatePath;
            }
        }

        // Fourth, try to find from the project root (going up from platform-tests)
        Path projectRoot = findProjectRoot(currentDir);
        if (projectRoot != null) {
            Path libnd4jPath = projectRoot.resolve("libnd4j");
            if (isValidLibnd4jHome(libnd4jPath)) {
                log.info("Found libnd4j home from project root: {}", libnd4jPath);
                return libnd4jPath;
            }
        }

        log.warn("Could not find libnd4j home directory. Set {} system property or LIBND4J_HOME environment variable.",
                LIBND4J_HOME_PROPERTY);
        return null;
    }

    /**
     * Find the project root by looking for pom.xml with deeplearning4j-parent.
     */
    private static Path findProjectRoot(Path startDir) {
        Path current = startDir;
        while (current != null) {
            Path pomPath = current.resolve("pom.xml");
            if (Files.exists(pomPath)) {
                try {
                    String content = new String(Files.readAllBytes(pomPath));
                    if (content.contains("<artifactId>deeplearning4j</artifactId>") ||
                        content.contains("<artifactId>deeplearning4j-parent</artifactId>")) {
                        return current;
                    }
                } catch (IOException e) {
                    // Ignore and continue
                }
            }
            current = current.getParent();
        }
        return null;
    }

    /**
     * Validate that a path is a valid libnd4j home directory.
     */
    private static boolean isValidLibnd4jHome(Path path) {
        if (!Files.exists(path) || !Files.isDirectory(path)) {
            return false;
        }
        // Check for characteristic files/directories
        return Files.exists(path.resolve("tests_cpu")) &&
               Files.exists(path.resolve("include")) &&
               Files.exists(path.resolve("CMakeLists.txt"));
    }

    /**
     * Get the chip type (cpu or cuda).
     */
    public static String getChip() {
        return System.getProperty(LIBND4J_CHIP_PROPERTY, DEFAULT_CHIP).toLowerCase();
    }

    /**
     * Get the path to the test executable.
     */
    public static Path getTestExecutable() {
        if (testExecutable == null) {
            synchronized (Libnd4jTestHelper.class) {
                if (testExecutable == null) {
                    testExecutable = resolveTestExecutable();
                }
            }
        }
        return testExecutable;
    }

    /**
     * Resolve the test executable path.
     */
    private static Path resolveTestExecutable() {
        Path home = getLibnd4jHome();
        if (home == null) {
            return null;
        }

        String chip = getChip();

        // Check custom build directory
        String customBuildDir = System.getProperty(LIBND4J_BUILD_DIR_PROPERTY);
        if (customBuildDir != null && !customBuildDir.isEmpty()) {
            Path customPath = Paths.get(customBuildDir).resolve("runtests");
            if (isExecutable(customPath)) {
                log.info("Using test executable from custom build dir: {}", customPath);
                return customPath;
            }
            // Also check with .exe extension on Windows
            customPath = Paths.get(customBuildDir).resolve("runtests.exe");
            if (isExecutable(customPath)) {
                return customPath;
            }
        }

        // Standard build locations
        List<Path> possibleLocations = Arrays.asList(
            home.resolve("blasbuild").resolve(chip).resolve("tests_cpu").resolve("layers_tests").resolve("runtests"),
            home.resolve("blasbuild").resolve(chip).resolve("tests_cpu").resolve("layers_tests").resolve("runtests.exe"),
            home.resolve("build").resolve("tests_cpu").resolve("layers_tests").resolve("runtests"),
            home.resolve("build").resolve("tests_cpu").resolve("layers_tests").resolve("runtests.exe"),
            home.resolve("cmake-build-debug").resolve("tests_cpu").resolve("layers_tests").resolve("runtests"),
            home.resolve("cmake-build-release").resolve("tests_cpu").resolve("layers_tests").resolve("runtests")
        );

        for (Path path : possibleLocations) {
            if (isExecutable(path)) {
                log.info("Found test executable at: {}", path);
                return path;
            }
        }

        log.warn("Test executable not found. Build libnd4j tests first or set {} system property.",
                LIBND4J_BUILD_DIR_PROPERTY);
        return null;
    }

    /**
     * Check if a path is an executable file.
     */
    private static boolean isExecutable(Path path) {
        return Files.exists(path) && Files.isRegularFile(path) && Files.isExecutable(path);
    }

    /**
     * Check if native tests are available (executable exists).
     */
    public static boolean isNativeTestsAvailable() {
        return getTestExecutable() != null;
    }

    /**
     * Get the test timeout in minutes.
     */
    public static long getTestTimeoutMinutes() {
        String timeoutStr = System.getProperty(LIBND4J_TEST_TIMEOUT_PROPERTY);
        if (timeoutStr != null) {
            try {
                return Long.parseLong(timeoutStr);
            } catch (NumberFormatException e) {
                log.warn("Invalid test timeout value: {}, using default", timeoutStr);
            }
        }
        return DEFAULT_TEST_TIMEOUT_MINUTES;
    }

    /**
     * Get the test filter pattern (for --gtest_filter).
     */
    public static String getTestFilter() {
        return System.getProperty(LIBND4J_TEST_FILTER_PROPERTY, "*");
    }

    /**
     * Discover available test suites by querying the test executable.
     */
    public static List<String> discoverTestSuites() {
        if (availableTestSuites != null) {
            return availableTestSuites;
        }

        synchronized (Libnd4jTestHelper.class) {
            if (availableTestSuites != null) {
                return availableTestSuites;
            }

            Path executable = getTestExecutable();
            if (executable == null) {
                log.warn("Cannot discover test suites: executable not found");
                return Collections.emptyList();
            }

            try {
                ProcessBuilder pb = new ProcessBuilder(
                    executable.toString(),
                    "--gtest_list_tests"
                );
                pb.directory(executable.getParent().toFile());
                pb.redirectErrorStream(true);

                Process process = pb.start();
                List<String> suites = new ArrayList<>();
                String currentSuite = null;

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String line;
                    Pattern suitePattern = Pattern.compile("^(\\w+)\\.$");

                    while ((line = reader.readLine()) != null) {
                        Matcher matcher = suitePattern.matcher(line);
                        if (matcher.matches()) {
                            currentSuite = matcher.group(1);
                            if (!suites.contains(currentSuite)) {
                                suites.add(currentSuite);
                            }
                        }
                    }
                }

                boolean completed = process.waitFor(60, TimeUnit.SECONDS);
                if (!completed) {
                    process.destroyForcibly();
                    log.warn("Test discovery timed out");
                }

                availableTestSuites = Collections.unmodifiableList(suites);
                log.info("Discovered {} test suites", suites.size());
                return availableTestSuites;

            } catch (Exception e) {
                log.error("Failed to discover test suites", e);
                return Collections.emptyList();
            }
        }
    }

    /**
     * Discover individual test cases within a test suite.
     */
    public static List<String> discoverTestCases(String suiteName) {
        Path executable = getTestExecutable();
        if (executable == null) {
            return Collections.emptyList();
        }

        try {
            ProcessBuilder pb = new ProcessBuilder(
                executable.toString(),
                "--gtest_list_tests",
                "--gtest_filter=" + suiteName + ".*"
            );
            pb.directory(executable.getParent().toFile());
            pb.redirectErrorStream(true);

            Process process = pb.start();
            List<String> testCases = new ArrayList<>();
            boolean inSuite = false;

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith(suiteName + ".")) {
                        inSuite = true;
                    } else if (inSuite && line.startsWith("  ")) {
                        String testName = line.trim();
                        // Remove any parameterized test suffix or comment
                        int slashIndex = testName.indexOf('/');
                        if (slashIndex > 0) {
                            testName = testName.substring(0, slashIndex);
                        }
                        int hashIndex = testName.indexOf('#');
                        if (hashIndex > 0) {
                            testName = testName.substring(0, hashIndex).trim();
                        }
                        if (!testName.isEmpty() && !testCases.contains(testName)) {
                            testCases.add(testName);
                        }
                    } else if (!line.startsWith(" ") && !line.isEmpty()) {
                        inSuite = false;
                    }
                }
            }

            process.waitFor(30, TimeUnit.SECONDS);
            return testCases;

        } catch (Exception e) {
            log.error("Failed to discover test cases for suite: {}", suiteName, e);
            return Collections.emptyList();
        }
    }

    /**
     * Build the test executable if auto-build is enabled.
     */
    public static boolean buildTestsIfNeeded() {
        if (!Boolean.parseBoolean(System.getProperty(LIBND4J_AUTO_BUILD_PROPERTY, "false"))) {
            return isNativeTestsAvailable();
        }

        if (isNativeTestsAvailable()) {
            return true;
        }

        log.info("Attempting to build libnd4j tests...");
        Path home = getLibnd4jHome();
        if (home == null) {
            log.error("Cannot build: libnd4j home not found");
            return false;
        }

        try {
            String chip = getChip();
            Path buildDir = home.resolve("blasbuild").resolve(chip);
            Files.createDirectories(buildDir);

            // Run CMake configure
            ProcessBuilder cmakePb = new ProcessBuilder(
                "cmake",
                "-B", buildDir.toString(),
                "-S", home.toString(),
                "-DBUILD_TESTS=ON",
                "-DSD_CPU=ON"
            );
            cmakePb.directory(home.toFile());
            cmakePb.inheritIO();

            Process cmakeProcess = cmakePb.start();
            int cmakeResult = cmakeProcess.waitFor();
            if (cmakeResult != 0) {
                log.error("CMake configure failed with exit code: {}", cmakeResult);
                return false;
            }

            // Run build
            ProcessBuilder buildPb = new ProcessBuilder(
                "cmake",
                "--build", buildDir.toString(),
                "--target", "runtests",
                "-j", String.valueOf(Runtime.getRuntime().availableProcessors())
            );
            buildPb.directory(home.toFile());
            buildPb.inheritIO();

            Process buildProcess = buildPb.start();
            int buildResult = buildProcess.waitFor();
            if (buildResult != 0) {
                log.error("Build failed with exit code: {}", buildResult);
                return false;
            }

            // Invalidate cached executable path
            testExecutable = null;
            return isNativeTestsAvailable();

        } catch (Exception e) {
            log.error("Failed to build libnd4j tests", e);
            return false;
        }
    }

    /**
     * Run a specific test or test suite.
     *
     * @param testFilter The gtest filter pattern (e.g., "ArrayOptionsTests.*" or "ArrayOptionsTests.test1")
     * @param outputXmlPath Optional path for XML output
     * @return TestResult containing exit code, output, and duration
     */
    public static TestResult runTest(String testFilter, Path outputXmlPath) {
        Path executable = getTestExecutable();
        if (executable == null) {
            return new TestResult(-1, "Test executable not found", "", 0, false);
        }

        long startTime = System.currentTimeMillis();

        try {
            List<String> command = new ArrayList<>();
            command.add(executable.toString());
            command.add("--gtest_filter=" + testFilter);
            command.add("--gtest_color=no");

            if (outputXmlPath != null) {
                command.add("--gtest_output=xml:" + outputXmlPath.toString());
            }

            ProcessBuilder pb = new ProcessBuilder(command);
            pb.directory(executable.getParent().toFile());

            // Set up environment
            Map<String, String> env = pb.environment();
            setupTestEnvironment(env);

            pb.redirectErrorStream(true);

            Process process = pb.start();
            StringBuilder output = new StringBuilder();
            StringBuilder errorOutput = new StringBuilder();

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                    // Log progress for long-running tests
                    if (line.contains("[       OK ]") || line.contains("[  FAILED  ]")) {
                        log.debug("{}", line);
                    }
                }
            }

            long timeoutMinutes = getTestTimeoutMinutes();
            boolean completed = process.waitFor(timeoutMinutes, TimeUnit.MINUTES);

            if (!completed) {
                process.destroyForcibly();
                long duration = System.currentTimeMillis() - startTime;
                return new TestResult(-1, output.toString(),
                    "Test timed out after " + timeoutMinutes + " minutes", duration, false);
            }

            int exitCode = process.exitValue();
            long duration = System.currentTimeMillis() - startTime;

            return new TestResult(exitCode, output.toString(), errorOutput.toString(),
                duration, exitCode == 0);

        } catch (Exception e) {
            long duration = System.currentTimeMillis() - startTime;
            log.error("Error running test: {}", testFilter, e);
            return new TestResult(-1, "", e.getMessage(), duration, false);
        }
    }

    /**
     * Set up environment variables for test execution.
     */
    private static void setupTestEnvironment(Map<String, String> env) {
        // CUDA environment settings
        env.put("CUDA_LAUNCH_BLOCKING", "1");
        env.put("OMP_NUM_THREADS", "1");

        // Grid and block size settings for reproducibility
        env.put("GRID_SIZE_TRANSFORM_SCAN", "1");
        env.put("BLOCK_SIZE_TRANSFORM_SCAN", "1");
        env.put("BLOCK_SIZE_SCALAR_SCAN", "1");
        env.put("GRID_SIZE_SCALAR_SCAN", "1");

        // Mac-specific settings
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("mac")) {
            env.put("DYLD_LIBRARY_PATH", "/usr/local/lib/gcc/8/:/usr/local/lib/gcc/7/:" +
                env.getOrDefault("DYLD_LIBRARY_PATH", ""));
        }
    }

    /**
     * Get the path to the test resources directory.
     */
    public static Path getTestResourcesPath() {
        Path home = getLibnd4jHome();
        if (home == null) {
            return null;
        }
        return home.resolve("tests_cpu").resolve("resources");
    }

    /**
     * Result class for test execution.
     */
    public static class TestResult {
        private final int exitCode;
        private final String stdout;
        private final String stderr;
        private final long durationMs;
        private final boolean passed;

        public TestResult(int exitCode, String stdout, String stderr, long durationMs, boolean passed) {
            this.exitCode = exitCode;
            this.stdout = stdout;
            this.stderr = stderr;
            this.durationMs = durationMs;
            this.passed = passed;
        }

        public int getExitCode() { return exitCode; }
        public String getStdout() { return stdout; }
        public String getStderr() { return stderr; }
        public long getDurationMs() { return durationMs; }
        public boolean isPassed() { return passed; }

        public String getFailureMessage() {
            if (passed) {
                return null;
            }
            StringBuilder msg = new StringBuilder();
            msg.append("Test failed with exit code: ").append(exitCode).append("\n");
            if (stderr != null && !stderr.isEmpty()) {
                msg.append("Error: ").append(stderr).append("\n");
            }
            // Extract failure details from stdout
            if (stdout != null) {
                for (String line : stdout.split("\n")) {
                    if (line.contains("[  FAILED  ]") || line.contains("Failure") ||
                        line.contains("Expected:") || line.contains("Actual:")) {
                        msg.append(line).append("\n");
                    }
                }
            }
            return msg.toString();
        }
    }
}
