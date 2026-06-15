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
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Helper class for building libnd4j native code from Java tests.
 * Provides utilities for CMake configuration and building.
 */
public class NativeBuildHelper {
    private static final Logger log = LoggerFactory.getLogger(NativeBuildHelper.class);

    // System properties for build configuration
    public static final String CMAKE_PATH_PROPERTY = "org.nd4j.cmake.path";
    public static final String CMAKE_GENERATOR_PROPERTY = "org.nd4j.cmake.generator";
    public static final String BUILD_PARALLEL_JOBS_PROPERTY = "org.nd4j.build.jobs";
    public static final String BUILD_TYPE_PROPERTY = "org.nd4j.build.type";

    /**
     * Check if CMake is available on the system.
     */
    public static boolean isCmakeAvailable() {
        String cmakePath = getCmakePath();
        try {
            ProcessBuilder pb = new ProcessBuilder(cmakePath, "--version");
            Process process = pb.start();
            boolean completed = process.waitFor(10, TimeUnit.SECONDS);
            return completed && process.exitValue() == 0;
        } catch (Exception e) {
            log.debug("CMake not available: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Get the CMake executable path.
     */
    public static String getCmakePath() {
        String path = System.getProperty(CMAKE_PATH_PROPERTY);
        if (path != null && !path.isEmpty()) {
            return path;
        }
        // Try common locations
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) {
            // Check Program Files
            List<String> windowsPaths = Arrays.asList(
                "C:\\Program Files\\CMake\\bin\\cmake.exe",
                "C:\\Program Files (x86)\\CMake\\bin\\cmake.exe",
                "cmake.exe"
            );
            for (String p : windowsPaths) {
                if (Files.exists(Path.of(p))) {
                    return p;
                }
            }
        }
        return "cmake";
    }

    /**
     * Get the CMake generator to use.
     */
    public static String getCmakeGenerator() {
        String generator = System.getProperty(CMAKE_GENERATOR_PROPERTY);
        if (generator != null && !generator.isEmpty()) {
            return generator;
        }
        // Default based on platform
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) {
            return "Visual Studio 17 2022";
        }
        return "Unix Makefiles";
    }

    /**
     * Get the number of parallel build jobs.
     */
    public static int getBuildJobs() {
        String jobs = System.getProperty(BUILD_PARALLEL_JOBS_PROPERTY);
        if (jobs != null) {
            try {
                return Integer.parseInt(jobs);
            } catch (NumberFormatException e) {
                log.debug("Invalid build jobs value: {}", jobs);
            }
        }
        return Runtime.getRuntime().availableProcessors();
    }

    /**
     * Get the build type (Release, Debug, RelWithDebInfo).
     */
    public static String getBuildType() {
        return System.getProperty(BUILD_TYPE_PROPERTY, "Release");
    }

    /**
     * Build configuration options.
     */
    public static class BuildConfig {
        private Path sourceDir;
        private Path buildDir;
        private String chip = "cpu";
        private String buildType = "Release";
        private boolean buildTests = true;
        private boolean verbose = false;
        private List<String> extraCmakeArgs = new ArrayList<>();
        private long timeoutMinutes = 60;

        public BuildConfig sourceDir(Path sourceDir) {
            this.sourceDir = sourceDir;
            return this;
        }

        public BuildConfig buildDir(Path buildDir) {
            this.buildDir = buildDir;
            return this;
        }

        public BuildConfig chip(String chip) {
            this.chip = chip;
            return this;
        }

        public BuildConfig buildType(String buildType) {
            this.buildType = buildType;
            return this;
        }

        public BuildConfig buildTests(boolean buildTests) {
            this.buildTests = buildTests;
            return this;
        }

        public BuildConfig verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }

        public BuildConfig addCmakeArg(String arg) {
            this.extraCmakeArgs.add(arg);
            return this;
        }

        public BuildConfig timeoutMinutes(long minutes) {
            this.timeoutMinutes = minutes;
            return this;
        }

        public Path getSourceDir() { return sourceDir; }
        public Path getBuildDir() { return buildDir; }
        public String getChip() { return chip; }
        public String getBuildType() { return buildType; }
        public boolean isBuildTests() { return buildTests; }
        public boolean isVerbose() { return verbose; }
        public List<String> getExtraCmakeArgs() { return extraCmakeArgs; }
        public long getTimeoutMinutes() { return timeoutMinutes; }
    }

    /**
     * Build result information.
     */
    public static class BuildResult {
        private final boolean success;
        private final String output;
        private final String errorOutput;
        private final long durationMs;
        private final int exitCode;

        public BuildResult(boolean success, String output, String errorOutput, long durationMs, int exitCode) {
            this.success = success;
            this.output = output;
            this.errorOutput = errorOutput;
            this.durationMs = durationMs;
            this.exitCode = exitCode;
        }

        public boolean isSuccess() { return success; }
        public String getOutput() { return output; }
        public String getErrorOutput() { return errorOutput; }
        public long getDurationMs() { return durationMs; }
        public int getExitCode() { return exitCode; }
    }

    /**
     * Configure and build libnd4j with tests.
     */
    public static BuildResult buildLibnd4j(BuildConfig config) {
        if (!isCmakeAvailable()) {
            return new BuildResult(false, "", "CMake not available", 0, -1);
        }

        try {
            // Create build directory
            Path buildDir = config.getBuildDir();
            if (buildDir == null) {
                buildDir = config.getSourceDir().resolve("blasbuild").resolve(config.getChip());
            }
            Files.createDirectories(buildDir);

            long startTime = System.currentTimeMillis();

            // Run CMake configure
            BuildResult configResult = runCmakeConfigure(config, buildDir);
            if (!configResult.isSuccess()) {
                return configResult;
            }

            // Run build
            BuildResult buildResult = runCmakeBuild(config, buildDir);
            long duration = System.currentTimeMillis() - startTime;

            if (buildResult.isSuccess()) {
                log.info("Build completed successfully in {}s", duration / 1000);
            } else {
                log.error("Build failed after {}s", duration / 1000);
            }

            return new BuildResult(buildResult.isSuccess(),
                configResult.getOutput() + "\n" + buildResult.getOutput(),
                configResult.getErrorOutput() + "\n" + buildResult.getErrorOutput(),
                duration,
                buildResult.getExitCode());

        } catch (Exception e) {
            log.error("Build failed with exception", e);
            return new BuildResult(false, "", e.getMessage(), 0, -1);
        }
    }

    /**
     * Run CMake configure step.
     */
    private static BuildResult runCmakeConfigure(BuildConfig config, Path buildDir) throws IOException, InterruptedException {
        List<String> command = new ArrayList<>();
        command.add(getCmakePath());
        command.add("-B");
        command.add(buildDir.toString());
        command.add("-S");
        command.add(config.getSourceDir().toString());
        command.add("-DCMAKE_BUILD_TYPE=" + config.getBuildType());

        // Platform-specific options
        if ("cpu".equalsIgnoreCase(config.getChip())) {
            command.add("-DSD_CPU=ON");
            command.add("-DSD_CUDA=OFF");
        } else if ("cuda".equalsIgnoreCase(config.getChip())) {
            command.add("-DSD_CPU=OFF");
            command.add("-DSD_CUDA=ON");
        }

        // Test build option
        if (config.isBuildTests()) {
            command.add("-DBUILD_TESTS=ON");
        }

        // Add extra CMake arguments
        command.addAll(config.getExtraCmakeArgs());

        log.info("Running CMake configure: {}", String.join(" ", command));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(config.getSourceDir().toFile());
        pb.redirectErrorStream(true);

        Process process = pb.start();
        StringBuilder output = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                if (config.isVerbose()) {
                    log.info("{}", line);
                }
            }
        }

        boolean completed = process.waitFor(config.getTimeoutMinutes(), TimeUnit.MINUTES);
        if (!completed) {
            process.destroyForcibly();
            return new BuildResult(false, output.toString(), "Configure timed out", 0, -1);
        }

        return new BuildResult(process.exitValue() == 0, output.toString(), "", 0, process.exitValue());
    }

    /**
     * Run CMake build step.
     */
    private static BuildResult runCmakeBuild(BuildConfig config, Path buildDir) throws IOException, InterruptedException {
        List<String> command = new ArrayList<>();
        command.add(getCmakePath());
        command.add("--build");
        command.add(buildDir.toString());
        command.add("--config");
        command.add(config.getBuildType());

        if (config.isBuildTests()) {
            command.add("--target");
            command.add("runtests");
        }

        command.add("-j");
        command.add(String.valueOf(getBuildJobs()));

        if (config.isVerbose()) {
            command.add("--verbose");
        }

        log.info("Running CMake build: {}", String.join(" ", command));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(config.getSourceDir().toFile());
        pb.redirectErrorStream(true);

        Process process = pb.start();
        StringBuilder output = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                if (config.isVerbose()) {
                    log.info("{}", line);
                }
            }
        }

        boolean completed = process.waitFor(config.getTimeoutMinutes(), TimeUnit.MINUTES);
        if (!completed) {
            process.destroyForcibly();
            return new BuildResult(false, output.toString(), "Build timed out", 0, -1);
        }

        return new BuildResult(process.exitValue() == 0, output.toString(), "", 0, process.exitValue());
    }

    /**
     * Clean the build directory.
     */
    public static boolean cleanBuildDir(Path buildDir) {
        if (!Files.exists(buildDir)) {
            return true;
        }

        try {
            Files.walk(buildDir)
                .sorted((a, b) -> -a.compareTo(b))
                .forEach(path -> {
                    try {
                        Files.deleteIfExists(path);
                    } catch (IOException e) {
                        log.debug("Could not delete: {}", path);
                    }
                });
            return true;
        } catch (IOException e) {
            log.error("Failed to clean build directory", e);
            return false;
        }
    }

    /**
     * Get the path to the built test executable.
     */
    public static Path getTestExecutablePath(Path buildDir, String chip) {
        List<Path> possiblePaths = Arrays.asList(
            buildDir.resolve("tests_cpu").resolve("layers_tests").resolve("runtests"),
            buildDir.resolve("tests_cpu").resolve("layers_tests").resolve("runtests.exe"),
            buildDir.resolve("tests_cpu").resolve("layers_tests").resolve("Release").resolve("runtests.exe"),
            buildDir.resolve("tests_cpu").resolve("layers_tests").resolve("Debug").resolve("runtests.exe")
        );

        for (Path path : possiblePaths) {
            if (Files.exists(path) && Files.isExecutable(path)) {
                return path;
            }
        }

        return null;
    }
}
