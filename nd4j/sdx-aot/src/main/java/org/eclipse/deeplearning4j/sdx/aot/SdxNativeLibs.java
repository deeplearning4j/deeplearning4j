/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.sdx.aot;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Side-loads the JavaCPP/ND4J/tokenizers native libraries for the AOT images.
 *
 * <p>The images deliberately do NOT embed the native binaries (they are excluded
 * via {@code -H:ExcludeResources}, the kompile pattern — see ADR 0109): the SDK
 * package ships them in {@code lib/} next to the executable, and this resolver
 * points JavaCPP at them before any ND4J/tokenizer class initializes.</p>
 *
 * <p>Resolution order in a native image:</p>
 * <ol>
 *   <li>{@code SDX_NATIVE_LIB_DIR} env var — explicit override</li>
 *   <li>{@code <binary-dir>/lib/} and {@code <binary-dir>/../lib/} — SDK layout
 *       ({@code bin/sdx-llm} + {@code lib/*.so})</li>
 *   <li>{@code ~/.javacpp/cache/} — populated by prior JVM-mode runs</li>
 * </ol>
 *
 * <p>On a JVM with platform-classifier jars on the classpath this is a no-op —
 * JavaCPP's Loader extracts from the classpath as usual.</p>
 */
final class SdxNativeLibs {

    private static final String PLATFORM = detectPlatform();
    private static volatile boolean bootstrapped = false;

    private SdxNativeLibs() {
    }

    /** Idempotent; call before any ND4J / tokenizer class initialization. */
    static synchronized void bootstrap() {
        if (bootstrapped) {
            return;
        }
        bootstrapped = true;

        if (isNativeImage()) {
            // Classpath scanning cannot run in an image: point ClassGraphHolder at
            // the scan baked by SdxClassGraphScanBaker (respect a user override).
            if (System.getProperty(org.nd4j.common.config.ND4JSystemProperties.CLASS_GRAPH_SCAN_RESOURCES) == null) {
                System.setProperty(org.nd4j.common.config.ND4JSystemProperties.CLASS_GRAPH_SCAN_RESOURCES,
                        "sdx-classgraph-scan.json");
            }
            // AWT (ImageIO/PDF rendering) runs headless in the image; the AWT/jsound
            // JDK natives are side-loaded from lib/ like everything else.
            if (System.getProperty("java.awt.headless") == null) {
                System.setProperty("java.awt.headless", "true");
            }
        }

        if (!isNativeImage() && hasClassifierJarsOnClasspath()) {
            return; // JVM dev mode — classpath extraction handles it
        }

        List<Path> libDirs = resolve();
        if (libDirs.isEmpty()) {
            System.err.println("sdx-llm: warning: no native libraries found. Options:\n"
                    + "  1. Set SDX_NATIVE_LIB_DIR=/path/to/libs\n"
                    + "  2. Keep the SDK layout: bin/" + executableName() + " next to lib/\n"
                    + "  3. Run once in JVM mode to populate ~/.javacpp/cache/");
            return;
        }
        configureJavaCpp(libDirs);
    }

    private static List<Path> resolve() {
        String envDir = System.getenv("SDX_NATIVE_LIB_DIR");
        if (envDir != null && !envDir.isBlank()) {
            Path envPath = Paths.get(envDir);
            List<Path> fromEnv = libDirsUnder(envPath);
            if (!fromEnv.isEmpty()) {
                return fromEnv;
            }
            System.err.println("sdx-llm: warning: SDX_NATIVE_LIB_DIR=" + envPath
                    + " contains no native libraries");
        }

        Path binaryDir = getBinaryDirectory();
        if (binaryDir != null) {
            List<Path> adjacent = libDirsUnder(binaryDir.resolve("lib"));
            if (!adjacent.isEmpty()) {
                return adjacent;
            }
            Path parent = binaryDir.getParent();
            if (parent != null) {
                List<Path> dist = libDirsUnder(parent.resolve("lib"));
                if (!dist.isEmpty()) {
                    return dist;
                }
            }
        }

        String home = System.getProperty("user.home");
        if (home != null) {
            List<Path> cached = libDirsUnder(Paths.get(home, ".javacpp", "cache"));
            if (!cached.isEmpty()) {
                return cached;
            }
        }
        return List.of();
    }

    /**
     * Every directory under {@code root} (including root itself, up to a shallow
     * depth) that directly contains a native library. Handles both the flat SDK
     * {@code lib/} layout and nested jar-path layouts.
     */
    private static List<Path> libDirsUnder(Path root) {
        if (root == null || !Files.isDirectory(root)) {
            return List.of();
        }
        Set<Path> dirs = new LinkedHashSet<>();
        try {
            Files.walkFileTree(root, Set.of(), 8, new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                    if (isNativeLib(file.getFileName().toString())) {
                        // For nested per-platform layouts only pick dirs for this platform
                        String p = file.toString();
                        if (p.contains(PLATFORM) || file.getParent().equals(root)
                                || !p.matches(".*(linux|windows|macosx|android|ios)-[a-z0-9_]+.*")) {
                            dirs.add(file.getParent());
                        }
                    }
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFileFailed(Path file, IOException exc) {
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException ignored) {
            // treat as not found
        }
        return new ArrayList<>(dirs);
    }

    /** Same property set kompile's NativeLibraryResolver uses — proven with ND4J. */
    private static void configureJavaCpp(List<Path> libDirs) {
        System.setProperty("org.bytedeco.javacpp.pathsFirst", "true");

        String pathStr = libDirs.stream()
                .map(p -> p.toAbsolutePath().toString())
                .collect(Collectors.joining(File.pathSeparator));
        String existing = System.getProperty("java.library.path", "");
        if (!existing.contains(pathStr)) {
            System.setProperty("java.library.path",
                    pathStr + (existing.isEmpty() ? "" : File.pathSeparator + existing));
        }
        if (libDirs.size() == 1) {
            System.setProperty("org.bytedeco.javacpp.cachedir",
                    libDirs.get(0).toAbsolutePath().toString());
        }
    }

    private static boolean isNativeLib(String name) {
        return name.endsWith(".so") || name.contains(".so.")
                || name.endsWith(".dylib") || name.endsWith(".dll");
    }

    private static boolean isNativeImage() {
        return System.getProperty("org.graalvm.nativeimage.imagecode") != null;
    }

    private static boolean hasClassifierJarsOnClasspath() {
        return System.getProperty("java.class.path", "").contains(PLATFORM + ".jar");
    }

    private static Path getBinaryDirectory() {
        if (isNativeImage()) {
            // ProcessProperties is part of the SVM runtime; reflection keeps this
            // class loadable on a plain JVM where the API jar (provided) is absent.
            try {
                Class<?> pp = Class.forName("org.graalvm.nativeimage.ProcessProperties");
                Object path = pp.getMethod("getExecutableName").invoke(null);
                if (path != null) {
                    return Paths.get(path.toString()).toAbsolutePath().getParent();
                }
            } catch (Throwable ignored) {
                // fall through to /proc/self/exe
            }
            try {
                Path exe = Paths.get("/proc/self/exe");
                if (Files.exists(exe)) {
                    return exe.toRealPath().getParent();
                }
            } catch (IOException ignored) {
                // fall through
            }
        }
        return Paths.get(".").toAbsolutePath().normalize();
    }

    private static String executableName() {
        return PLATFORM.startsWith("windows") ? "sdx-llm.exe" : "sdx-llm";
    }

    /** JavaCPP-format platform string (e.g. "linux-x86_64", "macosx-arm64"). */
    private static String detectPlatform() {
        String osName = System.getProperty("os.name", "").toLowerCase();
        String os;
        if (osName.contains("linux")) {
            os = "linux";
        } else if (osName.contains("mac") || osName.contains("darwin")) {
            os = "macosx";
        } else if (osName.contains("windows")) {
            os = "windows";
        } else {
            os = osName.replaceAll("[^a-z0-9]", "");
        }
        String archName = System.getProperty("os.arch", "").toLowerCase();
        String arch;
        switch (archName) {
            case "amd64":
            case "x86_64":
                arch = "x86_64";
                break;
            case "aarch64":
            case "arm64":
                arch = "arm64";
                break;
            default:
                arch = archName;
        }
        return os + "-" + arch;
    }
}
