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

package org.nd4j.nativeblas;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.nd4j.common.config.ND4JSystemProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads native plugin libraries from a configurable directory after the main
 * ND4J native backend has been initialized.
 *
 * <p>Plugin libraries are platform-specific shared objects (.so on Linux,
 * .dll on Windows, .dylib on macOS) placed in the plugin directory. They
 * are loaded via {@link System#load(String)} in alphabetical order.</p>
 *
 * <p>The plugin directory defaults to {@code ~/.dl4j/plugins/} and can be
 * overridden with the system property {@code nd4j.native.plugin.path}.</p>
 *
 * <p>This loader is designed to be called once from {@link NativeOpsHolder}
 * after the primary native library initialization completes. Plugin load
 * failures are logged as warnings but do not prevent ND4J from operating.</p>
 */
public class NativePluginLoader {

    private static final Logger log = LoggerFactory.getLogger(NativePluginLoader.class);

    /**
     * System property to override the plugin directory path.
     */
    // Centralized in ND4JSystemProperties; kept here as an alias for backward compatibility.
    public static final String PLUGIN_PATH_PROPERTY = ND4JSystemProperties.NATIVE_PLUGIN_PATH;

    /**
     * Default plugin directory relative to the user's home directory.
     */
    public static final String DEFAULT_PLUGIN_DIR = ".dl4j" + File.separator + "plugins";

    private static volatile boolean loaded = false;
    private static final List<String> loadedPlugins = Collections.synchronizedList(new ArrayList<>());

    private NativePluginLoader() {
        // static utility class
    }

    /**
     * Scan the plugin directory and load all native libraries found.
     * This method is idempotent -- subsequent calls after the first are no-ops.
     *
     * <p>Libraries are loaded in alphabetical order by filename. If a library
     * fails to load, a warning is logged and loading continues with the next
     * library.</p>
     */
    public static synchronized void loadPlugins() {
        if (loaded) {
            return;
        }
        loaded = true;

        File pluginDir = resolvePluginDirectory();
        if (pluginDir == null) {
            return;
        }

        String[] extensions = getPlatformLibraryExtensions();
        File[] libraries = pluginDir.listFiles(new NativeLibraryFilter(extensions));
        if (libraries == null || libraries.length == 0) {
            log.debug("No native plugin libraries found in {}", pluginDir.getAbsolutePath());
            return;
        }

        // Sort for deterministic load order
        java.util.Arrays.sort(libraries);

        log.info("Loading native plugins from {} ({} libraries found)", pluginDir.getAbsolutePath(), libraries.length);

        for (File lib : libraries) {
            try {
                System.load(lib.getAbsolutePath());
                loadedPlugins.add(lib.getName());
                log.info("Loaded native plugin: {}", lib.getName());
            } catch (UnsatisfiedLinkError e) {
                log.warn("Failed to load native plugin {}: {}", lib.getName(), e.getMessage());
            } catch (SecurityException e) {
                log.warn("Security restriction prevented loading plugin {}: {}", lib.getName(), e.getMessage());
            }
        }

        if (!loadedPlugins.isEmpty()) {
            log.info("Successfully loaded {} native plugin(s): {}", loadedPlugins.size(), loadedPlugins);
        }
    }

    /**
     * Returns an unmodifiable list of plugin filenames that were successfully loaded.
     *
     * @return list of loaded plugin filenames
     */
    public static List<String> getLoadedPlugins() {
        return Collections.unmodifiableList(new ArrayList<>(loadedPlugins));
    }

    /**
     * Returns true if plugins have already been loaded (or attempted).
     *
     * @return true if {@link #loadPlugins()} has been called
     */
    public static boolean isLoaded() {
        return loaded;
    }

    /**
     * Resolve the plugin directory from the system property or default location.
     * Returns null if the directory does not exist or is not readable.
     */
    private static File resolvePluginDirectory() {
        String path = System.getProperty(PLUGIN_PATH_PROPERTY);
        File pluginDir;

        if (path != null && !path.isEmpty()) {
            pluginDir = new File(path);
        } else {
            String userHome = System.getProperty("user.home");
            if (userHome == null || userHome.isEmpty()) {
                log.debug("Cannot determine user home directory; skipping native plugin loading");
                return null;
            }
            pluginDir = new File(userHome, DEFAULT_PLUGIN_DIR);
        }

        if (!pluginDir.exists()) {
            log.debug("Native plugin directory does not exist: {}", pluginDir.getAbsolutePath());
            return null;
        }

        if (!pluginDir.isDirectory()) {
            log.warn("Native plugin path is not a directory: {}", pluginDir.getAbsolutePath());
            return null;
        }

        if (!pluginDir.canRead()) {
            log.warn("Native plugin directory is not readable: {}", pluginDir.getAbsolutePath());
            return null;
        }

        return pluginDir;
    }

    /**
     * Returns the file extensions for native shared libraries on the current platform.
     */
    private static String[] getPlatformLibraryExtensions() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return new String[]{".dll"};
        } else if (os.contains("mac") || os.contains("darwin")) {
            return new String[]{".dylib", ".jnilib"};
        } else {
            // Linux, Unix, and other POSIX systems
            return new String[]{".so"};
        }
    }

    /**
     * FilenameFilter that accepts files ending with any of the given extensions.
     */
    private static class NativeLibraryFilter implements FilenameFilter {
        private final String[] extensions;

        NativeLibraryFilter(String[] extensions) {
            this.extensions = extensions;
        }

        @Override
        public boolean accept(File dir, String name) {
            for (String ext : extensions) {
                if (name.endsWith(ext)) {
                    return true;
                }
            }
            return false;
        }
    }
}
