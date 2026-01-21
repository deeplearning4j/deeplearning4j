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

package org.nd4j.linalg.api.ops.executioner;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.util.List;
import java.util.Map;

/**
 * Manager for dynamically loaded kernel plugins.
 * <p>
 * Kernel plugins allow loading custom kernel implementations at runtime from
 * shared libraries (.so/.dll/.dylib). This enables:
 * <ul>
 *   <li>Custom optimized kernels for specific hardware</li>
 *   <li>Third-party kernel implementations</li>
 *   <li>Hot-swapping kernels without recompilation</li>
 *   <li>Plugin architecture for operations</li>
 * </ul>
 * </p>
 * <p>
 * Example usage:
 * <pre>{@code
 * KernelPluginManager manager = Nd4j.getKernelPluginManager();
 *
 * // Load a custom kernel plugin
 * manager.loadPlugin("/path/to/my_kernels.so");
 *
 * // List loaded plugins
 * for (PluginInfo info : manager.getLoadedPlugins()) {
 *     System.out.println("Plugin: " + info.getName() + " - " + info.getKernelCount() + " kernels");
 * }
 *
 * // Unload a plugin
 * manager.unloadPlugin("MyKernels");
 * }</pre>
 * </p>
 *
 * @author Eclipse Deeplearning4j Development Team
 */
public interface KernelPluginManager {

    /**
     * Information about a loaded kernel plugin
     */
    @Data
    class PluginInfo {
        private String name;
        private String path;
        private String version;
        private boolean active;
        private int kernelCount;
        private long loadTime;
        private List<KernelInfo> providedKernels;
    }

    /**
     * Information about a kernel provided by a plugin
     */
    @Data
    class KernelInfo {
        private String name;
        private String engine;
        private String description;
        private int priority;
        private List<String> supportedDataTypes;
    }

    /**
     * Load a kernel plugin from a shared library file.
     *
     * @param path Path to the shared library (.so, .dll, .dylib)
     * @return true if plugin was loaded successfully
     * @throws KernelPluginException if loading fails
     */
    boolean loadPlugin(String path) throws KernelPluginException;

    /**
     * Load a kernel plugin from a File.
     *
     * @param file The shared library file
     * @return true if plugin was loaded successfully
     * @throws KernelPluginException if loading fails
     */
    default boolean loadPlugin(File file) throws KernelPluginException {
        return loadPlugin(file.getAbsolutePath());
    }

    /**
     * Unload a previously loaded plugin.
     *
     * @param nameOrPath Plugin name or path
     * @return true if plugin was unloaded
     */
    boolean unloadPlugin(String nameOrPath);

    /**
     * Reload a plugin (unload then load again).
     * Useful for development and hot-reloading.
     *
     * @param nameOrPath Plugin name or path
     * @return true if plugin was reloaded successfully
     * @throws KernelPluginException if reloading fails
     */
    boolean reloadPlugin(String nameOrPath) throws KernelPluginException;

    /**
     * Check if a plugin is loaded.
     *
     * @param nameOrPath Plugin name or path
     * @return true if plugin is loaded
     */
    boolean isPluginLoaded(String nameOrPath);

    /**
     * Get information about a loaded plugin.
     *
     * @param nameOrPath Plugin name or path
     * @return PluginInfo or null if not found
     */
    PluginInfo getPlugin(String nameOrPath);

    /**
     * Get all loaded plugins.
     *
     * @return List of plugin information
     */
    List<PluginInfo> getLoadedPlugins();

    /**
     * Add a directory to the plugin search path.
     *
     * @param path Directory path
     */
    void addSearchPath(String path);

    /**
     * Get current search paths.
     *
     * @return List of search paths
     */
    List<String> getSearchPaths();

    /**
     * Load all plugins found in search paths.
     *
     * @return Number of plugins loaded
     */
    int loadPluginsFromSearchPaths();

    /**
     * Load plugins matching a pattern from search paths.
     *
     * @param pattern Glob pattern (e.g., "*.so", "libsd_*.dll")
     * @return Number of plugins loaded
     */
    int loadPluginsFromSearchPaths(String pattern);

    /**
     * Discover plugins in a directory without loading them.
     *
     * @param directory Directory to scan
     * @return List of discovered plugin paths
     */
    List<String> discoverPlugins(String directory);

    /**
     * Discover plugins matching a pattern.
     *
     * @param directory Directory to scan
     * @param pattern File pattern (e.g., "*.so")
     * @return List of discovered plugin paths
     */
    List<String> discoverPlugins(String directory, String pattern);

    /**
     * Enable or disable a loaded plugin.
     * When disabled, the plugin's kernels won't be used.
     *
     * @param nameOrPath Plugin name or path
     * @param active true to enable, false to disable
     */
    void setPluginActive(String nameOrPath, boolean active);

    /**
     * Set a configuration option on a plugin.
     *
     * @param pluginName Plugin name
     * @param key Option key
     * @param value Option value
     * @return true if option was set successfully
     */
    boolean setPluginOption(String pluginName, String key, String value);

    /**
     * Get configuration options for a plugin.
     *
     * @param pluginName Plugin name
     * @return Map of option key-value pairs
     */
    Map<String, String> getPluginOptions(String pluginName);

    /**
     * Get all kernels provided by loaded plugins.
     *
     * @return List of kernel information
     */
    List<KernelInfo> getAllProvidedKernels();

    /**
     * Find plugins that provide a specific kernel.
     *
     * @param kernelName Name of the kernel (e.g., "conv2d")
     * @return List of plugins providing this kernel
     */
    List<PluginInfo> findPluginsForKernel(String kernelName);

    /**
     * Enable hot-reload functionality.
     * When enabled, plugins are automatically reloaded when their files change.
     *
     * @param enable true to enable
     */
    void enableHotReload(boolean enable);

    /**
     * Check if hot-reload is enabled.
     *
     * @return true if hot-reload is enabled
     */
    boolean isHotReloadEnabled();

    /**
     * Get a summary of loaded plugins and their status.
     *
     * @return Human-readable summary string
     */
    String getPluginSummary();

    /**
     * Exception thrown when plugin operations fail
     */
    class KernelPluginException extends Exception {
        public KernelPluginException(String message) {
            super(message);
        }

        public KernelPluginException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
