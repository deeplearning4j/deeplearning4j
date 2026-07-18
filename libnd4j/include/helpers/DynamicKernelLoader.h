/* ******************************************************************************
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

#ifndef SD_DYNAMIC_KERNEL_LOADER_H
#define SD_DYNAMIC_KERNEL_LOADER_H

#include <system/BackendNamespace.h>

#include <execution/Engine.h>
#include <graph/Context.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/common.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace ops {
namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

/**
 * Version information for plugins
 */
struct Version {
  int major;
  int minor;
  int patch;

  std::string toString() const {
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
  }
};

/**
 * Information about a kernel provided by a plugin
 */
struct KernelInfo {
  std::string name;
  samediff::Engine engine;
  std::string description;
  int priority;
  std::vector<DataType> supportedTypes;
};

/**
 * Base interface for kernel plugins.
 * Plugins provide custom kernel implementations that can be loaded at runtime.
 */
class SD_LIB_EXPORT KernelPlugin {
 public:
  virtual ~KernelPlugin() = default;

  /**
   * Get the plugin name
   */
  virtual std::string getName() const = 0;

  /**
   * Get the plugin version
   */
  virtual Version getVersion() const = 0;

  /**
   * Initialize the plugin. Called after loading.
   * @return true if initialization succeeded
   */
  virtual bool initialize() = 0;

  /**
   * Shutdown the plugin. Called before unloading.
   */
  virtual void shutdown() = 0;

  /**
   * Get list of kernels provided by this plugin
   */
  virtual std::vector<KernelInfo> getProvidedKernels() const = 0;

  /**
   * Set a configuration option
   */
  virtual bool setOption(const std::string& key, const std::string& value) { return false; }

  /**
   * Get a configuration option
   */
  virtual std::string getOption(const std::string& key) const { return ""; }
};

/**
 * Factory function type for creating platform helpers
 */
using PlatformHelperFactory = std::function<PlatformHelper*()>;

/**
 * Base class for simple kernel plugins.
 * Provides common registration functionality.
 */
class SD_LIB_EXPORT SimpleKernelPlugin : public KernelPlugin {
 protected:
  std::string _name;
  Version _version;
  std::vector<KernelInfo> _kernels;
  std::unordered_map<std::string, PlatformHelperFactory> _factories;

  /**
   * Register a kernel with the system
   */
  void registerKernel(const std::string& name, samediff::Engine engine,
                      PlatformHelperFactory factory, int priority = 100);

  /**
   * Register a kernel with full info
   */
  void registerKernel(const KernelInfo& info, PlatformHelperFactory factory);

 public:
  SimpleKernelPlugin(const std::string& name, Version version)
      : _name(name), _version(version) {}

  std::string getName() const override { return _name; }
  Version getVersion() const override { return _version; }

  void shutdown() override {
    _factories.clear();
    _kernels.clear();
  }

  std::vector<KernelInfo> getProvidedKernels() const override { return _kernels; }
};

/**
 * PlatformHelper that wraps lambda functions.
 * Useful for quick prototyping and inline kernel definitions.
 */
class SD_LIB_EXPORT LambdaPlatformHelper : public PlatformHelper {
 private:
  std::function<bool(graph::Context&)> _isUsableFunc;
  std::function<Status(graph::Context&)> _invokeFunc;

 public:
  LambdaPlatformHelper(const char* name, samediff::Engine engine,
                       std::function<bool(graph::Context&)> isUsableFunc,
                       std::function<Status(graph::Context&)> invokeFunc)
      : PlatformHelper(name, engine),
        _isUsableFunc(std::move(isUsableFunc)),
        _invokeFunc(std::move(invokeFunc)) {}

  bool isUsable(graph::Context& context) override {
    if (_isUsableFunc) {
      return _isUsableFunc(context);
    }
    return true;
  }

  Status invokeHelper(graph::Context& context) override {
    if (_invokeFunc) {
      return _invokeFunc(context);
    }
    return Status::OK;
  }
};

/**
 * Information about a loaded plugin
 */
struct LoadedPlugin {
  std::string path;
  void* handle;  // dlopen handle / HMODULE
  KernelPlugin* plugin;
  bool active;
};

/**
 * Dynamic kernel loader singleton.
 * Manages loading and unloading of kernel plugins from shared libraries.
 */
class SD_LIB_EXPORT DynamicKernelLoader {
 private:
  std::unordered_map<std::string, LoadedPlugin> _loadedPlugins;
  std::vector<std::string> _searchPaths;
  bool _hotReloadEnabled;
  mutable std::mutex _mutex;

  // Private constructor for Meyers singleton
  DynamicKernelLoader() : _hotReloadEnabled(false) {}

  /**
   * Platform-specific library loading
   */
  void* loadLibrary(const std::string& path);
  void unloadLibrary(void* handle);
  void* getSymbol(void* handle, const char* name);

 public:
  static DynamicKernelLoader& getInstance();

  // Prevent copying
  DynamicKernelLoader(const DynamicKernelLoader&) = delete;
  DynamicKernelLoader& operator=(const DynamicKernelLoader&) = delete;

  /**
   * Load a plugin from a shared library
   */
  bool loadPlugin(const std::string& path);

  /**
   * Unload a plugin
   */
  bool unloadPlugin(const std::string& nameOrPath);

  /**
   * Reload a plugin
   */
  bool reloadPlugin(const std::string& nameOrPath);

  /**
   * Check if a plugin is loaded
   */
  bool isPluginLoaded(const std::string& nameOrPath) const;

  /**
   * Get loaded plugin info
   */
  const LoadedPlugin* getPlugin(const std::string& nameOrPath) const;

  /**
   * Get all loaded plugins
   */
  std::vector<LoadedPlugin> getLoadedPlugins() const;

  /**
   * Add a search path for plugins
   */
  void addSearchPath(const std::string& path);

  /**
   * Get search paths
   */
  const std::vector<std::string>& getSearchPaths() const { return _searchPaths; }

  /**
   * Load all plugins from search paths
   */
  int loadPluginsFromSearchPaths();

  /**
   * Load plugins matching a pattern
   */
  int loadPluginsFromSearchPaths(const std::string& pattern);

  /**
   * Enable/disable hot reload
   */
  void enableHotReload(bool enable) { _hotReloadEnabled = enable; }
  bool isHotReloadEnabled() const { return _hotReloadEnabled; }

  /**
   * Set plugin active state
   */
  void setPluginActive(const std::string& nameOrPath, bool active);

  /**
   * Get plugin summary
   */
  std::string getPluginSummary() const;
};

/**
 * Macro to declare plugin export functions.
 * Use this at the end of your plugin source file.
 */
#define SD_DECLARE_KERNEL_PLUGIN(PluginClass) \
  extern "C" { \
    SD_LIB_EXPORT sd::ops::platforms::KernelPlugin* sd_plugin_create() { \
      return new PluginClass(); \
    } \
    SD_LIB_EXPORT void sd_plugin_destroy(sd::ops::platforms::KernelPlugin* plugin) { \
      delete plugin; \
    } \
    SD_LIB_EXPORT int sd_plugin_api_version() { \
      return 1; \
    } \
  }

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_DYNAMIC_KERNEL_LOADER_H
