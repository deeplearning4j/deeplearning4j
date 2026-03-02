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

// Platform-specific includes for dynamic loading — must be BEFORE project headers
// so that _WINDOWS_ is defined before types.h constexpr alias guards are checked
#ifdef _WIN32
#define NOMINMAX  // Prevent windows.h from defining min/max macros
#include <windows.h>
#endif

#include <helpers/DynamicKernelLoader.h>
#include <ops/declarable/OpRegistrator.h>

#include <sstream>
#include <unordered_set>

#ifndef _WIN32
#include <dlfcn.h>
#include <dirent.h>
#include <fnmatch.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

// SimpleKernelPlugin implementation

void SimpleKernelPlugin::registerKernel(const std::string& name, samediff::Engine engine,
                                         PlatformHelperFactory factory, int priority) {
  KernelInfo info;
  info.name = name;
  info.engine = engine;
  info.priority = priority;
  registerKernel(info, factory);
}

void SimpleKernelPlugin::registerKernel(const KernelInfo& info, PlatformHelperFactory factory) {
  _kernels.push_back(info);
  _factories[info.name + "_" + std::to_string(static_cast<int>(info.engine))] = factory;

  // Create the helper and register it with the system
  PlatformHelper* helper = factory();
  if (helper != nullptr) {
    OpRegistrator::getInstance().registerHelper(helper);
  }
}

// DynamicKernelLoader implementation

DynamicKernelLoader& DynamicKernelLoader::getInstance() {
  static DynamicKernelLoader instance;
  return instance;
}

void* DynamicKernelLoader::loadLibrary(const std::string& path) {
#ifdef _WIN32
  return LoadLibraryA(path.c_str());
#else
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void DynamicKernelLoader::unloadLibrary(void* handle) {
  if (handle == nullptr) return;

#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

void* DynamicKernelLoader::getSymbol(void* handle, const char* name) {
  if (handle == nullptr) return nullptr;

#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
  return dlsym(handle, name);
#endif
}

bool DynamicKernelLoader::loadPlugin(const std::string& path) {
  std::lock_guard<std::mutex> lock(_mutex);

  // Check if already loaded
  if (_loadedPlugins.find(path) != _loadedPlugins.end()) {
    return true;  // Already loaded
  }

  // Load the library
  void* handle = loadLibrary(path);
  if (handle == nullptr) {
#ifndef _WIN32
    sd_printf("Failed to load plugin %s: %s\n", path.c_str(), dlerror());
#else
    sd_printf("Failed to load plugin %s: error %d\n", path.c_str(), GetLastError());
#endif
    return false;
  }

  // Get the create function
  using CreateFunc = KernelPlugin* (*)();
  auto createFunc = reinterpret_cast<CreateFunc>(getSymbol(handle, "sd_plugin_create"));
  if (createFunc == nullptr) {
    sd_printf("Plugin %s does not export sd_plugin_create\n", path.c_str());
    unloadLibrary(handle);
    return false;
  }

  // Check API version
  using VersionFunc = int (*)();
  auto versionFunc = reinterpret_cast<VersionFunc>(getSymbol(handle, "sd_plugin_api_version"));
  if (versionFunc != nullptr) {
    int version = versionFunc();
    if (version != 1) {
      sd_printf("Plugin %s has incompatible API version %d\n", path.c_str(), version);
      unloadLibrary(handle);
      return false;
    }
  }

  // Create the plugin
  KernelPlugin* plugin = createFunc();
  if (plugin == nullptr) {
    sd_printf("Plugin %s returned null from sd_plugin_create\n", path.c_str());
    unloadLibrary(handle);
    return false;
  }

  // Initialize the plugin
  if (!plugin->initialize()) {
    sd_printf("Plugin %s failed to initialize\n", path.c_str());
    delete plugin;
    unloadLibrary(handle);
    return false;
  }

  // Store the plugin
  LoadedPlugin loaded;
  loaded.path = path;
  loaded.handle = handle;
  loaded.plugin = plugin;
  loaded.active = true;

  _loadedPlugins[path] = loaded;
  _loadedPlugins[plugin->getName()] = loaded;

  sd_printf("Loaded plugin %s (%s) with %d kernels\n", plugin->getName().c_str(),
            plugin->getVersion().toString().c_str(),
            static_cast<int>(plugin->getProvidedKernels().size()));

  return true;
}

bool DynamicKernelLoader::unloadPlugin(const std::string& nameOrPath) {
  std::lock_guard<std::mutex> lock(_mutex);

  auto it = _loadedPlugins.find(nameOrPath);
  if (it == _loadedPlugins.end()) {
    return false;
  }

  LoadedPlugin& loaded = it->second;

  // Shutdown the plugin
  if (loaded.plugin != nullptr) {
    loaded.plugin->shutdown();

    // Get destroy function
    using DestroyFunc = void (*)(KernelPlugin*);
    auto destroyFunc = reinterpret_cast<DestroyFunc>(getSymbol(loaded.handle, "sd_plugin_destroy"));
    if (destroyFunc != nullptr) {
      destroyFunc(loaded.plugin);
    } else {
      delete loaded.plugin;
    }
  }

  // Unload the library
  unloadLibrary(loaded.handle);

  // Remove from maps
  std::string path = loaded.path;
  std::string name = loaded.plugin ? loaded.plugin->getName() : "";

  _loadedPlugins.erase(path);
  if (!name.empty() && _loadedPlugins.find(name) != _loadedPlugins.end()) {
    _loadedPlugins.erase(name);
  }

  return true;
}

bool DynamicKernelLoader::reloadPlugin(const std::string& nameOrPath) {
  std::string path;

  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _loadedPlugins.find(nameOrPath);
    if (it != _loadedPlugins.end()) {
      path = it->second.path;
    } else {
      path = nameOrPath;
    }
  }

  if (!unloadPlugin(nameOrPath)) {
    return false;
  }

  return loadPlugin(path);
}

bool DynamicKernelLoader::isPluginLoaded(const std::string& nameOrPath) const {
  std::lock_guard<std::mutex> lock(_mutex);
  return _loadedPlugins.find(nameOrPath) != _loadedPlugins.end();
}

const LoadedPlugin* DynamicKernelLoader::getPlugin(const std::string& nameOrPath) const {
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _loadedPlugins.find(nameOrPath);
  if (it != _loadedPlugins.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<LoadedPlugin> DynamicKernelLoader::getLoadedPlugins() const {
  std::lock_guard<std::mutex> lock(_mutex);

  std::vector<LoadedPlugin> result;
  std::unordered_set<void*> seen;

  for (const auto& pair : _loadedPlugins) {
    if (seen.find(pair.second.handle) == seen.end()) {
      result.push_back(pair.second);
      seen.insert(pair.second.handle);
    }
  }

  return result;
}

void DynamicKernelLoader::addSearchPath(const std::string& path) {
  std::lock_guard<std::mutex> lock(_mutex);
  _searchPaths.push_back(path);
}

int DynamicKernelLoader::loadPluginsFromSearchPaths() {
#ifdef _WIN32
  return loadPluginsFromSearchPaths("*.dll");
#elif defined(__APPLE__)
  return loadPluginsFromSearchPaths("*.dylib");
#else
  return loadPluginsFromSearchPaths("*.so");
#endif
}

int DynamicKernelLoader::loadPluginsFromSearchPaths(const std::string& pattern) {
  int count = 0;

#ifndef _WIN32
  for (const auto& searchPath : _searchPaths) {
    DIR* dir = opendir(searchPath.c_str());
    if (dir == nullptr) continue;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
      if (fnmatch(pattern.c_str(), entry->d_name, 0) == 0) {
        std::string fullPath = searchPath + "/" + entry->d_name;
        if (loadPlugin(fullPath)) {
          count++;
        }
      }
    }

    closedir(dir);
  }
#else
  for (const auto& searchPath : _searchPaths) {
    std::string searchPattern = searchPath + "\\" + pattern;
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(searchPattern.c_str(), &findData);

    if (hFind != INVALID_HANDLE_VALUE) {
      do {
        std::string fullPath = searchPath + "\\" + findData.cFileName;
        if (loadPlugin(fullPath)) {
          count++;
        }
      } while (FindNextFileA(hFind, &findData));

      FindClose(hFind);
    }
  }
#endif

  return count;
}

void DynamicKernelLoader::setPluginActive(const std::string& nameOrPath, bool active) {
  std::lock_guard<std::mutex> lock(_mutex);

  auto it = _loadedPlugins.find(nameOrPath);
  if (it != _loadedPlugins.end()) {
    it->second.active = active;
  }
}

std::string DynamicKernelLoader::getPluginSummary() const {
  std::lock_guard<std::mutex> lock(_mutex);

  std::stringstream ss;
  ss << "Loaded Kernel Plugins\n";
  ss << "=====================\n\n";

  std::unordered_set<void*> seen;

  for (const auto& pair : _loadedPlugins) {
    if (seen.find(pair.second.handle) != seen.end()) continue;
    seen.insert(pair.second.handle);

    const LoadedPlugin& loaded = pair.second;
    if (loaded.plugin == nullptr) continue;

    ss << "Plugin: " << loaded.plugin->getName() << "\n";
    ss << "  Version: " << loaded.plugin->getVersion().toString() << "\n";
    ss << "  Path: " << loaded.path << "\n";
    ss << "  Active: " << (loaded.active ? "yes" : "no") << "\n";
    ss << "  Kernels:\n";

    for (const auto& kernel : loaded.plugin->getProvidedKernels()) {
      ss << "    - " << kernel.name << " (engine: " << static_cast<int>(kernel.engine)
         << ", priority: " << kernel.priority << ")\n";
    }
    ss << "\n";
  }

  if (seen.empty()) {
    ss << "No plugins loaded.\n";
  }

  return ss.str();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
