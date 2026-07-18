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

#ifndef SD_KERNEL_PLUGIN_TEMPLATE_H
#define SD_KERNEL_PLUGIN_TEMPLATE_H

#include <system/BackendNamespace.h>

/**
 * Example Kernel Plugin Template
 * ==============================
 *
 * This file demonstrates how to create a custom kernel plugin for ND4J/libnd4j.
 * Kernel plugins allow you to provide optimized implementations for operations
 * that can be loaded at runtime without recompiling the main library.
 *
 * ## Quick Start
 *
 * 1. Create a new .cpp file for your plugin
 * 2. Include the necessary headers
 * 3. Implement your custom PlatformHelper classes
 * 4. Create a plugin class that extends SimpleKernelPlugin
 * 5. Use the SD_DECLARE_KERNEL_PLUGIN macro to export the required functions
 * 6. Compile as a shared library (.so/.dll/.dylib)
 *
 * ## Example Plugin Code
 *
 * ```cpp
 * #include <helpers/DynamicKernelLoader.h>
 * #include <ops/declarable/PlatformHelper.h>
 * #include <graph/Context.h>
 *
 * using namespace sd;
 * using namespace sd::ops;
 * using namespace sd::ops::platforms;
 *
 * // Custom implementation of conv2d with special optimizations
 * class MyOptimizedConv2d : public PlatformHelper {
 * public:
 *   MyOptimizedConv2d() : PlatformHelper("conv2d", samediff::ENGINE_CPU) {}
 *
 *   bool isUsable(graph::Context& context) override {
 *     // Check if this implementation is suitable for the input
 *     auto input = context.getNDArray(0);
 *     if (!input) return false;
 *
 *     // Only handle FLOAT32 for this example
 *     if (input->dataType() != DataType::FLOAT32) return false;
 *
 *     // Only handle specific sizes where our optimization helps
 *     if (input->lengthOf() < 1024) return false;
 *
 *     return true;
 *   }
 *
 *   Status invokeHelper(graph::Context& context) override {
 *     auto input = context.getNDArray(0);
 *     auto weights = context.getNDArray(1);
 *     auto output = context.outputVariable(0)->getNDArray();
 *
 *     // Your optimized implementation here
 *     // ...
 *
 *     return Status::OK;
 *   }
 * };
 *
 * // Custom GEMM implementation using a specialized library
 * class MyFastGemm : public PlatformHelper {
 * public:
 *   MyFastGemm() : PlatformHelper("matmul", samediff::ENGINE_CPU) {}
 *
 *   bool isUsable(graph::Context& context) override {
 *     // Implementation-specific checks
 *     return true;
 *   }
 *
 *   Status invokeHelper(graph::Context& context) override {
 *     // Your implementation
 *     return Status::OK;
 *   }
 * };
 *
 * // Plugin class that registers our kernels
 * class MyKernelPlugin : public SimpleKernelPlugin {
 * public:
 *   MyKernelPlugin()
 *     : SimpleKernelPlugin("MyOptimizedKernels", {1, 0, 0}) {}
 *
 *   bool initialize() override {
 *     // Register conv2d kernel
 *     KernelInfo conv2dInfo;
 *     conv2dInfo.name = "conv2d";
 *     conv2dInfo.engine = samediff::ENGINE_CPU;
 *     conv2dInfo.description = "Optimized conv2d for large inputs";
 *     conv2dInfo.priority = 150;  // Higher priority than default (100)
 *     conv2dInfo.supportedTypes = {DataType::FLOAT32};
 *     registerKernel(conv2dInfo, []() { return new MyOptimizedConv2d(); });
 *
 *     // Register matmul kernel
 *     registerKernel("matmul", samediff::ENGINE_CPU,
 *                    []() { return new MyFastGemm(); }, 120);
 *
 *     return true;
 *   }
 * };
 *
 * // Export the plugin
 * SD_DECLARE_KERNEL_PLUGIN(MyKernelPlugin)
 * ```
 *
 * ## Compilation
 *
 * Linux:
 *   g++ -shared -fPIC -o libmy_kernels.so my_kernels.cpp \
 *       -I/path/to/libnd4j/include -L/path/to/libnd4j/lib -lnd4j
 *
 * macOS:
 *   clang++ -shared -fPIC -o libmy_kernels.dylib my_kernels.cpp \
 *       -I/path/to/libnd4j/include -L/path/to/libnd4j/lib -lnd4j
 *
 * Windows (MSVC):
 *   cl /LD /EHsc my_kernels.cpp /I"path\to\libnd4j\include" \
 *      /link /LIBPATH:"path\to\libnd4j\lib" nd4j.lib /OUT:my_kernels.dll
 *
 * ## Loading the Plugin
 *
 * C++:
 *   auto& loader = DynamicKernelLoader::getInstance();
 *   loader.loadPlugin("/path/to/libmy_kernels.so");
 *
 * Java:
 *   Nd4j.loadKernelPlugin("/path/to/libmy_kernels.so");
 *   // Or use the plugin manager
 *   KernelPluginManager manager = Nd4j.getKernelPluginManager();
 *   manager.loadPlugin("/path/to/libmy_kernels.so");
 *
 * ## Environment Variables
 *
 * SD_KERNEL_PLUGIN_PATH=/path/to/plugins:/another/path
 *   Colon-separated (semicolon on Windows) list of directories to search
 *
 * SD_KERNEL_PLUGIN_AUTO=1
 *   Automatically load all plugins from search paths on startup
 *
 * ## Best Practices
 *
 * 1. **Priority**: Set higher priority (> 100) for genuinely faster implementations
 * 2. **isUsable()**: Be specific about when your implementation should be used
 * 3. **Data types**: Only claim to support types you actually handle
 * 4. **Error handling**: Return appropriate Status codes, don't throw exceptions
 * 5. **Thread safety**: Ensure your implementation is thread-safe
 * 6. **Memory**: Use the provided workspace/allocator when possible
 * 7. **Testing**: Test your plugin with various input shapes and types
 *
 * ## Debugging
 *
 * Set SD_KERNEL_VERBOSE=1 to see kernel selection decisions:
 *   export SD_KERNEL_VERBOSE=1
 *
 * In Java, enable verbose mode:
 *   Nd4j.getKernelSelector().setVerbose(true);
 */

#include <helpers/DynamicKernelLoader.h>

namespace sd {
namespace ops {
namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

/**
 * Example: A simple plugin that wraps a lambda function.
 * Useful for quick prototyping.
 */
inline KernelPlugin* createLambdaPlugin(
    const std::string& pluginName,
    const std::string& kernelName,
    samediff::Engine engine,
    std::function<bool(graph::Context&)> isUsable,
    std::function<Status(graph::Context&)> invoke) {

  class LambdaPlugin : public SimpleKernelPlugin {
   public:
    std::string _kernelName;
    samediff::Engine _engine;
    std::function<bool(graph::Context&)> _isUsable;
    std::function<Status(graph::Context&)> _invoke;

    LambdaPlugin(const std::string& name,
                 const std::string& kernel,
                 samediff::Engine eng,
                 std::function<bool(graph::Context&)> isUsableF,
                 std::function<Status(graph::Context&)> invokeF)
        : SimpleKernelPlugin(name, {1, 0, 0}),
          _kernelName(kernel), _engine(eng),
          _isUsable(std::move(isUsableF)), _invoke(std::move(invokeF)) {}

    bool initialize() override {
      registerKernel(_kernelName, _engine, [this]() {
        return new LambdaPlatformHelper(
            _kernelName.c_str(), _engine, _isUsable, _invoke);
      });
      return true;
    }
  };

  return new LambdaPlugin(pluginName, kernelName, engine,
                           std::move(isUsable), std::move(invoke));
}

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_KERNEL_PLUGIN_TEMPLATE_H
