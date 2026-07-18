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

#ifndef LIBND4J_BACKEND_NAMESPACE_H
#define LIBND4J_BACKEND_NAMESPACE_H

/**
 * @file BackendNamespace.h
 * @brief Backend-specific namespace isolation for multi-backend support
 *
 * Backend DSOs coexist in the same process and use distinct internal C++
 * namespaces while retaining the shared exported C ABI and ordinary dynamic
 * linkage. This header configures that namespace from the selected chip.
 *
 * Build Configuration:
 * - CPU build:    -DSD_BACKEND_NAMESPACE=sd_cpu  (CMake auto-sets this)
 * - CUDA build:   -DSD_BACKEND_NAMESPACE=sd_cuda
 * - TPU build:    -DSD_BACKEND_NAMESPACE=sd_tpu
 * - Vulkan build: -DSD_BACKEND_NAMESPACE=sd_vulkan
 *
 * Usage in code:
 * @code
 *   namespace SD_NS {
 *       class MyClass { ... };
 *   }
 *   // MyClass is now in sd_cpu:: or sd_cuda:: depending on build
 * @endcode
 *
 * For backward compatibility, we also define 'sd' as an alias to the backend namespace.
 * This means existing code using 'sd::' will continue to work, but will resolve to
 * the backend-specific namespace.
 */

// Helper macro for token pasting
#define SD_CONCAT_IMPL(a, b) a##b
#define SD_CONCAT(a, b) SD_CONCAT_IMPL(a, b)

// SD_BACKEND_NAMESPACE is set by CMake based on the backend being built.
// Remember whether the build supplied it before installing the IDE/parser default.
#if defined(SD_BACKEND_NAMESPACE) && !defined(SD_BACKEND_NAMESPACE_CONFIGURED)
#define SD_BACKEND_NAMESPACE_CONFIGURED 1
#endif

// Default to 'sd' for backward compatibility if not defined (e.g., IDE code completion)
#ifndef SD_BACKEND_NAMESPACE
    // In production builds, CMake always sets this. This default is for:
    // - IDE code completion/intellisense
    // - Direct compilation without CMake (testing)
    #define SD_BACKEND_NAMESPACE sd
#endif

/**
 * @def SD_NS
 * @brief The primary namespace for all libnd4j symbols
 *
 * Use this macro instead of hardcoded 'sd' to enable multi-backend isolation.
 * Example:
 * @code
 *   namespace SD_NS {
 *       namespace ops {
 *           class add_op : public DeclarableOp { ... };
 *       }
 *   }
 * @endcode
 */
#define SD_NS SD_BACKEND_NAMESPACE

/**
 * @def SD_NS_NAME
 * @brief String representation of the namespace (for logging/debug)
 */
#define SD_STRINGIFY_IMPL(x) #x
#define SD_STRINGIFY(x) SD_STRINGIFY_IMPL(x)
#define SD_NS_NAME SD_STRINGIFY(SD_BACKEND_NAMESPACE)

/**
 * @def SD_BEGIN_NAMESPACE
 * @brief Begin the backend-specific namespace block
 */
#define SD_BEGIN_NAMESPACE namespace SD_NS {

/**
 * @def SD_END_NAMESPACE
 * @brief End the backend-specific namespace block
 */
#define SD_END_NAMESPACE }

/**
 * Backend-varying implementation types live in a nested ABI namespace while
 * remaining source-compatible through a selective sd:: alias. Unlike the
 * historical whole-namespace alias, this keeps shared framework types in sd
 * and isolates only state or behavior that genuinely differs by backend.
 */
#if defined(SD_BACKEND_NAMESPACE_CONFIGURED)
#define SD_BACKEND_ABI_NAMESPACE_BEGIN namespace sd { namespace SD_NS {
#define SD_BACKEND_ABI_NAMESPACE_END } }
#define SD_BACKEND_ABI_ALIAS(symbol) namespace sd { using symbol = SD_NS::symbol; }
#else
#define SD_BACKEND_ABI_NAMESPACE_BEGIN namespace sd {
#define SD_BACKEND_ABI_NAMESPACE_END }
#define SD_BACKEND_ABI_ALIAS(symbol)
#endif

// These macros are used from an existing parent namespace block. Configured
// native builds place ABI-varying classes and state in an inline child, so the
// public source spelling remains unchanged while ELF symbols carry sd_cpu,
// sd_cuda, or sd_vulkan. JavaCPP defines __JAVACPP_HACK__ only while parsing;
// keep that metadata surface flat. Generated JNI C++ is a configured native
// build and therefore binds to the same backend-qualified ABI as its library.
#if defined(SD_BACKEND_NAMESPACE_CONFIGURED) && !defined(__JAVACPP_HACK__)
// C++ requires a namespace's initial declaration to carry `inline` if any
// later declaration in the TU does ([namespace.def]). Headers open sd::SD_NS
// through both SD_BACKEND_ABI_NAMESPACE_BEGIN (non-inline spelling) and the
// inline macros below in arbitrary include order, so declare the backend child
// inline here, ahead of every use. Mangling is unaffected: inline and plain
// nested namespace members mangle identically, so this cannot skew the ABI.
namespace sd {
inline namespace SD_NS {}
}  // namespace sd
#define SD_BACKEND_INLINE_NAMESPACE_BEGIN inline namespace SD_NS {
#define SD_BACKEND_INLINE_NAMESPACE_END }
#define SD_BACKEND_INLINE_CLASS(symbol) SD_NS::symbol
#else
#define SD_BACKEND_INLINE_NAMESPACE_BEGIN
#define SD_BACKEND_INLINE_NAMESPACE_END
#define SD_BACKEND_INLINE_CLASS(symbol) symbol
#endif

#if defined(__JAVACPP_HACK__)
#define SD_BACKEND_ROOT_INLINE_NAMESPACE_BEGIN
#define SD_BACKEND_ROOT_INLINE_NAMESPACE_END
#define SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
#define SD_BACKEND_OPS_INLINE_NAMESPACE_END
#define SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN
#define SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
#define SD_BACKEND_PLATFORMS_CLASS(symbol) symbol
#else
#define SD_BACKEND_ROOT_INLINE_NAMESPACE_BEGIN SD_BACKEND_INLINE_NAMESPACE_BEGIN
#define SD_BACKEND_ROOT_INLINE_NAMESPACE_END SD_BACKEND_INLINE_NAMESPACE_END
#define SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN SD_BACKEND_INLINE_NAMESPACE_BEGIN
#define SD_BACKEND_OPS_INLINE_NAMESPACE_END SD_BACKEND_INLINE_NAMESPACE_END
#define SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN SD_BACKEND_INLINE_NAMESPACE_BEGIN
#define SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END SD_BACKEND_INLINE_NAMESPACE_END
#define SD_BACKEND_PLATFORMS_CLASS(symbol) SD_BACKEND_INLINE_CLASS(symbol)
#endif

// Out-of-line op definitions deliberately use the public spelling. In a
// configured native build inline-namespace lookup binds it to the backend
// child; in JavaCPP and unconfigured parser builds it stays directly in sd::ops.
#define SD_BACKEND_OPS_CLASS(symbol) sd::ops::symbol

/**
 * @def SD_NAMESPACE_USE
 * @brief 'using namespace' directive for the backend namespace
 */
#define SD_NAMESPACE_USE using namespace SD_NS

/**
 * For backward compatibility, we create 'sd' as an inline namespace alias.
 * This allows existing code using 'sd::' to work, while actually resolving
 * to the backend-specific namespace.
 *
 * NOTE: This only works within the same translation unit. For true isolation
 * across shared library boundaries, always use SD_NS or the full namespace name.
 */
#if defined(SD_ENABLE_NAMESPACE_ALIAS) && SD_ENABLE_NAMESPACE_ALIAS
    // Create 'sd' as an alias - this may cause issues if both libraries are linked
    namespace SD_NS {}
    namespace sd = SD_NS;
#endif

/**
 * @def SD_QUALIFIED(symbol)
 * @brief Fully qualify a symbol with the backend namespace
 *
 * Example: SD_QUALIFIED(ops::add_op) expands to sd_cuda::ops::add_op
 */
#define SD_QUALIFIED(symbol) SD_NS::symbol

/**
 * @def SD_OPS
 * @brief The ops sub-namespace
 */
#define SD_OPS SD_NS::ops

/**
 * @def SD_GRAPH
 * @brief The graph sub-namespace
 */
#define SD_GRAPH SD_NS::graph

/**
 * @def SD_HELPERS
 * @brief The helpers sub-namespace (platform-specific implementations)
 */
#define SD_HELPERS SD_NS::helpers

/**
 * Backend type identification macros
 * These are set by CMake to indicate which backend is being built
 */
#if defined(SD_CUDA)
    #define SD_IS_CUDA_BACKEND 1
    #define SD_IS_CPU_BACKEND 0
    #define SD_IS_TPU_BACKEND 0
    #define SD_IS_VULKAN_BACKEND 0
    #define SD_BACKEND_NAME "CUDA"
#elif defined(SD_BACKEND_TYPE_TPU)
    #define SD_IS_CUDA_BACKEND 0
    #define SD_IS_CPU_BACKEND 0
    #define SD_IS_TPU_BACKEND 1
    #define SD_IS_VULKAN_BACKEND 0
    #define SD_BACKEND_NAME "TPU"
#elif defined(SD_BACKEND_TYPE_VULKAN)
    #define SD_IS_CUDA_BACKEND 0
    #define SD_IS_CPU_BACKEND 0
    #define SD_IS_TPU_BACKEND 0
    #define SD_IS_VULKAN_BACKEND 1
    #define SD_BACKEND_NAME "VULKAN"
#else
    // Default to CPU
    #define SD_IS_CUDA_BACKEND 0
    #define SD_IS_CPU_BACKEND 1
    #define SD_IS_TPU_BACKEND 0
    #define SD_IS_VULKAN_BACKEND 0
    #define SD_BACKEND_NAME "CPU"
#endif

#endif // LIBND4J_BACKEND_NAMESPACE_H
