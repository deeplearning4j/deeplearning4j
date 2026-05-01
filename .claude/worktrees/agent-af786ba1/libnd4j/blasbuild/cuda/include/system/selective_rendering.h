#ifndef SD_SELECTIVE_RENDERING_H
#define SD_SELECTIVE_RENDERING_H

// ============================================================================
// Selective Rendering Type System - Partitioned Headers
// ============================================================================
// This master header includes all type category headers.
// Large translation units can include only the specific category headers
// they need to avoid Clang source location limits.
// ============================================================================

// Core type mappings (always required)
#include "selective_rendering/core.h"

// Type category headers
#include "selective_rendering/bool_types.h"
#include "selective_rendering/float_types.h"
#include "selective_rendering/bfloat_types.h"
#include "selective_rendering/int_types.h"
#include "selective_rendering/uint_types.h"
#include "selective_rendering/string_types.h"

#define SD_BUILD_TRIPLE_IF_VALID(t1, t2, t3, build_macro) \
    do { \
        if (SD_IS_TRIPLE_TYPE_COMPILED(t1, t2, t3)) { \
            SD_DISPATCH_TRIPLE_RUNTIME(t1, t2, t3, build_macro); \
        } \
    } while(0)

#define SD_BUILD_PAIR_IF_VALID(t1, t2, build_macro) \
    do { \
        if (SD_IS_PAIR_TYPE_COMPILED(t1, t2)) { \
            SD_DISPATCH_PAIR_RUNTIME(t1, t2, build_macro); \
        } \
    } while(0)

#define SD_BUILD_SINGLE_IF_VALID(t1, build_macro) \
    do { \
        if (SD_IS_SINGLE_TYPE_COMPILED(t1)) { \
            SD_DISPATCH_SINGLE_RUNTIME(t1, build_macro); \
        } \
    } while(0)

#endif // SD_SELECTIVE_RENDERING_H
