#ifndef CSDXRUNTIME_SHIM_H
#define CSDXRUNTIME_SHIM_H

/*
 * Resolve dsp_runtime_c.h across the two supported layouts:
 *  - staged SDK package:  <sdk-root>/include/dsp/runtime/dsp_runtime_c.h
 *    (this package lives at <sdk-root>/wrappers/swift/ or
 *     <sdk-root>/bindings/<platform>/<variant>/wrappers/swift/)
 *  - libnd4j source tree: libnd4j/include/dsp/runtime/dsp_runtime_c.h
 *    (this package lives at .../dsp/runtime/bindings/swift/)
 */
#if __has_include("../../../../include/dsp/runtime/dsp_runtime_c.h")
#include "../../../../include/dsp/runtime/dsp_runtime_c.h"
#elif __has_include("../../../../dsp_runtime_c.h")
#include "../../../../dsp_runtime_c.h"
#elif __has_include(<dsp/runtime/dsp_runtime_c.h>)
#include <dsp/runtime/dsp_runtime_c.h>
#else
#error "dsp_runtime_c.h not found. Add the SDK include dir to the search path (e.g. swift build -Xcc -I<sdk-root>/include)."
#endif

#endif /* CSDXRUNTIME_SHIM_H */
