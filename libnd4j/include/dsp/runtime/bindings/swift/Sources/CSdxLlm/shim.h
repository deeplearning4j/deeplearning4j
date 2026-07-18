#ifndef CSDXLLM_SHIM_H
#define CSDXLLM_SHIM_H

/*
 * Resolve sdx_llm_c.h across supported layouts:
 *  - unpacked AOT SDK:   <sdk-root>/include/sdx_llm_c.h
 *    Supply -Xcc -I<sdk-root>/include to swift build.
 *  - libnd4j source tree: nd4j/sdx-aot/include/sdx_llm_c.h
 *    This package lives at libnd4j/include/dsp/runtime/bindings/swift/,
 *    so the source-tree relative path is ../../../../../nd4j/sdx-aot/include/.
 *
 * Build command when using the source tree:
 *   swift build -Xcc -I$(git rev-parse --show-toplevel)/nd4j/sdx-aot/include \
 *               -Xlinker -L$SDX_LLM_AOT_HOME/lib
 */
#if __has_include("sdx_llm_c.h")
  /* SDK include/ dir is already on the search path. */
  #include "sdx_llm_c.h"
#elif __has_include("../../../../../nd4j/sdx-aot/include/sdx_llm_c.h")
  /* Source-tree path: package at libnd4j/include/dsp/runtime/bindings/swift. */
  #include "../../../../../nd4j/sdx-aot/include/sdx_llm_c.h"
#else
  #error "sdx_llm_c.h not found. Pass -Xcc -I<sdk-root>/include to swift build."
#endif

#endif /* CSDXLLM_SHIM_H */
