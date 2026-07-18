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

/*
 * Smoke test for the libsdx_llm C ABI (sdx_llm_c.h).
 *
 * Build:
 *   gcc -o sdx_llm_smoke sdx_llm_smoke.c -I../../../include -L<libdir> -lsdx_llm \
 *       -Wl,-rpath,<libdir>
 * Run:
 *   ./sdx_llm_smoke <model.gguf|model.sdz> [tokenizer-path] [prompt]
 *
 * Exercises: runtime create -> load -> info -> tokenize -> detokenize ->
 * generate -> stats -> unload -> destroy. Exits 0 on success.
 */

#include "sdx_llm_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_last_error(sdx_llm_runtime_t* rt, const char* where) {
  char buf[4096];
  sdxLlmGetLastError(rt, buf, (int32_t) sizeof(buf));
  fprintf(stderr, "FAIL %s: %s\n", where, buf);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model> [tokenizer] [prompt]\n", argv[0]);
    return 2;
  }
  const char* model_path = argv[1];
  const char* tokenizer_path = argc > 2 ? argv[2] : NULL;
  const char* prompt = argc > 3 ? argv[3] : "The capital of France is";

  sdx_llm_runtime_t* rt = sdxLlmCreateRuntime();
  if (!rt) {
    fprintf(stderr, "FAIL sdxLlmCreateRuntime returned NULL\n");
    return 1;
  }
  printf("abi=%d\n", sdxLlmAbiVersion(rt));

  sdx_llm_model_t* model = sdxLlmLoadModel(rt, model_path, tokenizer_path,
      "{\"maxNewTokens\":8,\"sampling\":{\"preset\":\"greedy\"}}");
  if (!model) {
    print_last_error(rt, "sdxLlmLoadModel");
    sdxLlmDestroyRuntime(rt);
    return 1;
  }

  char* info = NULL;
  if (sdxLlmInfoJson(rt, model, &info) == SDX_LLM_STATUS_OK) {
    printf("info=%s\n", info);
    sdxLlmFree(rt, info);
  }

  int32_t* ids = NULL;
  int32_t count = 0;
  if (sdxLlmTokenize(rt, model, prompt, 1, &ids, &count) != SDX_LLM_STATUS_OK) {
    print_last_error(rt, "sdxLlmTokenize");
    return 1;
  }
  printf("tokens=%d [", count);
  for (int32_t i = 0; i < count; i++) printf(i ? ",%d" : "%d", ids[i]);
  printf("]\n");

  char* roundtrip = NULL;
  if (sdxLlmDetokenize(rt, model, ids, count, 1, &roundtrip) != SDX_LLM_STATUS_OK) {
    print_last_error(rt, "sdxLlmDetokenize");
    return 1;
  }
  printf("roundtrip=%s\n", roundtrip);
  sdxLlmFree(rt, roundtrip);
  sdxLlmFree(rt, (void*) ids);

  char* text = NULL;
  if (sdxLlmGenerate(rt, model, prompt, NULL, &text) != SDX_LLM_STATUS_OK) {
    print_last_error(rt, "sdxLlmGenerate");
    return 1;
  }
  printf("generated=%s\n", text);
  sdxLlmFree(rt, text);

  char* stats = NULL;
  if (sdxLlmLastResultJson(rt, model, &stats) == SDX_LLM_STATUS_OK) {
    printf("stats=%s\n", stats);
    sdxLlmFree(rt, stats);
  }

  if (sdxLlmUnloadModel(rt, model) != SDX_LLM_STATUS_OK) {
    print_last_error(rt, "sdxLlmUnloadModel");
    return 1;
  }
  if (sdxLlmDestroyRuntime(rt) != 0) {
    fprintf(stderr, "FAIL sdxLlmDestroyRuntime\n");
    return 1;
  }
  printf("OK\n");
  return 0;
}
