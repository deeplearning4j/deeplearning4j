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

//
// Shared OneDNN eltwise (unary activation / math) helpers and wrapper macros.
//
// Every OneDNN eltwise op wrapper was a near-verbatim copy of the same three
// zones (extract -> validate -> one eltwise_forward/backward primitive), differing
// only by the dnnl::algorithm constant and the alpha/beta parameters. This header
// factors the primitive plumbing into two shared workers and stamps the
// PLATFORM_IMPL/PLATFORM_CHECK pair from a single macro line per op. The oneDNN
// kernel path is unchanged — only the per-op boilerplate is collapsed.
//

#ifndef LIBND4J_MKLDNN_ELTWISE_H
#define LIBND4J_MKLDNN_ELTWISE_H

#include "mkldnnUtils.h"

#include <system/BackendNamespace.h>

namespace sd {
SD_BACKEND_ROOT_INLINE_NAMESPACE_BEGIN
namespace onednnUtils {

/**
 * Generic OneDNN eltwise forward: z = alg(x; alpha, beta). FLOAT32, rank <= 6.
 * Generalizes the per-op forward workers (reluMKLDNN, eluMKLDNN, ...).
 */
void eltwiseFwdMKLDNN(NDArray* x, NDArray* z, dnnl::algorithm alg, float alpha, float beta);

/**
 * Generic OneDNN eltwise backward: dLdx = alg'(x) * dLdz; alpha/beta match the
 * forward hint. Generalizes the per-op backward workers (reluBpMKLDNN, ...).
 */
void eltwiseBpMKLDNN(NDArray* x, NDArray* dLdz, NDArray* dLdx, dnnl::algorithm alg, float alpha, float beta);

/** Map an sd DataType to the OneDNN memory data_type (f32/bf16/f16; f32 fallback). */
dnnl::memory::data_type toDnnlDataType(sd::DataType dt);

/**
 * True if OneDNN eltwise can run this dtype on the current CPU ISA. f32 is always
 * supported; bf16 needs AVX512_CORE_BF16; f16 needs AVX512_CORE_AMX_FP16. When it
 * returns false the PLATFORM_CHECK fails and the op falls back to the generic kernel
 * (rather than feeding an unsupported dtype into a f32-only primitive). Mirrors the
 * conv2d/batchnorm reference pattern.
 */
bool isSupportedEltwiseType(sd::DataType dt);

}  // namespace onednnUtils
SD_BACKEND_ROOT_INLINE_NAMESPACE_END
}  // namespace sd

//////////////////////////////////////////////////////////////////////
// Forward eltwise op: emits one PLATFORM_IMPL + PLATFORM_CHECK pair.
//   OPNAME     - op-name token (must have a matching DECLARE_PLATFORM(OPNAME, ENGINE_ONEDNN))
//   DISPLAY    - string literal used in diagnostics, e.g. "RELU"
//   ALG        - dnnl::algorithm::eltwise_*
//   ALPHA_EXPR - expression yielding the alpha param (may reference `block`)
//   BETA_EXPR  - expression yielding the beta param
#define DEFINE_MKLDNN_ELTWISE_FWD(OPNAME, DISPLAY, ALG, ALPHA_EXPR, BETA_EXPR)                          \
  PLATFORM_IMPL(OPNAME, ENGINE_ONEDNN) {                                                                   \
    auto input = INPUT_VARIABLE(0);                                                                     \
    auto output = OUTPUT_VARIABLE(0);                                                                   \
    const sd::LongType rank = input->rankOf();                                                          \
    REQUIRE_TRUE(rank <= 6, 0, DISPLAY "_MKLDNN OP: the rank of input must be <= 6, but got %i", rank); \
    sd::onednnUtils::eltwiseFwdMKLDNN(input, output, ALG, (ALPHA_EXPR), (BETA_EXPR));                   \
    return sd::Status::OK;                                                                              \
  }                                                                                                     \
  PLATFORM_CHECK(OPNAME, ENGINE_ONEDNN) {                                                                  \
    auto x = INPUT_VARIABLE(0);                                                                         \
    auto z = OUTPUT_VARIABLE(0);                                                                        \
    Requirements req("ONEDNN " DISPLAY " OP");                                                          \
    req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&              \
        req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&                             \
        req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&                          \
        req.expectTrue(makeInfoVariable(sd::onednnUtils::isSupportedEltwiseType(x->dataType()), TYPE_MSG_INPUT), EXPECTED_TRUE) && \
        req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), x->dataType());                  \
    req.logTheSuccess();                                                                                \
    return req;                                                                                         \
  }

//////////////////////////////////////////////////////////////////////
// Backward eltwise op: dLdx from (x, dLdz). Two inputs, one output.
// Same parameters as the forward macro; the fwd hint uses ALG/ALPHA/BETA too.
#define DEFINE_MKLDNN_ELTWISE_BP(OPNAME, DISPLAY, ALG, ALPHA_EXPR, BETA_EXPR)                             \
  PLATFORM_IMPL(OPNAME, ENGINE_ONEDNN) {                                                                     \
    auto input = INPUT_VARIABLE(0);                                                                       \
    auto dLdz = INPUT_VARIABLE(1);                                                                        \
    auto dLdx = OUTPUT_VARIABLE(0);                                                                       \
    const sd::LongType rank = input->rankOf();                                                            \
    const sd::LongType dLdzRank = dLdz->rankOf();                                                         \
    REQUIRE_TRUE(rank <= 6 && dLdzRank <= 6, 0,                                                           \
                 DISPLAY "_BP_MKLDNN OP: the rank of input and dLdz must be <= 6, but got %i and %i",     \
                 rank, dLdzRank);                                                                         \
    sd::onednnUtils::eltwiseBpMKLDNN(input, dLdz, dLdx, ALG, (ALPHA_EXPR), (BETA_EXPR));                  \
    return sd::Status::OK;                                                                                \
  }                                                                                                       \
  PLATFORM_CHECK(OPNAME, ENGINE_ONEDNN) {                                                                    \
    auto x = INPUT_VARIABLE(0);                                                                           \
    auto dLdz = INPUT_VARIABLE(1);                                                                        \
    auto dLdx = OUTPUT_VARIABLE(0);                                                                       \
    Requirements req("ONEDNN " DISPLAY " BP OP");                                                         \
    req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&               \
        req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&        \
        req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&                              \
        req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 0) &&                           \
        req.expectTrue(makeInfoVariable(sd::onednnUtils::isSupportedEltwiseType(x->dataType()), TYPE_MSG_INPUT0), EXPECTED_TRUE) && \
        req.expectEq(makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1), x->dataType()) &&               \
        req.expectEq(makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT), x->dataType()) &&               \
        req.expect(                                                                                       \
            makeShapeInfoVariable(x, SHAPE_MSG_INPUT0), makeShapeInfoVariable(dLdz, SHAPE_MSG_INPUT1),    \
            [](const decltype(x)& l, const decltype(dLdz)& r) {                                           \
              for (int i = 0; i < l->rankOf(); i++) {                                                     \
                if (l->sizeAt(i) != r->sizeAt(i)) {                                                       \
                  return false;                                                                           \
                }                                                                                         \
              }                                                                                           \
              return true;                                                                               \
            },                                                                                            \
            EXPECTED_EQ_MSG);                                                                             \
    req.logTheSuccess();                                                                                  \
    return req;                                                                                           \
  }

#endif  // LIBND4J_MKLDNN_ELTWISE_H
