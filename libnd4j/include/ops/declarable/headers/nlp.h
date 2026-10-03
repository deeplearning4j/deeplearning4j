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
//  @author raver119@gmail.com
//

#ifndef DEV_TESTS_NLP_H
#define DEV_TESTS_NLP_H
#include <ops/declarable/headers/common.h>

namespace sd {
namespace ops {

/**
 * Word2vec rounds. skipgram and cbow train syn0, syn1 and syn1Neg, or - with an inference vector - only that vector;
 * the *_inference forms carry their codes, indices and context in integer arguments instead of arrays.
 *
 * The arrays a configuration does not use are passed as EMPTY arrays: ngStarter, syn1, syn1Neg, negTable, indices,
 * codes, numLabels and the inference vector (empty means training) all mean "absent". The default EMPTY_SKIP policy
 * returned OK for any empty input without running the op, which left every table untouched, so these ops run with
 * EMPTY_EXECUTE and treat an empty optional input as absent.
 */
#if NOT_EXCLUDED(OP_skipgram)
// Expanded from DECLARE_CONFIGURABLE_OP(skipgram, 12, 12, true, 0, 0) to override emptyHandling() = EMPTY_EXECUTE.
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
class SD_LIB_EXPORT skipgram : public sd::ops::DeclarableOp {
 public:
  skipgram();
  sd::ShapeList* calculateOutputShape(sd::ShapeList* inputShape, sd::graph::Context& block);
  samediff::EmptyHandling emptyHandling() override { return samediff::EmptyHandling::EMPTY_EXECUTE; }

 protected:
  void registerTypes();
  SD_DECLARABLE_OP_EXECUTION_METHODS
};
SD_BACKEND_OPS_INLINE_NAMESPACE_END
REGISTER_H(skipgram)
#endif

#if NOT_EXCLUDED(OP_skipgram_inference)
// Expanded from DECLARE_CONFIGURABLE_OP(skipgram_inference, 6, 6, true, -2, -2) to override emptyHandling() =
// EMPTY_EXECUTE (syn1, syn1Neg, negTable and the inference vector may be empty).
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
class SD_LIB_EXPORT skipgram_inference : public sd::ops::DeclarableOp {
 public:
  skipgram_inference();
  sd::ShapeList* calculateOutputShape(sd::ShapeList* inputShape, sd::graph::Context& block);
  samediff::EmptyHandling emptyHandling() override { return samediff::EmptyHandling::EMPTY_EXECUTE; }

 protected:
  void registerTypes();
  SD_DECLARABLE_OP_EXECUTION_METHODS
};
SD_BACKEND_OPS_INLINE_NAMESPACE_END
REGISTER_H(skipgram_inference)
#endif

#if NOT_EXCLUDED(OP_cbow)
// Expanded from DECLARE_CONFIGURABLE_OP(cbow, 15, 15, true, 0, 0) to override emptyHandling() = EMPTY_EXECUTE.
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
class SD_LIB_EXPORT cbow : public sd::ops::DeclarableOp {
 public:
  cbow();
  sd::ShapeList* calculateOutputShape(sd::ShapeList* inputShape, sd::graph::Context& block);
  samediff::EmptyHandling emptyHandling() override { return samediff::EmptyHandling::EMPTY_EXECUTE; }

 protected:
  void registerTypes();
  SD_DECLARABLE_OP_EXECUTION_METHODS
};
SD_BACKEND_OPS_INLINE_NAMESPACE_END
REGISTER_H(cbow)
#endif

#if NOT_EXCLUDED(OP_cbow_inference)
// Expanded from DECLARE_CONFIGURABLE_OP(cbow_inference, 6, 6, true, -2, -2) to override emptyHandling() =
// EMPTY_EXECUTE (syn1, syn1Neg, negTable and the inference vector may be empty).
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
class SD_LIB_EXPORT cbow_inference : public sd::ops::DeclarableOp {
 public:
  cbow_inference();
  sd::ShapeList* calculateOutputShape(sd::ShapeList* inputShape, sd::graph::Context& block);
  samediff::EmptyHandling emptyHandling() override { return samediff::EmptyHandling::EMPTY_EXECUTE; }

 protected:
  void registerTypes();
  SD_DECLARABLE_OP_EXECUTION_METHODS
};
SD_BACKEND_OPS_INLINE_NAMESPACE_END
REGISTER_H(cbow_inference)
#endif
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTS_NLP_H
