/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_kv_scatter)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/kv_scatter.h>

namespace sd {
namespace ops {

// kv_scatter: copies present[..., lastPos, :] -> output[..., cachePos, :]
//
// Input 0: static_kv  [batch, heads, maxKvLen, dim]
// Input 1: present_kv [batch, heads, seqLen, dim]
// IArg 0: cachePos
// Output 0: same shape as input 0
CUSTOM_OP_IMPL(kv_scatter, 2, 1, false, 0, 1) {
  auto present = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);
  auto cachePos = INT_ARG(0);

  REQUIRE_TRUE(present->rankOf() == 4, 0,
               "kv_scatter: present must be rank 4, got %d", present->rankOf());
  REQUIRE_TRUE(output->rankOf() == 4, 0,
               "kv_scatter: output must be rank 4, got %d", output->rankOf());
  REQUIRE_TRUE(present->sizeAt(0) == output->sizeAt(0), 0,
               "kv_scatter: batch mismatch");
  REQUIRE_TRUE(present->sizeAt(1) == output->sizeAt(1), 0,
               "kv_scatter: heads mismatch");
  REQUIRE_TRUE(present->sizeAt(3) == output->sizeAt(3), 0,
               "kv_scatter: dim mismatch");
  REQUIRE_TRUE(cachePos >= 0 && cachePos < output->sizeAt(2), 0,
               "kv_scatter: cachePos %lld out of range [0, %lld)",
               (long long)cachePos, (long long)output->sizeAt(2));

  helpers::kvScatter(present, output, cachePos, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(kv_scatter) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true);
}

DECLARE_SHAPE_FN(kv_scatter) {
  // Output shares input 0's buffer — set ARRAY_COPY_OFFSET_INPUT_0 so Java
  // creates the output as a view of input[0] (the static KV cache buffer).
  auto inShape = inputShape->at(0);
  auto rank = shape::rank(inShape);
  auto dtype = ArrayOptions::dataType(inShape);

  auto newShapeInfo = new LongType[shape::shapeInfoLength(rank)];
  memcpy(newShapeInfo, inShape, shape::shapeInfoByteLength(rank));
  ArrayOptions::resetFlags(newShapeInfo);
  ArrayOptions::setDataType(newShapeInfo, dtype);
  ArrayOptions::togglePropertyBit(newShapeInfo, ARRAY_COPY_OFFSET_INPUT_0);

  auto result = ConstantShapeHelper::getInstance().createFromExisting(newShapeInfo);
  delete[] newShapeInfo;
  return SHAPELIST(result);
}

}  // namespace ops
}  // namespace sd

#endif
