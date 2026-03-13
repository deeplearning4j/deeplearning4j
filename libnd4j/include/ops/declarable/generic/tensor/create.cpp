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
// @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_shapes_of)

#include <ops/declarable/headers/shape.h>
#include <cstdio>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(create, 1, 1, false, 0, 1) {
  auto init = block.numB() > 0 ? B_ARG(0) : true;

  if (init) OUTPUT_VARIABLE(0)->nullify();

  return Status::OK;
}

DECLARE_SHAPE_FN(create) {
  auto shapeInput = INPUT_VARIABLE(0);
  auto order = (char)INT_ARG(0);
  auto dtype = DataTypeUtils::fromInt(INT_ARG(1));

  REQUIRE_TRUE(order == 'c' || order == 'f', 0, "create: order must be either c or f");

  // Read shape values directly from synced host buffer to avoid e<T>() bugs on CUDA.
  shapeInput->syncToHost();
  std::vector<LongType> shape;
  auto* db = shapeInput->dataBuffer();
  auto numEl = shapeInput->lengthOf();
  auto inputDtype = shapeInput->dataType();
  auto arrOffset = shapeInput->offset();
  if (inputDtype == INT64) {
    auto* hostLongs = reinterpret_cast<LongType*>(db->primary()) + arrOffset;
    for (sd::LongType i = 0; i < numEl; i++)
      shape.push_back(hostLongs[i]);
  } else if (inputDtype == INT32) {
    auto* hostInts = reinterpret_cast<int*>(db->primary()) + arrOffset;
    for (sd::LongType i = 0; i < numEl; i++)
      shape.push_back(static_cast<LongType>(hostInts[i]));
  } else {
    shape = shapeInput->getBufferAsVector<LongType>();
  }

  // DIAGNOSTIC: dump create shape info to file
  {
    FILE* f = fopen("/tmp/create_diag.txt", "a");
    if (f) {
      fprintf(f, "CREATE shape_fn: numEl=%lld, inputDtype=%d, offset=%lld, primary=%p\n",
              (long long)numEl, (int)inputDtype, (long long)arrOffset, (void*)db->primary());
      fprintf(f, "  shape values: [");
      for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%lld", (long long)shape[i]);
      }
      fprintf(f, "]\n");
      fclose(f);
    }
  }

  return SHAPELIST(sd::ConstantShapeHelper::getInstance().createShapeInfo(dtype, order, shape));
}

DECLARE_TYPES(create) { getOpDescriptor()->setAllowedInputTypes({ALL_INTS})->setAllowedOutputTypes(ANY); }
}  // namespace ops
}  // namespace sd

#endif
