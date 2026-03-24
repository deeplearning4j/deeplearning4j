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
//  @author Adam Gibson
//

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_where_np)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/headers/boolean.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(where_np, -1, 1, false, 0, 0) {
  auto condition = INPUT_VARIABLE(0);

  if (block.width() == 3) {
    auto x = INPUT_VARIABLE(1);
    auto y = INPUT_VARIABLE(2);

    auto z = OUTPUT_VARIABLE(0);
    int numMatches = 0;
    // if cond matches x/y shape - we have per-element mask
    if (condition->isSameShape(x)) {
      // Sync all inputs to host for direct buffer access, avoiding O(n^2) sync from per-element p()/e()
      condition->syncToHost();
      x->syncToHost();
      y->syncToHost();

      auto condBuf = condition->bufferAsT<bool>();
      auto len = condition->lengthOf();

      if (y->isScalar()) {
        if (y->isR()) {
#ifdef HAS_DOUBLE
          auto xBuf = x->bufferAsT<double>();
          auto zBuf = z->bufferAsT<double>();
          auto yVal = y->bufferAsT<double>()[0];
          for (int e = 0; e < len; e++) {
            zBuf[e] = condBuf[e] ? yVal : xBuf[e];
          }
#elif defined(HAS_FLOAT32)
          auto xBuf = x->bufferAsT<float>();
          auto zBuf = z->bufferAsT<float>();
          auto yVal = y->bufferAsT<float>()[0];
          for (int e = 0; e < len; e++) {
            zBuf[e] = condBuf[e] ? yVal : xBuf[e];
          }
#else
#error "No floating-point type available for where_np operation"
#endif
        } else {
          auto xBuf = x->bufferAsT<LongType>();
          auto zBuf = z->bufferAsT<LongType>();
          auto yVal = y->bufferAsT<LongType>()[0];
          for (int e = 0; e < len; e++) {
            zBuf[e] = condBuf[e] ? yVal : xBuf[e];
          }
        }
      } else {
        if (y->isR()) {
#ifdef HAS_DOUBLE
          auto xBuf = x->bufferAsT<double>();
          auto yBuf = y->bufferAsT<double>();
          auto zBuf = z->bufferAsT<double>();
          for (int e = 0; e < len; e++) {
            if (condBuf[e]) {
              zBuf[e] = yBuf[numMatches];
              numMatches++;
            } else {
              zBuf[e] = xBuf[e];
            }
          }
#elif defined(HAS_FLOAT32)
          auto xBuf = x->bufferAsT<float>();
          auto yBuf = y->bufferAsT<float>();
          auto zBuf = z->bufferAsT<float>();
          for (int e = 0; e < len; e++) {
            if (condBuf[e]) {
              zBuf[e] = yBuf[numMatches];
              numMatches++;
            } else {
              zBuf[e] = xBuf[e];
            }
          }
#else
#error "No floating-point type available for where_np operation"
#endif
        } else {
          auto xBuf = x->bufferAsT<LongType>();
          auto yBuf = y->bufferAsT<LongType>();
          auto zBuf = z->bufferAsT<LongType>();
          for (int e = 0; e < len; e++) {
            if (condBuf[e]) {
              zBuf[e] = yBuf[numMatches];
              numMatches++;
            } else {
              zBuf[e] = xBuf[e];
            }
          }
        }
      }

      z->tickWriteHost();
      z->syncToDevice();
    } else {
      REQUIRE_TRUE(condition->lengthOf() == x->sizeAt(0), 0,
                   "Condition length should be equal to the dim0 of x/y to act as TAD-mask, but got %d instead",
                   condition->lengthOf());

      std::vector<LongType> idxs;
      idxs.push_back(0);
      auto dims = ShapeUtils::evalDimsToExclude(x->rankOf(), 1,idxs.data());
      auto tadsX = x->allTensorsAlongDimension(*dims);
      auto tadsY = y->allTensorsAlongDimension(*dims);
      auto tadsZ = z->allTensorsAlongDimension(*dims);

      for (int e = 0; e < tadsX.size(); e++) {
        if (!condition->e<bool>(e))
          tadsZ.at(e)->assign(tadsY.at(e));
        else
          tadsZ.at(e)->assign(tadsX.at(e));
      }

      delete dims;
    }
  } else {
    // in this case we return 2D matrix, which basically contains coordinates fo true

    REQUIRE_TRUE(block.width() == 1, 0, "Where op takes either 1 or 3 operands, But got %d operands instead",
                 block.width());
    LongType width = condition->rankOf();

    Where op;
    auto res(op.evaluate({condition}));
    REQUIRE_OK(res.status());
    NDArray* whereTrue = res.at(0);

    if (whereTrue->isEmpty()) return Status::OK;

    // Sync whereTrue to host for direct buffer access, avoiding O(n^2) sync from per-element p()/e()
    whereTrue->syncToHost();

    for (LongType outNext = 0; outNext < width; ++outNext) {
      auto output = OUTPUT_VARIABLE(outNext);
      auto outBuf = output->bufferAsT<LongType>();
      auto outLen = output->lengthOf();
      for (LongType e = 0; e < outLen; ++e) {
        // whereTrue is 2D [numTrue, rank], read element at (e, outNext)
        outBuf[e] = whereTrue->e<LongType>(e, outNext);
      }
      output->tickWriteHost();
      output->syncToDevice();
    }
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(where_np) {
  auto shapes = SHAPELIST();
  if (block.width() == 3) {
    auto inShape = inputShape->at(1);
    shapes->push_back(CONSTANT(inShape));
  } else {
    auto condition = INPUT_VARIABLE(0);

    LongType numOfTrue = 0LL;  // condition->reduceNumber(reduce::CountNonZero).e<sd::LongType>(0);
    for (LongType i = 0; i < condition->lengthOf(); ++i)
      if (condition->e<bool>(i)) numOfTrue++;

    // output shape - a tuple of rank(inShape) 1D tensors with numOfTrue len
    if (numOfTrue) {
      for (LongType e = 0; e < condition->rankOf(); ++e) {
        shapes->push_back(ConstantShapeHelper::getInstance().vectorShapeInfo(numOfTrue, INT64));
      }
    } else {
      shapes->push_back(ConstantShapeHelper::getInstance().emptyShapeInfo(INT64));
    }
  }
  return shapes;
}

DECLARE_TYPES(where_np) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, BOOL)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedInputTypes(2, ANY)
      ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS});
}
}  // namespace ops
}  // namespace sd

#endif
