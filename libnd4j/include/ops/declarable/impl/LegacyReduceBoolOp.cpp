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
// Created by raver119 on 16.10.2017.
//
#include <array/DataTypeUtils.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/LegacyReduceBoolOp.h>

namespace sd {
namespace ops {
LegacyReduceBoolOp::LegacyReduceBoolOp() : LegacyOp::LegacyOp(1) {
  //
}

LegacyReduceBoolOp::LegacyReduceBoolOp(int opNum) : LegacyOp::LegacyOp(1, opNum) {
  // this->_opNum = opNum;
}

LegacyOp* LegacyReduceBoolOp::clone() { return new LegacyReduceBoolOp(this->_opNum); }

sd::Status LegacyReduceBoolOp::validateAndExecute(Context& block) {
  auto x = INPUT_VARIABLE(0);

  auto z = OUTPUT_VARIABLE(0);

  NDArray::prepareSpecialUse({z}, {x});

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();
  sd_debug("Executing LegacyReduceFloatOp: [%i]\n", opNum);

  auto axis = *block.getAxis();

  bool allAxes = false;

  ExtraArguments extras(*block.getTArguments());
  PointersManager manager(block.launchContext(), "LegacyReduceBoolOp");

  if (block.width() == 1) {
    if (axis.size() == x->rankOf()) allAxes = true;

    if ((axis.empty()) || (axis.size() == 1 && axis[0] == sd::DataTypeUtils::max<int>()) || allAxes) {
      // scalar
      NativeOpExecutioner::execReduceBoolScalar(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    } else {
      // TAD
      std::vector<int> dims(axis);

      for (int e = 0; e < dims.size(); e++)
        if (dims[e] < 0) dims[e] += x->rankOf();

      REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

      const sd::LongType* zShapeInfoH = z->shapeInfo();
      const sd::LongType* zShapeInfoD = z->specialShapeInfo();

      if (x->rankOf() - dims.size() != z->rankOf()) {
        auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
            z->shapeInfo(), dims, z->getContext()->getWorkspace());
        zShapeInfoH = reinterpret_cast<sd::LongType const*>(zPack.primary());
        zShapeInfoD = reinterpret_cast<sd::LongType const*>(zPack.special());
      }

      std::vector<int> dims2 = ShapeUtils::evalDimsForReduceOp(x->rankOf(), dims);
      NativeOpExecutioner::execReduceBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), nullptr, z->buffer(), zShapeInfoH, z->specialBuffer(),
                                          zShapeInfoD, dims2.data(), dims2.size());

      // auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dims);

      // auto pTadShape = Environment::getInstance().isCPU() ? packX.primaryShapeInfo() : packX.specialShapeInfo();
      // //manager.replicatePointer(tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo)); auto
      // pTadOffsets = Environment::getInstance().isCPU() ? packX.primaryOffsets() : packX.specialOffsets();
      // //manager.replicatePointer(tad.tadOffsets, tad.numTads * sizeof(sd::LongType));

      // NativeOpExecutioner::execReduceBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(),
      // x->specialBuffer(), x->specialShapeInfo(),
      //         extras.argumentsAsT(x->dataType()),
      //         z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
      //         dims.data(), (int) dims.size(), reinterpret_cast<sd::LongType const*>(pTadShape),
      //         reinterpret_cast<sd::LongType const*>(pTadOffsets));
    }

    STORE_RESULT(*z);
  } else {
    auto indices = INPUT_VARIABLE(1);
    if (indices->lengthOf() == x->rankOf()) allAxes = true;


    std::vector<int> dims(indices->lengthOf());
    for (sd::LongType e = 0; e < indices->lengthOf(); e++) {
      //segfault on macOS
      int f = indices->e<int>(e);
      dims[e] = f >= 0 ? f : f += x->rankOf();
    }

    if ((block.getIArguments()->size() == 1 && INT_ARG(0) == sd::DataTypeUtils::max<int>()) || allAxes) {
      // scalar
      NativeOpExecutioner::execReduceBoolScalar(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    } else {
      // TAD
      if (indices->lengthOf() > 1) std::sort(dims.begin(), dims.end());

      REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

      const sd::LongType* zShapeInfoH = z->shapeInfo();
      const sd::LongType* zShapeInfoD = z->specialShapeInfo();

      if (x->rankOf() - dims.size() != z->rankOf()) {
        auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
            z->shapeInfo(), dims, z->getContext()->getWorkspace());
        zShapeInfoH = reinterpret_cast<sd::LongType const*>(zPack.primary());
        zShapeInfoD = reinterpret_cast<sd::LongType const*>(zPack.special());
      }

      std::vector<int> dims2 = ShapeUtils::evalDimsForReduceOp(x->rankOf(), dims);
      NativeOpExecutioner::execReduceBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), nullptr, z->buffer(), zShapeInfoH, z->specialBuffer(),
                                          zShapeInfoD, dims2.data(), dims2.size());

      // auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dims);

      // auto pTadShape = Environment::getInstance().isCPU() ? packX.primaryShapeInfo() : packX.specialShapeInfo();
      // //(sd::LongType *) manager.replicatePointer(tad.tadOnlyShapeInfo,
      // shape::shapeInfoByteLength(tad.tadOnlyShapeInfo)); auto pTadOffsets = Environment::getInstance().isCPU() ?
      // packX.primaryOffsets() : packX.specialOffsets(); //(sd::LongType *) manager.replicatePointer(tad.tadOffsets,
      // tad.numTads * sizeof(sd::LongType));

      // NativeOpExecutioner::execReduceBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(),
      // x->specialBuffer(), x->specialShapeInfo(), extras.argumentsAsT(x->dataType()),
      //         z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), dims.data(), (int) dims.size(),
      //         pTadShape, pTadOffsets);
    }
  }

  manager.synchronize();
  return sd::Status::OK;
}

/**
 *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
 *   It solely depends on input shape, and requested dimensions
 */
ShapeList* LegacyReduceBoolOp::calculateOutputShape(ShapeList* inputShape, sd::graph::Context& block) {
  auto inShape = inputShape->at(0);

  sd::LongType* newShape;

  bool allAxes = false;

  auto keepDims = block.numB() > 0 ? B_ARG(0) : false;
  auto newFormat = block.numB() > 1 ? B_ARG(1) : true;

  auto axis = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<int>() : *block.getAxis();

  if (axis.size() == shape::rank(inShape)) allAxes = true;

  // in this case we're building proper shape for reduction
  return SHAPELIST(ShapeUtils::evalReduceShapeInfo(shape::order(inShape), axis, inShape, DataType::BOOL, keepDims,
                                                   !newFormat, block.workspace()));
}
}  // namespace ops
}  // namespace sd
