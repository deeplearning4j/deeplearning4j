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
#include <ops/declarable/LegacyReduceLongOp.h>

#include <ops/declarable/OpRegistrator.h>

namespace sd {
namespace ops {
LegacyReduceLongOp::LegacyReduceLongOp() : LegacyOp::LegacyOp(1) {
  //
}

LegacyReduceLongOp::LegacyReduceLongOp(int opNum) : LegacyOp::LegacyOp(1, opNum) {
  // this->_opNum = opNum;
}

LegacyOp* LegacyReduceLongOp::clone() { return new LegacyReduceLongOp(this->_opNum); }

sd::Status LegacyReduceLongOp::validateAndExecute(Context& block) {
  auto x = INPUT_VARIABLE(0);

  auto z = OUTPUT_VARIABLE(0);

  NDArray::prepareSpecialUse({z}, {x});

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();
  sd_debug("Executing LegacyReduceFloatOp: [%i]\n", opNum);

  auto axis = *block.getAxis();
  bool allAxes = false;

  ExtraArguments extras(*block.getTArguments());
  PointersManager manager(block.launchContext(), "LegacyReduceLongOp");

  if (block.width() == 1) {
    if (axis.size() == x->rankOf()) allAxes = true;

    if ((axis.empty()) || (axis.size() == 1 && axis[0] == sd::DataTypeUtils::max<int>()) || allAxes) {
      // scalar
      NativeOpExecutioner::execReduceLongScalar(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    } else {
      // TAD
      std::vector<sd::LongType> dims(axis);

      for (int e = 0; e < dims.size(); e++)
        if (dims[e] < 0) dims[e] += x->rankOf();

      if (dims.size() > 1) std::sort(dims.begin(), dims.end());

      REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

      const sd::LongType* zShapeInfoH = z->shapeInfo();
      const sd::LongType* zShapeInfoD = z->specialShapeInfo();

      if (x->rankOf() - dims.size() != z->rankOf()) {
        auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
            z->shapeInfo(), dims, z->getContext()->getWorkspace());
        zShapeInfoH = reinterpret_cast<sd::LongType const*>(zPack->primary());
        zShapeInfoD = reinterpret_cast<sd::LongType const*>(zPack->special());
      }

      std::vector<sd::LongType> dims2 = ShapeUtils::evalDimsForReduceOp(x->rankOf(), dims);
      NativeOpExecutioner::execReduceLong(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), nullptr, z->buffer(), zShapeInfoH, z->specialBuffer(),
                                          zShapeInfoD, dims2.data(), dims2.size());

    }

    STORE_RESULT(*z);
  } else {
    auto indices = INPUT_VARIABLE(1);
    if (indices->lengthOf() == x->rankOf()) allAxes = true;


    std::vector<sd::LongType> dims(indices->lengthOf());
    for (int e = 0; e < indices->lengthOf(); e++) {
      // segfault on macOS if not like this
      int f = indices->e<int>(e);
      dims[e] = f >= 0 ? f : f += x->rankOf();
    }

    if ((block.getIArguments()->size() == 1 && INT_ARG(0) == sd::DataTypeUtils::max<int>()) || allAxes) {
      // scalar
      NativeOpExecutioner::execReduceLongScalar(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    } else {
      // TAD
      REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

      const sd::LongType* zShapeInfoH = z->shapeInfo();
      const sd::LongType* zShapeInfoD = z->specialShapeInfo();

      if (x->rankOf() - dims.size() != z->rankOf()) {
        auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
            z->shapeInfo(), dims, z->getContext()->getWorkspace());
        zShapeInfoH = reinterpret_cast<sd::LongType const*>(zPack->primary());
        zShapeInfoD = reinterpret_cast<sd::LongType const*>(zPack->special());
      }

      std::vector<sd::LongType> dims2 = ShapeUtils::evalDimsForReduceOp(x->rankOf(), dims);
      NativeOpExecutioner::execReduceLong(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), nullptr, z->buffer(), zShapeInfoH, z->specialBuffer(),
                                          zShapeInfoD, dims2.data(), dims2.size());

    }
  }

  manager.synchronize();


  if(OpRegistrator::getInstance().traceOps()) {
    std::vector<const sd::LongType *> *inputShapeBuffers = new std::vector<const sd::LongType *>();
    for(int i = 0; i < block.width(); i++) {
      inputShapeBuffers->push_back(block.variable(i)->getNDArray()->shapeInfo());
    }
    std::vector<const sd::LongType *> *outputShapeBuffers = new std::vector<const sd::LongType *>();
    for(int i = 0; i < block.outputWidth(); i++) {
      outputShapeBuffers->push_back(block.fastpath_out()[i]->shapeInfo());
    }

    OpExecTrace *opExecTrace = new OpExecTrace(inputShapeBuffers,outputShapeBuffers,this->getOpName());
    OpRegistrator::getInstance().registerOpExec(opExecTrace);
  }

  return sd::Status::OK;
}

/**
 *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
 *   It solely depends on input shape, and requested dimensions
 */
ShapeList* LegacyReduceLongOp::calculateOutputShape(ShapeList* inputShape, sd::graph::Context& block) {
  auto inShape = inputShape->at(0);

  sd::LongType* newShape;

  bool allAxes = false;

  auto keepDims = block.numB() > 0 ? B_ARG(0) : false;
  auto newFormat = block.numB() > 1 ? B_ARG(1) : true;

  auto axis = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<sd::LongType>() : *block.getAxis();

  if (axis.size() == shape::rank(inShape)) allAxes = true;

  // in this case we're building proper shape for reduction
  return SHAPELIST(ShapeUtils::evalReduceShapeInfo(shape::order(inShape), axis, inShape, DataType::INT64, keepDims,
                                                   !newFormat, block.workspace()));
}
}  // namespace ops
}  // namespace sd
