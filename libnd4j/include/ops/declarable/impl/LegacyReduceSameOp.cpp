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

#include <ops/declarable/LegacyReduceSameOp.h>

#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
namespace ops {
LegacyReduceSameOp::LegacyReduceSameOp() : LegacyOp(1) {
  //
}

LegacyReduceSameOp::LegacyReduceSameOp(int opNum) : LegacyOp(1, opNum) {
}

LegacyOp* LegacyReduceSameOp::clone() { return new LegacyReduceSameOp(this->_opNum); }

Status LegacyReduceSameOp::validateAndExecute(Context& block) {
  auto x = INPUT_VARIABLE(0);

  auto z = OUTPUT_VARIABLE(0);

  NDArray::prepareSpecialUse({z}, {x});

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();
  sd_debug("Executing LegacyReduceSameOp: [%i]\n", opNum);

  auto axis = *block.getAxis();
  bool allAxes = false;

  ExtraArguments extras(*block.getTArguments());
  PointersManager manager(block.launchContext(), "LegacyReduceSameOp");

  if (block.width() == 1) {
    if (axis.size() == static_cast<size_t>(x->rankOf())) allAxes = true;

    if (axis.empty() || allAxes) {
      // scalar
      NativeOpExecutioner::execReduceSameScalar(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(z->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    } else {
      // TAD
      std::vector<LongType> dims(axis);

      for (size_t e = 0; e < dims.size(); e++)
        if (dims[e] < 0) dims[e] += x->rankOf();

      REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

      const LongType* zShapeInfoH = z->shapeInfo();
      const LongType* zShapeInfoD = z->specialShapeInfo();

      if (x->rankOf() - dims.size() != static_cast<size_t>(z->rankOf())) {
        auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
            z->shapeInfo(), &dims, z->getContext()->getWorkspace());
        zShapeInfoH = reinterpret_cast<LongType const*>(zPack->primary());
        zShapeInfoD = reinterpret_cast<LongType const*>(zPack->special());
      }

      std::vector<LongType> *dims2 = ShapeUtils::evalDimsForReduceOp(x->rankOf(), &dims);
      NativeOpExecutioner::execReduceSame(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), nullptr, z->buffer(), zShapeInfoH, z->specialBuffer(),
                                          zShapeInfoD, dims2->data(), dims2->size());

    }

    STORE_RESULT(*z);
  } else {
    auto indices = INPUT_VARIABLE(1);
    if (indices->lengthOf() == x->rankOf()) allAxes = true;


    std::vector<LongType> dims(indices->lengthOf());
    for (int e = 0; e < indices->lengthOf(); e++) {
      // segfault on macOS if not like this
      int f = indices->e<LongType>(e);
      dims[e] = f >= 0 ? f : f += x->rankOf();
    }

    if ((block.getIArguments()->size() == 1 && INT_ARG(0) == DataTypeUtils::max<int>()) || allAxes) {
      // scalar
      NativeOpExecutioner::execReduceSameScalar(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(z->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    } else {
      // TAD
      REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

      const LongType* zShapeInfoH = z->shapeInfo();
      const LongType* zShapeInfoD = z->specialShapeInfo();

      if (x->rankOf() - dims.size() != static_cast<size_t>(z->rankOf())) {
        auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
            z->shapeInfo(), &dims, z->getContext()->getWorkspace());
        zShapeInfoH = reinterpret_cast<LongType const*>(zPack->primary());
        zShapeInfoD = reinterpret_cast<LongType const*>(zPack->special());
      }

      std::vector<LongType> *dims2 = ShapeUtils::evalDimsForReduceOp(x->rankOf(), &dims);
      NativeOpExecutioner::execReduceSame(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), nullptr, z->buffer(), zShapeInfoH, z->specialBuffer(),
                                          zShapeInfoD, dims2->data(), dims2->size());

      delete dims2;

    }
  }

  manager.synchronize();
  if(OpRegistrator::getInstance().traceOps()) {
    std::vector<const LongType*> *inputShapeBuffers = new std::vector<const LongType*>();
    for(size_t i = 0; i < block.width(); i++) {
      inputShapeBuffers->push_back(block.variable(i)->getNDArray()->shapeInfo());
    }
    std::vector<const LongType*> *outputShapeBuffers = new std::vector<const LongType*>();
    for(size_t i = 0; i < block.outputWidth(); i++) {
      outputShapeBuffers->push_back(getZ(block,i)->shapeInfo());
    }

    OpExecTrace *opExecTrace = new OpExecTrace(inputShapeBuffers,outputShapeBuffers,this->getOpName());
    OpRegistrator::getInstance().registerOpExec(opExecTrace);
  }
  return Status::OK;
}

/**
 *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
 *   It solely depends on input shape, and requested dimensions
 */
ShapeList* LegacyReduceSameOp::calculateOutputShape(ShapeList* inputShape, Context& block) {
  auto inShape = inputShape->at(0);


  auto keepDims = block.numB() > 0 ? B_ARG(0) : false;
  auto newFormat = block.numB() > 1 ? B_ARG(1) : true;

  auto axis = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<LongType>() : *block.getAxis();


  // in this case we're building proper shape for reduction
  auto newShape =
      ShapeUtils::evalReduceShapeInfo(shape::order(inShape), &axis, inShape, keepDims, !newFormat, block.workspace());

  return SHAPELIST(newShape);
}
}  // namespace ops
}  // namespace sd
