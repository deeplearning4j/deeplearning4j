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
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/LegacyReduceOp.h>
#include <ops/declarable/OpRegistrator.h>

#ifdef LEGACY_REDUCE_SAME_ONLY
namespace sd {
namespace ops {
LegacyReduceOp::LegacyReduceOp() : LegacyOp::LegacyOp(1) {
  //
}

LegacyReduceOp::LegacyReduceOp(int opType) : LegacyOp::LegacyOp(1, opType) {
}

LegacyOp *LegacyReduceOp::clone() { return new LegacyReduceOp(this->_opNum); }

sd::Status LegacyReduceOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);

  int opType = block.opType() < 0 ? this->_opNum : block.opType();
  sd_debug("Executing LegacyReduceOp: [%i]\n", opType);

  bool allAxes = false;

  if (block.width() == 1) {
    auto z = OUTPUT_VARIABLE(0);

    if (block.getIArguments()->size() == x->rankOf()) allAxes = true;

    if ((block.getIArguments()->size() == 0) || (block.getIArguments()->size() == 1 && INT_ARG(0) == SD_MAX_INT) ||
        allAxes) {
      // scalar
      NativeOpExcutioner::execReduceFloatScalar(opType, x->buffer(), x->shapeInfo(), block.getTArguments()->data(),
                                                z->buffer(), z->shapeInfo());
    } else {
      // TAD
      std::vector<int> dims(*block.getIArguments());

      for (int e = 0; e < dims.size(); e++)
        if (dims[e] < 0) dims[e] += x->rankOf();

      std::sort(dims.begin(), dims.end());

      REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

      shape::TAD tad(x->shapeInfo(), dims.data(), dims.size());
      tad.createTadOnlyShapeInfo();
      tad.createOffsets();

      NativeOpExcutioner::execReduceFloat(opType, x->buffer(), x->shapeInfo(), block.getTArguments()->data(),
                                          z->buffer(), z->shapeInfo(), dims.data(), (int)dims.size(),
                                          tad.tadOnlyShapeInfo, tad.tadOffsets);
    }

    STORE_RESULT(*z);
  } else {
    auto indices = INPUT_VARIABLE(1);
    if (indices->lengthOf() == x->rankOf()) allAxes = true;

    std::vector<int> axis(indices->lengthOf());
    for (int e = 0; e < indices->lengthOf(); e++) {
      // lol otherwise we segfault on macOS
      int f = indices->e<int>(e);
      axis[e] = f >= 0 ? f : f += x->rankOf();
    }

    if ((block.getIArguments()->size() == 1 && INT_ARG(0) == SD_MAX_INT) || allAxes) {
      auto z = OUTPUT_VARIABLE(0);

      auto b = x->buffer();
      auto s = x->shapeInfo();
      auto e = block.numT() > 0 ? block.getTArguments()->data() : nullptr;


      // scalar
      NativeOpExcutioner::execReduceFloatScalar(opType, b, s, e, z->buffer(), z->shapeInfo());
    } else {
      // TAD
      if (indices->lengthOf() > 1) std::sort(axis.begin(), axis.end());

      REQUIRE_TRUE(axis.size() > 0, 0, "Some dimensions required for reduction!");

      shape::TAD tad(x->shapeInfo(), axis.data(), axis.size());
      tad.createTadOnlyShapeInfo();
      tad.createOffsets();

      auto newShape = ShapeUtils::evalReduceShapeInfo(x->ordering(), axis, *x);
      auto z = new NDArray(newShape, x->getWorkspace());

      NativeOpExcutioner::execReduceFloat(opType, x->buffer(), x->shapeInfo(), block.getTArguments()->data(),
                                          z->buffer(), z->shapeInfo(), axis.data(), (int)axis.size(),
                                          tad.tadOnlyShapeInfo, tad.tadOffsets);

      // keepDims processing, for TF compatibility
      if (block.getIArguments()->size() > 0 && block.getIArguments()->at(0) == 1) {
        // z->printShapeInfo("z shape before");
        std::vector<sd::LongType> newshape(z->getShapeAsVector());
        for (int e = 0; e < axis.size(); e++) {
          auto a = axis.at(e);
          newshape.insert(newshape.begin() + a, 1);
        }
        z->reshapei(z->ordering(), newshape);
        // z->printShapeInfo("z shape after");
      }

      OVERWRITE_RESULT(z);
    }
  }

  if(OpRegistrator::getInstance().traceOps()) {
    std::vector<const sd::LongType *> *inputShapeBuffers = new std::vector<const sd::LongType *>();
    for(int i = 0; i < block.width(); i++) {
      inputShapeBuffers.push_back(block.variable(i)->getNDArray()->shapeInfo());
    }
    std::vector<const sd::LongType *> *outputShapeBuffers = new std::vector<const sd::LongType *>();
    for(int i = 0; i < block.outputWidth(); i++) {
      outputShapeBuffers.push_back(getZ(block,i)->shapeInfo());
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
ShapeList *LegacyReduceOp::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
  auto inShape = inputShape->at(0);

  sd::LongType *newShape;

  bool allAxes = false;

  if (block.getIArguments()->size() == shape::rank(inShape)) allAxes = true;

  if (block.getIArguments()->size() == 0 || (block.getIArguments()->size() == 1 && INT_ARG(0) == SD_MAX_INT) ||
      allAxes) {
    if (block.getIArguments()->size() > 0 && block.getIArguments()->at(0) == 1) {
      // in this case we just return legacy scalar
      ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), sd::LongType);
      newShape[0] = 2;
      newShape[1] = 1;
      newShape[2] = 1;
      newShape[3] = 1;
      newShape[4] = 1;
      newShape[5] = 0;
      newShape[6] = 1;
      newShape[7] = 99;
      // ArrayOptions::setDataType(newShape, block.dataType() ==
      // DataType::BOOL?block.dataType():ArrayOptions::dataType(inShape));
    } else {
      ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(0), sd::LongType);
      newShape[0] = 0;
      newShape[1] = 0;
      newShape[2] = 1;
      newShape[3] = 99;
      // ArrayOptions::setDataType(newShape, block.dataType() ==
      // DataType::BOOL?block.dataType():ArrayOptions::dataType(inShape));
    }
  } else {
    // in this case we're building proper shape for reduction
    auto array = new NDArray(nullptr, inShape, block.getWorkspace());

    newShape = ShapeUtils::evalReduceShapeInfo(shape::order(inShape), *block.getIArguments(), *array, false, false,
                                               block.workspace());

    delete array;
  }

  return SHAPELIST(newShape);
}
}  // namespace ops
}  // namespace sd
#endif
