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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_squeeze)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(squeeze, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> axis;

  if (block.numI() > 0)
    for (size_t e = 0; e < block.numI(); e++) {
      int _a = INT_ARG(e);
      if (_a < 0) _a += input->rankOf();

      axis.emplace_back(_a);
    }
  else if (block.width() > 1) {
    auto a = INPUT_VARIABLE(1);
    for (LongType e = 0; e < a->lengthOf(); e++) {
      int _a = a->e<LongType>(e);

      if (_a < 0) _a += input->rankOf();

      axis.emplace_back(_a);
    }
  }

  if (input->rankOf() == 0 || (input->rankOf() == 1 && input->lengthOf() == 1)) {
    output->assign(input);
    return Status::OK;
  }

  std::vector<LongType> shape;
  if (axis.size() == 0) {
    for (int d = 0; d < input->rankOf(); d++)
      if (input->sizeAt(d) > 1) shape.emplace_back(input->sizeAt(d));
  } else {
    for (int d = 0; d < input->rankOf(); d++) {
      if (input->sizeAt(d) == 1) {
        if (std::find(axis.begin(), axis.end(), d) == axis.end()) shape.emplace_back(input->sizeAt(d));
      } else
        shape.emplace_back(input->sizeAt(d));
    }
  }

  if (block.isInplace()) {
    output->reshapei(input->ordering(), shape);
  } else {
    if (input->ews() == 1 && output->ews() == 1 && input->ordering() == output->ordering()) {
      output->dataBuffer()->copyBufferFrom(*input->dataBuffer(),
                                           output->lengthOf() * DataTypeUtils::sizeOfElement(output->dataType()), 0,
                                           input->offset());
    } else {
      auto tmp = input->reshape(input->ordering(), shape);
      output->assign(&tmp);
    }
  }

  return Status::OK;
}

DECLARE_TYPES(squeeze) { getOpDescriptor()->setAllowedInputTypes(ANY)->setSameMode(true); }

DECLARE_SHAPE_FN(squeeze) {
  auto shapeList = SHAPELIST();

  auto in = inputShape->at(0);
  auto rank = shape::rank(in);
  auto length = shape::length(in);
  if (rank == 0 || (rank == 1 && length == 1)) {
    shapeList->push_back(ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(in)));
    return shapeList;
  }

  std::vector<LongType> axis;

  if (block.numI() > 0)
    for (size_t e = 0; e < block.numI(); e++) {
      int _a = INT_ARG(e);
      if (_a < 0) _a += rank;

      axis.emplace_back(_a);
    }
  else if (block.width() > 1) {
    auto a = INPUT_VARIABLE(1);
    for (LongType e = 0; e < a->lengthOf(); e++) {
      LongType _a = a->e<LongType>(e);

      if (_a < 0) _a += rank;

      axis.emplace_back(_a);
    }
  }

  auto order = shape::order(in);
  auto oldShape = shape::shapeOf(in);

  std::vector<LongType> shape;
  if (axis.size() == 0) {
    for (LongType d = 0; d < rank; d++)
      if (oldShape[d] > 1) shape.emplace_back(oldShape[d]);
  } else {
    for (int d = 0; d < rank; d++) {
      if (oldShape[d] == 1) {
        if (std::find(axis.begin(), axis.end(), d) == axis.end()) shape.emplace_back(oldShape[d]);
      } else
        shape.emplace_back(oldShape[d]);
    }
  }

  if (shape.size() == 0) {
    shapeList->push_back(ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(in)));
    return shapeList;
  }

  if(shape::isEmptyConst(in)) {
    if(shape::rank(in) < 1) {
      shapeList->push_back(ConstantShapeHelper::getInstance().emptyShapeInfo(ArrayOptions::dataType(in)));
      return shapeList;
    }


    shapeList->push_back(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(ArrayOptions::dataType(in),shape));
    return shapeList;
  } else {
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(in), order, shape);
    shapeList->push_back(newShape);
  }


  return shapeList;
}
}  // namespace ops
}  // namespace sd

#endif
