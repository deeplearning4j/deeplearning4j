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
//  xw_plus_b op. Created by GS <george@skymind.io> 31.01.2018
//  @author Oleg Semeniv <oleg.semeniv@gmail.com>
//
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_xw_plus_b)

#include <helpers/MmulHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matmul.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(xw_plus_b, 3, 1, false, 0, 0) {
  bool aTranspose = (block.getIArguments()->size() > 0 ? INT_ARG(0) == 1 : false);
  bool bTranspose = (block.getIArguments()->size() > 1 ? INT_ARG(1) == 1 : false);
  bool cTranspose = (block.getIArguments()->size() > 2 ? INT_ARG(2) == 1 : false);

  auto x = aTranspose ? INPUT_VARIABLE(0)->transpose() : INPUT_VARIABLE(0);
  auto w = bTranspose ? INPUT_VARIABLE(1)->transpose() : INPUT_VARIABLE(1);
  auto b = INPUT_VARIABLE(2);
  auto z = cTranspose ? OUTPUT_VARIABLE(0)->transpose() : OUTPUT_VARIABLE(0);
  bool deleteBias = false;

  if (x->isEmpty() || w->isEmpty() || b->isEmpty()) return Status::OK;

  // Handle higher rank inputs by reshaping to 2D
  NDArray* xReshaped = nullptr;
  NDArray* zReshaped = nullptr;
  std::vector<sd::LongType> originalShape;
  
  if (x->rankOf() > 2) {
    // Save original shape for later
    auto* originalShapePtr = x->getShapeAsVector();
    originalShape = *originalShapePtr;
    delete originalShapePtr;
    
    // Calculate the 2D shape: flatten all but last dimension
    sd::LongType batchSize = 1;
    for (int i = 0; i < x->rankOf() - 1; i++) {
      batchSize *= x->sizeAt(i);
    }
    sd::LongType lastDim = x->sizeAt(x->rankOf() - 1);
    
    // Reshape x to 2D
    std::vector<sd::LongType> reshapeVec = {batchSize, lastDim};
    xReshaped = x->reshape('c', reshapeVec);
    x = xReshaped;
    
    // Calculate output shape
    sd::LongType outputLastDim = bTranspose ? w->sizeAt(0) : w->sizeAt(1);
    
    // Reshape z to 2D for computation
    std::vector<sd::LongType> zReshapeVec = {batchSize, outputLastDim};
    zReshaped = z->reshape('c', zReshapeVec);
    z = zReshaped;
  }

  REQUIRE_TRUE(x->rankOf() == 2, 0, "xw_plus_b: After reshaping, input x array should have rank equal 2, but got instead %i!",
               x->rankOf());
  REQUIRE_TRUE(w->rankOf() == 2, 0, "xw_plus_b: Input weights array should have rank equal 2, but got instead %i!",
               w->rankOf());
  REQUIRE_TRUE(z->rankOf() == 2, 0, "xw_plus_b: After reshaping, output array should have rank equal 2, but got instead %i!",
               z->rankOf());

  // multiply x to y
  MmulHelper::mmul(x, w, z, 1.0, 0.0);
  
  if(bTranspose && b->rankOf() == 1) {
    std::vector<sd::LongType> bShape = {b->lengthOf(), 1};
    b = b->reshape('c', bShape);
    deleteBias = true;
    if(z->isMatrix()) {
      z->addiColumnVector(b);
    } else {
      *z += *b;
    }
  } else {
    if(b->rankOf() == 1) {
      std::vector<sd::LongType> bShape = {1, b->lengthOf()};
      b = b->reshape('c', bShape);
      deleteBias = true;
    }

    if(z->isMatrix()) {
      // adding b vector
      z->addiRowVector(b);
    } else  {
      *z += *b;
    }
  }

  // If we reshaped, copy back to original output shape
  if (zReshaped != nullptr) {
    // Calculate final output shape
    std::vector<sd::LongType> outputShape = originalShape;
    outputShape[outputShape.size() - 1] = bTranspose ? w->sizeAt(0) : w->sizeAt(1);
    
    // Reshape z back to original dimensions
    auto zFinal = z->reshape('c', outputShape);
    OUTPUT_VARIABLE(0)->assign(zFinal);
    delete zFinal;
  }

  // Cleanup
  if (xReshaped != nullptr) {
    delete xReshaped;
  }
  if (zReshaped != nullptr) {
    delete zReshaped;
  }
  if(deleteBias) {
    delete b;
  }
  if (bTranspose) {
    delete w;
  }
  if (aTranspose && xReshaped == nullptr) {
    delete x;
  }
  if (cTranspose && zReshaped == nullptr) {
    delete z;
  }
  
  return Status::OK;
}

DECLARE_SHAPE_FN(xw_plus_b) {
  auto xShape = inputShape->at(0);
  auto weights = INPUT_VARIABLE(1);
  bool aTranspose = (block.getIArguments()->size() > 0 ? INT_ARG(0) == 1 : false);
  bool bTranspose = (block.getIArguments()->size() > 1 ? INT_ARG(1) == 1 : false);
  bool cTranspose = (block.getIArguments()->size() > 2 ? INT_ARG(2) == 1 : false);

  int nWeightsFormat = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

  auto weightsShape =
      (1 == nWeightsFormat) ? ShapeUtils::evalTransposeShapeInfo(*weights, block.getWorkspace()) : inputShape->at(1);

  // Handle higher rank inputs
  if (shape::rank(xShape) > 2) {
    // Calculate 2D shapes for matmul
    sd::LongType batchSize = 1;
    for (int i = 0; i < shape::rank(xShape) - 1; i++) {
      batchSize *= shape::sizeAt(xShape, i);
    }
    sd::LongType lastDim = shape::sizeAt(xShape, shape::rank(xShape) - 1);
    
    // Create temporary 2D shape for x
    std::vector<sd::LongType> x2dShape = {batchSize, lastDim};
    auto x2dShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(xShape), 
                                                                         'c', x2dShape);
    
    // Get the output shape from matmul
    auto matmulOutput = ShapeUtils::matrixProductShape(x2dShapeInfo, const_cast<sd::LongType *>(weightsShape), 
                                                       aTranspose, bTranspose,
                                                       ArrayOptions::dataType(xShape), block.getWorkspace());
    
    // Calculate final output shape
    std::vector<sd::LongType> outputShape;
    for (int i = 0; i < shape::rank(xShape) - 1; i++) {
      outputShape.push_back(shape::sizeAt(xShape, i));
    }
    // Add the output dimension from the weights
    outputShape.push_back(shape::sizeAt(matmulOutput, 1));
    
    auto finalShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(xShape), 
                                                                         'c', outputShape);
    return SHAPELIST(finalShape);
  } else {
    // Original behavior for rank 2 inputs
    auto outputShape = ShapeUtils::matrixProductShape(xShape, const_cast<sd::LongType *>(weightsShape), aTranspose,
                                                      bTranspose,
                                                      ArrayOptions::dataType(xShape), block.getWorkspace());
    return SHAPELIST(outputShape);
  }
}

DECLARE_TYPES(xw_plus_b) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(xw_plus_b_bp, 4, 3, false, 0, 0) {

  bool aTranspose = (block.getIArguments()->size() > 0 ? INT_ARG(0) == 1 : false);
  bool bTranspose = (block.getIArguments()->size() > 1 ? INT_ARG(1) == 1 : false);
  auto x = aTranspose ? new NDArray(INPUT_VARIABLE(0)->transpose()) : INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(2);
  auto dLdz = INPUT_VARIABLE(3);

  if (x->isEmpty() || INPUT_VARIABLE(1)->isEmpty() || b->isEmpty() || dLdz->isEmpty()) return Status::OK;

  auto w = bTranspose ? new NDArray(INPUT_VARIABLE(1)->transpose()) : INPUT_VARIABLE(1);

  REQUIRE_TRUE(x->rankOf() == 2, 0, "xw_plus_b BP: Input x array should have rank equal 2, but got instead %i!",
               x->rankOf());
  REQUIRE_TRUE(w->rankOf() == 2, 0, "xw_plus_b BP: Input weights array should have rank equal 2, but got instead %i!",
               w->rankOf());
  REQUIRE_TRUE(dLdz->rankOf() == 2, 0, "xw_plus_b BP: Output array should have rank equal 2, but got instead %i!",
               dLdz->rankOf());

  auto dLdx = aTranspose ? new NDArray(OUTPUT_VARIABLE(0)->transpose()) : OUTPUT_VARIABLE(0);
  auto dLdb = OUTPUT_VARIABLE(2);

  auto dLdw = (bTranspose) ? new NDArray(OUTPUT_VARIABLE(1)->transpose()) : OUTPUT_VARIABLE(1);

  // dLdb - reduceAlongDimension returns pointer
  std::vector<LongType> dims({0});
  auto* assign = dLdz->reduceAlongDimension(reduce::Sum, &dims);
  dLdb->assign(assign);
  delete assign;

  matmul_bp mmul_bp;
  mmul_bp.execute({x, w, dLdz}, std::vector<NDArray*>{dLdx, dLdw}, {}, {}, {});

  if(aTranspose) {
    delete x;
    delete dLdx;
  }

  if (bTranspose) {
    delete w;
    delete dLdw;
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(xw_plus_b_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)), CONSTANT(inputShape->at(2)));
}

DECLARE_TYPES(xw_plus_b_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}
}  // namespace ops
}  // namespace sd

#endif
