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
#if NOT_EXCLUDED(OP_tensormmul)

#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

#include <numeric>

namespace sd {
namespace ops {

////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(tensormmul, 2, 1, false, 0, -1) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);

  auto c = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(a->dataType() == b->dataType(), 0, "tensormmul: A, B and C data types must be the same");

  // building axes
  LongType axe0_size = INT_ARG(0);
  LongType axe1_size = INT_ARG(axe0_size + 1);
  std::vector<LongType> axes_0(axe0_size), axes_1(axe1_size);
  for (LongType e = 0; e < axe0_size; e++) axes_0[e] = INT_ARG(e + 1);
  for (LongType e = 0; e < axe1_size; e++) axes_1[e] = INT_ARG(e + axe0_size + 2);


  std::vector<sd::LongType> permuteC = {};
  MmulHelper::tensorDot(a, b, c, axes_0, axes_1,permuteC);
  return Status::OK;
}
DECLARE_SYN(tensordot, tensormmul);

////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(tensormmul) {
  auto aShapeInfo = inputShape->at(0);
  auto bShapeInfo = inputShape->at(1);

  REQUIRE_TRUE(ArrayOptions::dataType(aShapeInfo) == ArrayOptions::dataType(bShapeInfo), 0,
               "tensormmul: A and B data types must be the same");

  // building axes
  LongType axe0_size = INT_ARG(0);
  LongType axe1_size = INT_ARG(axe0_size + 1);
  std::vector<LongType> axes_0(axe0_size), axes_1(axe1_size);
  for (LongType e = 0; e < axe0_size; e++) axes_0[e] = INT_ARG(e + 1);

  for (LongType e = 0; e < axe1_size; e++) axes_1[e] = INT_ARG(e + axe0_size + 2);

  sd_verbose("axe0: %i; axe1: %i;\n", axes_0.size(), axes_1.size());
  // evaluate shapes
  std::vector<LongType> permutAt, permutBt;
  std::vector<LongType> shapeAt, shapeBt;
  auto outShape =
      ShapeUtils::evalShapeForTensorDot(aShapeInfo, bShapeInfo, axes_0, axes_1, permutAt, permutBt,
                                                        shapeAt, shapeBt);

  auto desc = new  ShapeDescriptor(ArrayOptions::dataType(aShapeInfo), 'c', outShape);
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
      desc));
}

////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(tensormmul) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {FLOAT32, DOUBLE, HALF})
      ->setAllowedInputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->setAllowedInputTypes(2, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(0, {FLOAT32, DOUBLE, HALF});
}

// Comparator for sorting indices vector based on comparison of array values
struct IndexComparator
{
  const std::vector<LongType>& array;

  IndexComparator(const std::vector<LongType>& arr): array(arr) {}

  bool operator() (LongType i1, LongType i2)
  {
    return array[i1] < array[i2];
  }
};


std::vector<LongType> argsort(const std::vector<LongType>& array)
{
  std::vector<LongType> indices(array.size());
  for (LongType i = 0; i < array.size(); ++i) indices[i] = i;

  std::sort(indices.begin(), indices.end(), IndexComparator(array));

  return indices;
}

////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(tensormmul_bp, 4, 2, false, 0, -1) {
  auto A = INPUT_VARIABLE(0);
  auto B = INPUT_VARIABLE(1);
  auto C = INPUT_VARIABLE(2);
  auto dC = INPUT_VARIABLE(3);
  auto originalDC = dC;

  //scalar case, tile value to be whatever the c value is. common when directly attached to the loss
  if(dC->isScalar()) {
    auto newVec = const_cast<NDArray *>(C);
    auto newShape = newVec->getShapeAsVector();
    dC = new NDArray('c',newShape, dC->dataType(), dC->getContext());
  }


  auto gradA = OUTPUT_VARIABLE(0);
  auto gradB = OUTPUT_VARIABLE(1);

  LongType axe0_size = INT_ARG(0);
  LongType axe1_size = INT_ARG(axe0_size + 1);
  std::vector<LongType> axes0Sum(axe0_size), axes1Sum(axe1_size);

  //find the passed in axes for the feed forward
  for (LongType e = 0; e < axe0_size; e++) axes0Sum[e] = INT_ARG(e + 1);
  for (LongType e = 0; e < axe1_size; e++) axes1Sum[e] = INT_ARG(e + axe0_size + 2);


  auto Arank = A->rankOf();
  auto Brank = B->rankOf();
  auto dCrank = dC->rankOf();


  //part of the permtue axes before matrix multiply happens
  std::vector<LongType> axes_a_grad;
  for (LongType i = 0; i < Arank; ++i)
    axes_a_grad.push_back(i);

  for (LongType i = 0; i < axes0Sum.size(); ++i)
    axes_a_grad.erase(std::remove(axes_a_grad.begin(), axes_a_grad.end(), axes0Sum[i]), axes_a_grad.end());



  //part of matrix multiply axes before matrix multiply happens
  std::vector<LongType> axes_b_grad;
  for (LongType i = 0; i < Brank; ++i)
    axes_b_grad.push_back(i);

  for (LongType i = 0; i < axes1Sum.size(); ++i)
    axes_b_grad.erase(std::remove(axes_b_grad.begin(), axes_b_grad.end(), axes1Sum[i]), axes_b_grad.end());

  //used for post result permute to reshape result to be expected output
  std::vector<LongType> grad_a_axes;
  grad_a_axes.insert(grad_a_axes.end(), axes_a_grad.begin(), axes_a_grad.end());
  grad_a_axes.insert(grad_a_axes.end(), axes1Sum.begin(), axes1Sum.end());

  //used for post result permute to reshape result to be expected output
  std::vector<LongType> grad_b_axes;
  grad_b_axes.insert(grad_b_axes.end(), axes0Sum.begin(), axes0Sum.end());
  grad_b_axes.insert(grad_b_axes.end(), axes_b_grad.begin(), axes_b_grad.end());

  LongType starting = dCrank - axes_a_grad.size();
  std::vector<LongType> axes_a_gradA;
  for (LongType i = starting; i < dCrank; i++) {
    axes_a_gradA.push_back(i);
  }

  std::vector<LongType> axes_b_gradA;
  for (LongType i = 0; i < axes_b_grad.size(); i++) {
    axes_b_gradA.push_back(i);
  }

  std::vector<LongType> axes_a_gradB;
  for (LongType i = 0; i < axes_a_grad.size(); i++) {
    axes_a_gradB.push_back(i);
  }

  LongType start = dCrank - axes_a_gradA.size();
  std::vector<LongType> axes_b_gradB;
  for (LongType i = start; i < dCrank; i++) {
    axes_b_gradB.push_back(i);
  }

  //create final axes before for matrix multiply
  std::vector<LongType> aPermuteAxesBefore;
  aPermuteAxesBefore.insert(aPermuteAxesBefore.end(), axes_a_grad.begin(), axes_a_grad.end());
  aPermuteAxesBefore.insert(aPermuteAxesBefore.end(), axes0Sum.begin(), axes0Sum.end());



  //create final axes before for matrix multiply
  std::vector<LongType> bPermuteAxesBefore;
  bPermuteAxesBefore.insert(bPermuteAxesBefore.end(), axes_b_grad.begin(), axes_b_grad.end());
  bPermuteAxesBefore.insert(bPermuteAxesBefore.end(), axes1Sum.begin(), axes1Sum.end());

  auto aPermArgsAfter = argsort(grad_a_axes);
  auto bPermArgsAfter = argsort(grad_b_axes);
  auto newA = A->permute(aPermuteAxesBefore, false, false);
  std::vector<LongType> empty;
  auto newB = B->permute(bPermuteAxesBefore, false, false);


  //perform the actual matrix multiplication
  MmulHelper::tensorDot2(dC, &newB, gradA, axes_a_gradA, axes_b_gradA, empty, empty, aPermArgsAfter, gradA);
  MmulHelper::tensorDot2(&newA, dC, gradB, axes_a_gradB, axes_b_gradB, empty, empty, bPermArgsAfter, gradB);



  return Status::OK;
}

////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(tensormmul_bp) {
  auto aShapeInfo = inputShape->at(0);
  auto bShapeInfo = inputShape->at(1);
  auto cShapeInfo = inputShape->at(2);
  auto dLShapeInfo = inputShape->at(3);

  REQUIRE_TRUE((ArrayOptions::dataType(aShapeInfo) == ArrayOptions::dataType(bShapeInfo) &&
                (ArrayOptions::dataType(dLShapeInfo) == ArrayOptions::dataType(aShapeInfo))),
               0, "tensormmul_bp: A, B and dLdC data types must be the same");

  LongType* dLdAShapeInfo = nullptr;
  LongType* dLdBShapeInfo = nullptr;

  COPY_SHAPE(aShapeInfo, dLdAShapeInfo);
  COPY_SHAPE(bShapeInfo, dLdBShapeInfo);

  return SHAPELIST(CONSTANT(dLdAShapeInfo), CONSTANT(dLdBShapeInfo));
}

////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(tensormmul_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {FLOAT32, DOUBLE, HALF})  // maybe better ALL_FLOATS
      ->setAllowedInputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->setAllowedInputTypes(2, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(0, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(1, {FLOAT32, DOUBLE, HALF});
}
}  // namespace ops
}  // namespace sd

#endif
