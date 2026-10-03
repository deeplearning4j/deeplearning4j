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
#include <ops/declarable/headers/blas.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace sd {
namespace ops {

////////////////////////////////////////////////////////////////////////
// The boolean arguments transpose the first input, the second input and the result: a transposed array has its axes
// reversed (numpy's .T), and the contracted axes refer to the transposed inputs. A rank-0 or rank-1 array is its own
// transpose.

// The permutation that transposes an array of the given rank: its axes in reverse order.
static std::vector<LongType> tmmulReversedAxes(const LongType rank) {
  std::vector<LongType> axes(rank);
  for (LongType i = 0; i < rank; ++i) axes[i] = rank - 1 - i;
  return axes;
}

static bool tmmulTransposeArgument(Context& block, const int index) {
  return block.numB() > index && block.getBArguments()->at(index);
}

// The array transposed as a view, or nullptr when it is not transposed (or is its own transpose).
static NDArray* tmmulTransposedView(NDArray* array, const bool transpose) {
  if (!transpose || array->rankOf() < 2) return nullptr;
  std::vector<LongType> reversed = tmmulReversedAxes(array->rankOf());
  return array->permute(reversed, false, false);
}

////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(tensormmul, 2, 1, false, 0, -1) {
  auto a = INPUT_VARIABLE(0);
  auto b = INPUT_VARIABLE(1);

  auto c = OUTPUT_VARIABLE(0);

  // Auto-cast to matching dtype if inputs differ (same as matmul behavior).
  // MmulHelper::tensorDot -> mmulMxM handles mixed types via pickPairwiseResultType,
  // but the output buffer 'c' is already allocated with A's dtype from DECLARE_SHAPE_FN.
  // Cast the lower-precision input up so the computation uses the output dtype.
  NDArray* aCast = nullptr;
  NDArray* bCast = nullptr;
  if (a->dataType() != b->dataType()) {
    auto higherType = DataTypeUtils::pickPairwiseResultType(a->dataType(), b->dataType());
    if (a->dataType() != higherType) {
      aCast = a->cast(higherType);
      a = aCast;
    }
    if (b->dataType() != higherType) {
      bCast = b->cast(higherType);
      b = bCast;
    }
  }

  // building axes
  LongType axe0_size = INT_ARG(0);
  LongType axe1_size = INT_ARG(axe0_size + 1);
  std::vector<LongType> axes_0(axe0_size), axes_1(axe1_size);
  for (LongType e = 0; e < axe0_size; e++) axes_0[e] = INT_ARG(e + 1);
  for (LongType e = 0; e < axe1_size; e++) axes_1[e] = INT_ARG(e + axe0_size + 2);


  NDArray* aTransposed = tmmulTransposedView(a, tmmulTransposeArgument(block, 0));
  NDArray* bTransposed = tmmulTransposedView(b, tmmulTransposeArgument(block, 1));
  // A transposed result: the product is written through c's transposed view.
  std::vector<sd::LongType> permuteC = {};
  if (tmmulTransposeArgument(block, 2) && c->rankOf() > 1) permuteC = tmmulReversedAxes(c->rankOf());
  MmulHelper::tensorDot(aTransposed != nullptr ? aTransposed : a, bTransposed != nullptr ? bTransposed : b, c, axes_0,
                        axes_1, permuteC);

  delete aTransposed;
  delete bTransposed;
  delete aCast;
  delete bCast;

  return Status::OK;
}
DECLARE_SYN(tensordot, tensormmul);

////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(tensormmul) {
  auto aShapeInfo = inputShape->at(0);
  auto bShapeInfo = inputShape->at(1);

  // building axes
  LongType axe0_size = INT_ARG(0);
  LongType axe1_size = INT_ARG(axe0_size + 1);
  std::vector<LongType> axes_0(axe0_size), axes_1(axe1_size);
  for (LongType e = 0; e < axe0_size; e++) axes_0[e] = INT_ARG(e + 1);

  for (LongType e = 0; e < axe1_size; e++) axes_1[e] = INT_ARG(e + axe0_size + 2);

  sd_verbose("axe0: %i; axe1: %i;\n", axes_0.size(), axes_1.size());
  // A transposed input contracts over its reversed axes: evaluate the product on the transposed shapes.
  auto transposedShape = [](LongType* shapeInfo, const bool transpose) -> LongType* {
    const LongType rank = shape::rank(shapeInfo);
    if (!transpose || rank < 2) return shapeInfo;
    std::vector<LongType> reversed(rank);
    for (LongType i = 0; i < rank; ++i) reversed[i] = shape::shapeOf(shapeInfo)[rank - 1 - i];
    return ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(shapeInfo), 'c', reversed);
  };
  LongType* aEffective = transposedShape(aShapeInfo, tmmulTransposeArgument(block, 0));
  LongType* bEffective = transposedShape(bShapeInfo, tmmulTransposeArgument(block, 1));

  // evaluate shapes
  std::vector<LongType> permutAt, permutBt;
  std::vector<LongType> shapeAt, shapeBt;
  auto outShape =
      ShapeUtils::evalShapeForTensorDot(aEffective, bEffective, axes_0, axes_1, permutAt, permutBt,
                                                        shapeAt, shapeBt);
  if (tmmulTransposeArgument(block, 2)) std::reverse(outShape.begin(), outShape.end());

  auto outType = DataTypeUtils::pickPairwiseResultType(ArrayOptions::dataType(aShapeInfo),
                                                       ArrayOptions::dataType(bShapeInfo));
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(outType, 'c', outShape));
}

////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(tensormmul) {

  getOpDescriptor()
      ->setAllowedInputTypes(0, {FLOAT32, DOUBLE, HALF})
      ->setAllowedInputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->setAllowedInputTypes(2, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(0, {FLOAT32, DOUBLE, HALF})
      ->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);

}

////////////////////////////////////////////////////////////////////////
// Helpers of tensormmul_bp. The names are prefixed because translation units may be unity-built.

// The axes of an array of the given rank that are not contracted, ascending.
static std::vector<LongType> tmmulBpUncontractedAxes(const LongType rank, const std::vector<LongType>& contracted) {
  std::vector<LongType> axes;
  for (LongType axis = 0; axis < rank; ++axis)
    if (std::find(contracted.begin(), contracted.end(), axis) == contracted.end()) axes.push_back(axis);
  return axes;
}

// The shape of the array along the given axes, in the order given.
static std::vector<LongType> tmmulBpShapeAlong(NDArray* array, const std::vector<LongType>& axes) {
  std::vector<LongType> shape(axes.size());
  for (size_t i = 0; i < axes.size(); ++i) shape[i] = array->sizeAt(static_cast<int>(axes[i]));
  return shape;
}

static LongType tmmulBpElementCount(const std::vector<LongType>& shape) {
  LongType count = 1;
  for (const auto dim : shape) count *= dim;
  return count;
}

static std::vector<LongType> tmmulBpConcatenated(const std::vector<LongType>& first,
                                                 const std::vector<LongType>& second) {
  std::vector<LongType> result(first);
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

// The permutation that undoes `order`: inverse[order[i]] = i.
static std::vector<LongType> tmmulBpInversePermutation(const std::vector<LongType>& order) {
  std::vector<LongType> inverse(order.size());
  for (size_t i = 0; i < order.size(); ++i) inverse[order[i]] = static_cast<LongType>(i);
  return inverse;
}

// Drops an array derived from `source` by reshape: a wrapper over the source's DataBuffer is only a view, anything
// else owns a buffer of its own that CUDA kernels may still be reading.
static void tmmulBpRelease(NDArray* derived, NDArray* source) {
  if (derived == source) return;
  if (derived->getDataBuffer() != source->getDataBuffer())
    MmulHelper::deleteTemporary(derived);
  else
    delete derived;
}

// An owned, C-contiguous [rows, cols] copy of the array whose axes are reordered as `order` (the leading axes
// of `order` make up the rows). Row and column index run over their axes in C order.
static NDArray* tmmulBpFold(NDArray* array, std::vector<LongType>& order, const LongType rows, const LongType cols,
                            LaunchContext* context) {
  std::vector<LongType> matrixShape = {rows, cols};
  if (array->rankOf() == 0) {
    NDArray* single = new NDArray('c', matrixShape, array->dataType(), context);
    single->assign(array);
    return single;
  }

  NDArray* permuted = array->permute(order, false, false);
  NDArray* matrix = permuted->dup('c');
  delete permuted;
  matrix->reshapei('c', matrixShape);
  return matrix;
}

// The inverse of tmmulBpFold: writes the [rows, cols] matrix, whose rows and columns run over the target's axes
// listed in `order` (C order), into the target in the target's own axis order.
static void tmmulBpUnfold(NDArray* matrix, std::vector<LongType>& order, NDArray* target) {
  if (target->rankOf() == 0) {
    target->assign(matrix);
    return;
  }

  std::vector<LongType> orderedShape = tmmulBpShapeAlong(target, order);
  matrix->reshapei('c', orderedShape);
  std::vector<LongType> inverse = tmmulBpInversePermutation(order);
  NDArray* inTargetOrder = matrix->permute(inverse, false, false);
  target->assign(inTargetOrder);
  delete inTargetOrder;
}

////////////////////////////////////////////////////////////////////////
// Gradients of C = tensordot(A, B, axesA, axesB) given dC, the gradient at C.
//
// C's axes are A's uncontracted axes (freeA) followed by B's (freeB), and
//   C[a_free, b_free] = sum over k of A[a_free, k] * B[k, b_free]
// where k runs over the contracted axes of A in the order of axesA and of B in the order of axesB. Folding every
// operand into a matrix turns both gradients into one matrix product each:
//   dC  -> [FA, FB]   (FA, FB: the element counts of freeA and freeB)
//   B   -> [FB, K]    (B's axes reordered freeB, axesB; K: the element count of the contracted axes)
//   A^T -> [K, FA]    (A's axes reordered axesA, freeA)
//   dA = dC * B   : [FA, K], its axes are freeA then axesA of A
//   dB = A^T * dC : [K, FB], its axes are axesB then freeB of B
// Each product is unfolded back into the shape of its input by the inverse of that axis order.
//
// dC is a scalar when the output feeds the loss directly: then every element of C has that same gradient.
CUSTOM_OP_IMPL(tensormmul_bp, 4, 2, false, 0, -1) {
  auto A = INPUT_VARIABLE(0);
  auto B = INPUT_VARIABLE(1);
  // INPUT_VARIABLE(2) is C itself; its shape follows from A, B and the axes, so it is not read.
  auto dC = INPUT_VARIABLE(3);

  auto gradA = OUTPUT_VARIABLE(0);
  auto gradB = OUTPUT_VARIABLE(1);

  // With transposes the forward op multiplied the transposed inputs and transposed the product: differentiate that
  // product, reading the inputs and writing their gradients through transposed views, with the gradient at the
  // product the transposed gradient at the output.
  const bool transposeA = tmmulTransposeArgument(block, 0);
  const bool transposeB = tmmulTransposeArgument(block, 1);
  NDArray* aTransposed = tmmulTransposedView(A, transposeA);
  NDArray* bTransposed = tmmulTransposedView(B, transposeB);
  NDArray* gradATransposed = tmmulTransposedView(gradA, transposeA);
  NDArray* gradBTransposed = tmmulTransposedView(gradB, transposeB);
  NDArray* dCTransposed = dC->lengthOf() > 1 ? tmmulTransposedView(dC, tmmulTransposeArgument(block, 2)) : nullptr;
  auto releaseTransposedViews = [&]() {
    delete aTransposed;
    delete bTransposed;
    delete gradATransposed;
    delete gradBTransposed;
    delete dCTransposed;
  };
  if (aTransposed != nullptr) A = aTransposed;
  if (bTransposed != nullptr) B = bTransposed;
  if (gradATransposed != nullptr) gradA = gradATransposed;
  if (gradBTransposed != nullptr) gradB = gradBTransposed;
  if (dCTransposed != nullptr) dC = dCTransposed;

  const LongType aRank = A->rankOf();
  const LongType bRank = B->rankOf();
  const LongType numArgs = static_cast<LongType>(block.numI());

  // The integer arguments are [numAxesA, axesA..., numAxesB, axesB...].
  REQUIRE_TRUE(numArgs >= 2, 0, "TENSORMMUL_BP: expected the contracted axes of both inputs as integer arguments");
  const LongType numAxesA = INT_ARG(0);
  REQUIRE_TRUE(numAxesA >= 0 && numAxesA + 2 <= numArgs, 0,
               "TENSORMMUL_BP: the integer arguments do not hold %lld axes of the first input",
               static_cast<long long>(numAxesA));
  const LongType numAxesB = INT_ARG(numAxesA + 1);
  REQUIRE_TRUE(numAxesB == numAxesA && numAxesA + numAxesB + 2 <= numArgs, 0,
               "TENSORMMUL_BP: both inputs need the same number of contracted axes, got %lld and %lld",
               static_cast<long long>(numAxesA), static_cast<long long>(numAxesB));

  std::vector<LongType> axesA(numAxesA), axesB(numAxesB);
  for (LongType e = 0; e < numAxesA; e++) {
    axesA[e] = INT_ARG(e + 1);
    axesB[e] = INT_ARG(e + numAxesA + 2);
    if (axesA[e] < 0) axesA[e] += aRank;
    if (axesB[e] < 0) axesB[e] += bRank;
    REQUIRE_TRUE(axesA[e] >= 0 && axesA[e] < aRank && axesB[e] >= 0 && axesB[e] < bRank, 0,
                 "TENSORMMUL_BP: contracted axes %lld and %lld are out of range for ranks %lld and %lld",
                 static_cast<long long>(axesA[e]), static_cast<long long>(axesB[e]), static_cast<long long>(aRank),
                 static_cast<long long>(bRank));
    REQUIRE_TRUE(A->sizeAt(static_cast<int>(axesA[e])) == B->sizeAt(static_cast<int>(axesB[e])), 0,
                 "TENSORMMUL_BP: contracted axes %lld and %lld have different sizes %lld and %lld",
                 static_cast<long long>(axesA[e]), static_cast<long long>(axesB[e]),
                 static_cast<long long>(A->sizeAt(static_cast<int>(axesA[e]))),
                 static_cast<long long>(B->sizeAt(static_cast<int>(axesB[e]))));
    for (LongType p = 0; p < e; p++) {
      REQUIRE_TRUE(axesA[p] != axesA[e] && axesB[p] != axesB[e], 0, "TENSORMMUL_BP: contracted axes must be unique");
    }
  }

  const std::vector<LongType> freeA = tmmulBpUncontractedAxes(aRank, axesA);
  const std::vector<LongType> freeB = tmmulBpUncontractedAxes(bRank, axesB);
  const LongType freeASize = tmmulBpElementCount(tmmulBpShapeAlong(A, freeA));
  const LongType freeBSize = tmmulBpElementCount(tmmulBpShapeAlong(B, freeB));
  const LongType contractedSize = tmmulBpElementCount(tmmulBpShapeAlong(A, axesA));
  const LongType cLength = freeASize * freeBSize;

  REQUIRE_TRUE(dC->lengthOf() == cLength || dC->lengthOf() == 1, 0,
               "TENSORMMUL_BP: the gradient at the output has %lld elements, expected %lld or a scalar",
               static_cast<long long>(dC->lengthOf()), static_cast<long long>(cLength));

  // With nothing to sum over or nothing to differentiate, the gradients are zero.
  if (A->isEmpty() || B->isEmpty() || dC->isEmpty() || cLength == 0 || contractedSize == 0) {
    if (!gradA->isEmpty()) gradA->nullify();
    if (!gradB->isEmpty()) gradB->nullify();
    releaseTransposedViews();
    return Status::OK;
  }

  LaunchContext* context = block.launchContext();

  // dC as a [FA, FB] matrix: a scalar gradient is spread over every element of C.
  std::vector<LongType> cMatrixShape = {freeASize, freeBSize};
  NDArray* dCMatrix = nullptr;
  if (dC->lengthOf() == 1) {
    dCMatrix = new NDArray('c', cMatrixShape, dC->dataType(), context);
    dCMatrix->assign(dC);
  } else {
    // A view that cannot be reshaped in place (a transposed one, say) is copied first.
    dCMatrix = dC->reshape('c', cMatrixShape, false);
    if (dCMatrix == nullptr) {
      dCMatrix = dC->dup('c');
      dCMatrix->reshapei('c', cMatrixShape);
    }
  }

  std::vector<LongType> bOrder = tmmulBpConcatenated(freeB, axesB);
  std::vector<LongType> aOrder = tmmulBpConcatenated(axesA, freeA);
  NDArray* bMatrix = tmmulBpFold(B, bOrder, freeBSize, contractedSize, context);
  NDArray* aMatrix = tmmulBpFold(A, aOrder, contractedSize, freeASize, context);

  std::vector<LongType> gradAShape = {freeASize, contractedSize};
  std::vector<LongType> gradBShape = {contractedSize, freeBSize};
  NDArray* gradAMatrix = new NDArray('c', gradAShape, gradA->dataType(), context);
  NDArray* gradBMatrix = new NDArray('c', gradBShape, gradB->dataType(), context);

  MmulHelper::mmul(dCMatrix, bMatrix, gradAMatrix, 1.0, 0.0);
  MmulHelper::mmul(aMatrix, dCMatrix, gradBMatrix, 1.0, 0.0);

  std::vector<LongType> gradAOrder = tmmulBpConcatenated(freeA, axesA);
  std::vector<LongType> gradBOrder = tmmulBpConcatenated(axesB, freeB);
  tmmulBpUnfold(gradAMatrix, gradAOrder, gradA);
  tmmulBpUnfold(gradBMatrix, gradBOrder, gradB);

  tmmulBpRelease(dCMatrix, dC);
  MmulHelper::deleteTemporary(bMatrix);
  MmulHelper::deleteTemporary(aMatrix);
  MmulHelper::deleteTemporary(gradAMatrix);
  MmulHelper::deleteTemporary(gradBMatrix);
  releaseTransposedViews();

  return Status::OK;
}

////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(tensormmul_bp) {
  auto aShapeInfo = inputShape->at(0);
  auto bShapeInfo = inputShape->at(1);
  auto cShapeInfo = inputShape->at(2);
  auto dLShapeInfo = inputShape->at(3);

  return SHAPELIST(CONSTANT(aShapeInfo), CONSTANT(bShapeInfo));
}

////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(tensormmul_bp) {

  getOpDescriptor()
      ->setAllowedInputTypes(0, {FLOAT32, DOUBLE, HALF})  // maybe better ALL_FLOATS
      ->setAllowedInputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->setAllowedInputTypes(2, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(0, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD | OP_TRAIT_EXTERNAL_WORKSPACE);
}
}  // namespace ops
}  // namespace sd

#endif
