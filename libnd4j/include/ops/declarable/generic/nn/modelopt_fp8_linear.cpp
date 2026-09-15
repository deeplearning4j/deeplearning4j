/* SPDX-License-Identifier: Apache-2.0 */
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_modelopt_fp8_linear)
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/modelopt_linear.h>
#include <helpers/ConstantShapeHelper.h>
#include <cstdint>

namespace sd {
namespace ops {

// Distinct DataBuffer wrappers can still cover overlapping external storage.
static bool modelOptFp8StorageAliases(NDArray* a, NDArray* b) {
  auto* ab = a->dataBuffer();
  auto* bb = b->dataBuffer();
  if (a == b) return true;
  if (!ab || !bb) return false;
  if (ab == bb) return true;
  auto overlaps = [](void* p, size_t pn, void* q, size_t qn) {
    if (!p || !q || !pn || !qn) return false;
    const auto x = reinterpret_cast<uintptr_t>(p), y = reinterpret_cast<uintptr_t>(q);
    return x <= y ? y - x < pn : x - y < qn;
  };
  return overlaps(ab->primary(), ab->getLenInBytes(), bb->primary(), bb->getLenInBytes()) ||
         overlaps(ab->special(), ab->getLenInBytes(), bb->special(), bb->getLenInBytes());
}

CUSTOM_OP_IMPL(modelopt_fp8_linear, 4, 1, false, 0, 1) {
  REQUIRE_TRUE(block.width() == 4, 0, "modelopt_fp8_linear: exactly four inputs required");
  auto x = INPUT_VARIABLE(0);
  auto w = INPUT_VARIABLE(1);
  auto weightScale = INPUT_VARIABLE(2);
  auto inputScale = INPUT_VARIABLE(3);
  auto z = OUTPUT_VARIABLE(0);
  REQUIRE_TRUE(block.numI() == 1 && block.numT() == 0 && block.numB() == 0 && block.numD() == 0 &&
               (INT_ARG(0) == 0 || INT_ARG(0) == 1), 0,
               "modelopt_fp8_linear: floatOutput must be 0 or 1 (one IArg)");
  REQUIRE_TRUE(x->rankOf() >= 1 && w->rankOf() == 2, 0,
               "modelopt_fp8_linear: expected X[...,K] and W[N,K]");
  REQUIRE_TRUE(w->sizeAt(1) == x->sizeAt(-1), 0, "modelopt_fp8_linear: K dimensions differ");
  REQUIRE_TRUE(weightScale->rankOf() == 0 && inputScale->rankOf() == 0 &&
               weightScale->lengthOf() == 1 && inputScale->lengthOf() == 1, 0,
               "modelopt_fp8_linear: scales must be nonempty scalars");
  REQUIRE_TRUE((x->dataType() == FLOAT32 || x->dataType() == HALF || x->dataType() == BFLOAT16) &&
               w->dataType() == FLOAT8 && weightScale->dataType() == FLOAT32 && inputScale->dataType() == FLOAT32, 0,
               "modelopt_fp8_linear: invalid input dtypes");
  REQUIRE_TRUE(z->dataType() == (INT_ARG(0) ? FLOAT32 : x->dataType()) && z->rankOf() == x->rankOf() &&
               z->sizeAt(-1) == w->sizeAt(0), 0, "modelopt_fp8_linear: incorrect output shape or dtype");
  for (int d = 0; d < x->rankOf() - 1; ++d)
    REQUIRE_TRUE(z->sizeAt(d) == x->sizeAt(d), 0, "modelopt_fp8_linear: output leading dimensions differ");
  for (auto input : {x, w, weightScale, inputScale})
    REQUIRE_TRUE(!modelOptFp8StorageAliases(z, input), 0,
                 "modelopt_fp8_linear: output must not alias an input");
  helpers::modelOptLinear(block.launchContext(), x, w, weightScale, inputScale, z, false, INT_ARG(0) == 1);
  return Status::OK;
}

DECLARE_TYPES(modelopt_fp8_linear) {
  getOpDescriptor()->setAllowedInputTypes(0, {FLOAT32, HALF, BFLOAT16})
      ->setAllowedInputTypes(1, {FLOAT8})->setAllowedInputTypes(2, {FLOAT32})
      ->setAllowedInputTypes(3, {FLOAT32})->setAllowedOutputTypes({FLOAT32, HALF, BFLOAT16})
      ->setShapeValueInputs({})->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(modelopt_fp8_linear) {
  REQUIRE_TRUE(block.numI() == 1 && block.numT() == 0 && block.numB() == 0 && block.numD() == 0 &&
               (INT_ARG(0) == 0 || INT_ARG(0) == 1), 0,
               "modelopt_fp8_linear: floatOutput must be 0 or 1 (one IArg)");
  REQUIRE_TRUE(inputShape->size() == 4, 0, "modelopt_fp8_linear: exactly four inputs required");
  auto x = inputShape->at(0);
  auto w = inputShape->at(1);
  auto ws = inputShape->at(2);
  auto xs = inputShape->at(3);
  const auto xType = ArrayOptions::dataType(x);
  REQUIRE_TRUE((xType == FLOAT32 || xType == HALF || xType == BFLOAT16) &&
               ArrayOptions::dataType(w) == FLOAT8 && ArrayOptions::dataType(ws) == FLOAT32 &&
               ArrayOptions::dataType(xs) == FLOAT32, 0, "modelopt_fp8_linear: invalid input dtypes");
  const int rank = shape::rank(x);
  REQUIRE_TRUE(rank >= 1 && shape::rank(w) == 2 && shape::rank(ws) == 0 && shape::rank(xs) == 0 &&
               !shape::isEmptyConst(ws) && !shape::isEmptyConst(xs), 0,
               "modelopt_fp8_linear: incorrect input ranks");
  REQUIRE_TRUE(shape::sizeAt(w, 1) == shape::sizeAt(x, rank - 1), 0,
               "modelopt_fp8_linear: K dimensions differ");
  std::vector<LongType> dims(shape::shapeOf(x), shape::shapeOf(x) + rank);
  dims.back() = shape::sizeAt(w, 0);
  const auto dtype = INT_ARG(0) ? FLOAT32 : ArrayOptions::dataType(x);
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', dims));
}

}  // namespace ops
}  // namespace sd
#endif
