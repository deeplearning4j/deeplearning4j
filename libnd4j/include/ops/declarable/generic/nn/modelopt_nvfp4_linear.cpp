/* SPDX-License-Identifier: Apache-2.0 */
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_modelopt_nvfp4_linear)
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/modelopt_linear.h>
#include <helpers/ConstantShapeHelper.h>
#include <cstdint>

namespace sd {
namespace ops {

// Distinct DataBuffer wrappers can still cover overlapping external storage.
static bool modelOptNvfp4StorageAliases(NDArray* a, NDArray* b) {
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

CUSTOM_OP_IMPL(modelopt_nvfp4_linear, 4, 1, false, 0, 1) {
  REQUIRE_TRUE(block.width() == 4, 0, "modelopt_nvfp4_linear: exactly four inputs required");
  auto x = INPUT_VARIABLE(0);
  auto w = INPUT_VARIABLE(1);
  auto scales = INPUT_VARIABLE(2);
  auto global = INPUT_VARIABLE(3);
  auto z = OUTPUT_VARIABLE(0);
  REQUIRE_TRUE(block.numI() == 1 && block.numT() == 0 && block.numB() == 0 && block.numD() == 0 &&
               (INT_ARG(0) == 0 || INT_ARG(0) == 1), 0,
               "modelopt_nvfp4_linear: floatOutput must be 0 or 1 (one IArg)");
  REQUIRE_TRUE(x->rankOf() >= 1 && w->rankOf() == 2 && scales->rankOf() == 2, 0,
               "modelopt_nvfp4_linear: expected X[...,K], W[N,K/2], scales[N,K/16]");
  const LongType k = x->sizeAt(-1);
  const LongType n = w->sizeAt(0);
  REQUIRE_TRUE(k % 16 == 0 && w->sizeAt(1) == k / 2 &&
               scales->sizeAt(0) == n && scales->sizeAt(1) == k / 16, 0,
               "modelopt_nvfp4_linear: inconsistent packed shapes or K not divisible by 16");
  REQUIRE_TRUE(global->rankOf() == 0 && global->lengthOf() == 1, 0,
               "modelopt_nvfp4_linear: global_scale must be a nonempty scalar");
  REQUIRE_TRUE((x->dataType() == FLOAT32 || x->dataType() == HALF || x->dataType() == BFLOAT16) &&
               w->dataType() == UINT8 && scales->dataType() == FLOAT8 && global->dataType() == FLOAT32, 0,
               "modelopt_nvfp4_linear: invalid input dtypes");
  REQUIRE_TRUE(z->dataType() == (INT_ARG(0) ? FLOAT32 : x->dataType()) && z->rankOf() == x->rankOf() &&
               z->sizeAt(-1) == n, 0, "modelopt_nvfp4_linear: incorrect output shape or dtype");
  for (int d = 0; d < x->rankOf() - 1; ++d)
    REQUIRE_TRUE(z->sizeAt(d) == x->sizeAt(d), 0, "modelopt_nvfp4_linear: output leading dimensions differ");
  for (auto input : {x, w, scales, global})
    REQUIRE_TRUE(!modelOptNvfp4StorageAliases(z, input), 0,
                 "modelopt_nvfp4_linear: output must not alias an input");
  helpers::modelOptLinear(block.launchContext(), x, w, scales, global, z, true, INT_ARG(0) == 1);
  return Status::OK;
}

DECLARE_TYPES(modelopt_nvfp4_linear) {
  getOpDescriptor()->setAllowedInputTypes(0, {FLOAT32, HALF, BFLOAT16})
      ->setAllowedInputTypes(1, {UINT8})->setAllowedInputTypes(2, {FLOAT8})
      ->setAllowedInputTypes(3, {FLOAT32})->setAllowedOutputTypes({FLOAT32, HALF, BFLOAT16})
      ->setShapeValueInputs({})->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(modelopt_nvfp4_linear) {
  REQUIRE_TRUE(block.numI() == 1 && block.numT() == 0 && block.numB() == 0 && block.numD() == 0 &&
               (INT_ARG(0) == 0 || INT_ARG(0) == 1), 0,
               "modelopt_nvfp4_linear: floatOutput must be 0 or 1 (one IArg)");
  REQUIRE_TRUE(inputShape->size() == 4, 0, "modelopt_nvfp4_linear: exactly four inputs required");
  auto x = inputShape->at(0);
  auto w = inputShape->at(1);
  auto s = inputShape->at(2);
  auto g = inputShape->at(3);
  const auto xType = ArrayOptions::dataType(x);
  REQUIRE_TRUE((xType == FLOAT32 || xType == HALF || xType == BFLOAT16) &&
               ArrayOptions::dataType(w) == UINT8 && ArrayOptions::dataType(s) == FLOAT8 &&
               ArrayOptions::dataType(g) == FLOAT32, 0, "modelopt_nvfp4_linear: invalid input dtypes");
  const int rank = shape::rank(x);
  REQUIRE_TRUE(rank >= 1 && shape::rank(w) == 2 && shape::rank(s) == 2 && shape::rank(g) == 0 &&
               !shape::isEmptyConst(g), 0, "modelopt_nvfp4_linear: incorrect input ranks");
  const LongType k = shape::sizeAt(x, rank - 1);
  const LongType n = shape::sizeAt(w, 0);
  REQUIRE_TRUE(k >= 0 && n >= 0 && k % 16 == 0 && shape::sizeAt(w, 1) == k / 2 &&
               shape::sizeAt(s, 0) == n && shape::sizeAt(s, 1) == k / 16, 0,
               "modelopt_nvfp4_linear: inconsistent packed shapes or K not divisible by 16");
  std::vector<LongType> dims(shape::shapeOf(x), shape::shapeOf(x) + rank);
  dims.back() = n;
  const auto dtype = INT_ARG(0) ? FLOAT32 : ArrayOptions::dataType(x);
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', dims));
}

}  // namespace ops
}  // namespace sd
#endif
