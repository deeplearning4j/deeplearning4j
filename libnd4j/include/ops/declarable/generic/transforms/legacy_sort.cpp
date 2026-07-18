/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <system/op_boilerplate.h>

#include <array/NDArrayFactory.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/headers/legacy_sort.h>

namespace sd {
namespace ops {
namespace {

bool descending(const graph::Context& block) {
  return block.getBArguments()->size() > 0 && B_ARG(0);
}

std::vector<sd::LongType> dimensions(const graph::Context& block) {
  return *block.getIArguments();
}

// ALL sorts route through the backend-portable NativeOps free functions.
// On CPU these dispatch to SpecialMethods<X>/DoubleMethods<X,Y> (host
// quicksort); on CUDA they launch the bitonic/oes GPU kernels in
// legacy/cuda/NativeOps_sort.cu. Calling SpecialMethods<X> or
// DoubleMethods<X,Y> directly here would require host-side template
// instantiations that only exist on the CPU backend (specials/specials_double
// are host-only) and would host-sort stale buffers on CUDA even if linked.

void copyUnlessAliased(NDArray* input, NDArray* output) {
  if (input != output && input->buffer() != output->buffer()) output->assign(input);
}

void registerSortTraits(DeclarableOp* op) {
  op->getOpDescriptor()->setAllowedInputTypes({ALL_INTS, ALL_FLOATS});
  op->getOpDescriptor()->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
  op->getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT |
                                   OP_TRAIT_FULLY_WRITING |
                                   OP_TRAIT_DATA_DEPENDENT);
}

}  // namespace

CUSTOM_OP_IMPL(legacy_sort, 1, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  copyUnlessAliased(input, output);
  sort(nullptr, output, descending(block));
  return Status::OK;
}

DECLARE_TYPES(legacy_sort) { registerSortTraits(this); }

DECLARE_SHAPE_FN(legacy_sort) {
  return SHAPELIST(ConstantShapeHelper::getInstance().createFromExisting(
      inputShape->at(0)));
}

CUSTOM_OP_IMPL(legacy_sort_tad, 1, 1, true, 0, -1) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  auto dims = dimensions(block);
  REQUIRE_TRUE(!dims.empty(), 0, "legacy_sort_tad: dimensions must not be empty");
  copyUnlessAliased(input, output);
  sortTad(nullptr, output, dims.data(),
                              static_cast<sd::LongType>(dims.size()),
                              nullptr, nullptr, descending(block));
  return Status::OK;
}

DECLARE_TYPES(legacy_sort_tad) { registerSortTraits(this); }

DECLARE_SHAPE_FN(legacy_sort_tad) {
  return SHAPELIST(ConstantShapeHelper::getInstance().createFromExisting(
      inputShape->at(0)));
}

#define LEGACY_PAIR_SORT_IMPL(OP_NAME, FUNCTION)                               \
  CUSTOM_OP_IMPL(OP_NAME, 2, 2, true, 0, 0) {                                \
    auto keys = OUTPUT_VARIABLE(0);                                            \
    auto values = OUTPUT_VARIABLE(1);                                          \
    REQUIRE_TRUE(INPUT_VARIABLE(0)->lengthOf() == INPUT_VARIABLE(1)->lengthOf(),\
                 0, #OP_NAME ": keys and values must have the same size");    \
    copyUnlessAliased(INPUT_VARIABLE(0), keys);                                \
    copyUnlessAliased(INPUT_VARIABLE(1), values);                              \
    FUNCTION(nullptr, keys, values, descending(block));     \
    return Status::OK;                                                         \
  }                                                                            \
  DECLARE_TYPES(OP_NAME) { registerSortTraits(this); }                         \
  DECLARE_SHAPE_FN(OP_NAME) {                                                  \
    return SHAPELIST(                                                         \
        ConstantShapeHelper::getInstance().createFromExisting(inputShape->at(0)),\
        ConstantShapeHelper::getInstance().createFromExisting(inputShape->at(1)));\
  }

LEGACY_PAIR_SORT_IMPL(legacy_sort_by_key, sortByKey)
LEGACY_PAIR_SORT_IMPL(legacy_sort_by_value, sortByValue)

#define LEGACY_PAIR_TAD_SORT_IMPL(OP_NAME, FUNCTION)                           \
  CUSTOM_OP_IMPL(OP_NAME, 2, 2, true, 0, -1) {                                \
    auto keys = OUTPUT_VARIABLE(0);                                            \
    auto values = OUTPUT_VARIABLE(1);                                          \
    auto dims = dimensions(block);                                             \
    REQUIRE_TRUE(!dims.empty(), 0, #OP_NAME ": dimensions must not be empty");\
    REQUIRE_TRUE(INPUT_VARIABLE(0)->lengthOf() == INPUT_VARIABLE(1)->lengthOf(),\
                 0, #OP_NAME ": keys and values must have the same size");    \
    copyUnlessAliased(INPUT_VARIABLE(0), keys);                                \
    copyUnlessAliased(INPUT_VARIABLE(1), values);                              \
    NDArray* dimension = NDArrayFactory::create<sd::LongType>(                  \
        'c', {static_cast<sd::LongType>(dims.size())}, dims);                   \
    FUNCTION(nullptr, keys, values, dimension,              \
                                 descending(block));                            \
    delete dimension;                                                          \
    return Status::OK;                                                         \
  }                                                                            \
  DECLARE_TYPES(OP_NAME) { registerSortTraits(this); }                         \
  DECLARE_SHAPE_FN(OP_NAME) {                                                  \
    return SHAPELIST(                                                         \
        ConstantShapeHelper::getInstance().createFromExisting(inputShape->at(0)),\
        ConstantShapeHelper::getInstance().createFromExisting(inputShape->at(1)));\
  }

LEGACY_PAIR_TAD_SORT_IMPL(legacy_sort_tad_by_key, sortTadByKey)
LEGACY_PAIR_TAD_SORT_IMPL(legacy_sort_tad_by_value, sortTadByValue)

#undef LEGACY_PAIR_SORT_IMPL
#undef LEGACY_PAIR_TAD_SORT_IMPL

}  // namespace ops
}  // namespace sd
