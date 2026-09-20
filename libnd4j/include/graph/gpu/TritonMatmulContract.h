/* SPDX-License-Identifier: Apache-2.0 */
#ifndef LIBND4J_TRITON_MATMUL_CONTRACT_H
#define LIBND4J_TRITON_MATMUL_CONTRACT_H

#include <graph/DspAnalysisUtils.h>
#include <graph/gpu/TritonTargetDispatch.h>
#include <system/Environment.h>
#include <algorithm>
#include <limits>

namespace sd {
namespace graph {
namespace triton_matmul {

// Structural eligibility only. Concrete metadata is checked before cache lookup
// and again by every module entry point. No other matmul-family ABI is assumed.
inline bool supportsArguments(const NativeSlot& slot) {
  if (!dsp::hasNonLegacyMatmulArithmetic(slot)) return true;
  const auto& name = slot.ident.opName;
  return (name == "matmul" || name == "mmul" || name == "mMul") &&
      slot.args.numIArgs == 4 && slot.args.iArgs[3] == 1 &&
      slot.args.numTArgs >= 0 && slot.args.numTArgs <= 2 &&
      (slot.args.numTArgs == 0 || slot.args.tArgs != nullptr) &&
      slot.args.numBArgs == 0 && slot.args.numDArgs >= 0 && slot.args.numDArgs <= 1 &&
      (slot.args.numDArgs == 0 || slot.args.dArgs != nullptr) && slot.args.numSArgs == 0 &&
      !Environment::getInstance().isTritonExcludedOp(name) &&
      slot.wiring.numInputs == 2 && slot.wiring.numOutputs == 1 &&
      TritonTargetDispatch::detectTarget() == TritonGpuTarget::NVIDIA;
}

inline bool supportsArguments(NativeSlot* slots, int start, int end) {
  for (int i = start; i <= end; ++i)
    if (!supportsArguments(slots[i])) return false;
  return true;
}

inline NDArray* resolve(int src, NDArray** outputs, int nOutputs,
                        NDArray** inputs, int nInputs) {
  NDArray* a = src < 0 ? (inputs && -(src + 1) < nInputs ? inputs[-(src + 1)] : nullptr)
                       : (outputs && src < nOutputs ? outputs[src] : nullptr);
  return a && a->hasValidShapeInfo() ? a : nullptr;
}

// Initial compiled layout domain: nonempty rank>=2 matrices/equal-rank batches
// and ND/2D combinations, positive-stride inputs, dense C-order output. Reject
// vectors, aliasing, unavailable metadata and oversized indices. Explicit FLOAT
// output admits independent HALF/BFLOAT16/FLOAT input storage.
// Buffer pointers already include the view base; strides below are relative.
inline bool supports(const NativeSlot& slot, NDArray* a, NDArray* b, NDArray* c) {
  if (!supportsArguments(slot) || !a || !b || !c) return false;
  const auto dt = a->dataType();
  const auto outputType = slot.args.numDArgs > 0 ? slot.args.dArgs[0] : dt;
  if (c->dataType() != outputType) return false;
  auto floatStorage = [](DataType type) {
    return type == DataType::HALF || type == DataType::BFLOAT16 || type == DataType::FLOAT32;
  };
  if (slot.args.numDArgs > 0 && outputType == DataType::FLOAT32) {
    if (!floatStorage(dt) || !floatStorage(b->dataType())) return false;
  } else if (b->dataType() != dt || outputType != dt ||
             (!floatStorage(dt) && dt != DataType::DOUBLE)) return false;
  if (a->getDataBuffer() == c->getDataBuffer() || b->getDataBuffer() == c->getDataBuffer()) return false;
  const LongType limit = std::numeric_limits<int>::max() - 16384;
  for (auto* array : {a, b, c}) {
    if (array->rankOf() < 2 || array->isEmpty() || array->lengthOf() > limit) return false;
    LongType span = 0;
    for (int d = 0; d < array->rankOf(); ++d) {
      const auto size = array->sizeAt(d), stride = array->stridesOf()[d];
      if (size <= 0 || size > limit || stride <= 0 || stride > limit ||
          size - 1 > (limit - span) / stride) return false;
      span += (size - 1) * stride;
    }
  }
  LongType stride = 1;
  for (int d = c->rankOf() - 1; d >= 0; --d) {
    if (c->sizeAt(d) > 1 && c->stridesOf()[d] != stride) return false;
    stride *= c->sizeAt(d);
  }
  bool tx = slot.args.iArgs[0] != 0, ty = slot.args.iArgs[1] != 0;
  if (slot.args.iArgs[2]) {
    std::swap(a, b);
    const bool oldTx = tx;
    tx = !ty;
    ty = !oldTx;
  }
  const int ar = a->rankOf(), br = b->rankOf(), cr = c->rankOf();
  if (ar != br && ar != 2 && br != 2) return false;
  if (cr != std::max(ar, br) || a->sizeAt(ar - (tx ? 2 : 1)) != b->sizeAt(br - (ty ? 1 : 2)) ||
      c->sizeAt(cr - 2) != a->sizeAt(ar - (tx ? 1 : 2)) ||
      c->sizeAt(cr - 1) != b->sizeAt(br - (ty ? 2 : 1))) return false;
  for (int d = 0; d < cr - 2; ++d) {
    if ((ar > 2 && a->sizeAt(d) != c->sizeAt(d)) ||
        (br > 2 && b->sizeAt(d) != c->sizeAt(d))) return false;
  }
  return true;
}

inline bool supports(NativeSlot* slots, int start, int end,
                     NDArray** outputs, int nOutputs, NDArray** inputs, int nInputs) {
  for (int i = start; i <= end; ++i) {
    const auto& s = slots[i];
    if (!dsp::hasNonLegacyMatmulArithmetic(s)) continue;
    if (!supportsArguments(s)) return false;
    if (!supports(s, resolve(s.wiring.inputSourceIndices[0], outputs, nOutputs, inputs, nInputs),
                     resolve(s.wiring.inputSourceIndices[1], outputs, nOutputs, inputs, nInputs),
                     resolve(s.wiring.outputSlotIndices[0], outputs, nOutputs, inputs, nInputs))) return false;
  }
  return true;
}

}  // namespace triton_matmul
}  // namespace graph
}  // namespace sd
#endif
