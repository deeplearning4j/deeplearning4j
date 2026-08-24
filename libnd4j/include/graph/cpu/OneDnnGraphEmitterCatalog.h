/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_ONEDNN_GRAPH_EMITTER_CATALOG_H
#define LIBND4J_ONEDNN_GRAPH_EMITTER_CATALOG_H

#include <config.h>

#if HAVE_ONEDNN

#include <graph/NativeDynamicShapePlan.h>
#include <oneapi/dnnl/dnnl_graph.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace sd {
namespace graph {

namespace dg = dnnl::graph;

/**
 * Concrete arrays and frozen arguments used to validate one exact oneDNN
 * lowering. Capability is deliberately descriptor-based: op-local traits
 * describe intrinsic semantics, while this catalog describes the combinations
 * for which oneDNN has an exact implementation.
 */
struct OneDnnLoweringContext {
  const NativeSlot& slot;
  const std::vector<NDArray*>& inputs;
  const std::vector<NDArray*>& outputs;
};

/** A fully validated oneDNN operation and its framework operand order. */
struct OneDnnLoweredOp {
  dg::op operation;
  std::vector<int> frameworkInputOrder;

  OneDnnLoweredOp(size_t opId, dg::op::kind kind, const std::string& name)
      : operation(opId, kind, name) {}
};

using OneDnnLowerer = bool (*)(const OneDnnLoweringContext& context,
                               OneDnnLoweredOp& lowered,
                               std::string& rejectionReason);

struct OneDnnGraphEmitterInfo {
  LongType descriptorHash;
  uint64_t intrinsicTraits;
  dg::op::kind kind;
  bool anchor;
  OneDnnLowerer lower;
};

/** Find an exact emitter by the canonical descriptor hash carried by a slot. */
const OneDnnGraphEmitterInfo* findOneDnnGraphEmitter(const NativeSlot& slot);

/** Enumerate the catalog for validation tests. */
const std::vector<OneDnnGraphEmitterInfo>& getOneDnnGraphEmitterCatalog();

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ONEDNN
#endif  // LIBND4J_ONEDNN_GRAPH_EMITTER_CATALOG_H
