/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef LIBND4J_DSP_RUNTIME_DETAIL_DSP_RUNTIME_INTERNAL_H
#define LIBND4J_DSP_RUNTIME_DETAIL_DSP_RUNTIME_INTERNAL_H

#include <dsp/runtime/dsp_runtime_c.h>

#include <string>
#include <vector>

namespace sd {
class NDArray;

namespace graph {
class Context;
class NativeDynamicShapePlan;
}

namespace dsp {
namespace runtime {
namespace detail {

/**
 * Internal bridge used by higher-level SDX sessions.
 *
 * This deliberately exposes only the already-compiled plan, its persistent
 * graph context, and NDArray bindings needed by the canonical native decode
 * primitive. Public consumers continue to use the stable C ABI.
 */
void setModelError(sdx_model_t* model, const std::string& error);

sdx_status_t runOwnedArrays(
    sdx_context_t* context,
    const std::vector<NDArray*>& publicInputs);

NDArray* contextOutputArray(sdx_context_t* context, int32_t outputIndex);

graph::NativeDynamicShapePlan* contextPlan(sdx_context_t* context);

graph::Context* contextGraph(sdx_context_t* context);

int32_t contextPlanInputCount(const sdx_context_t* context);

int32_t contextOutputCount(const sdx_context_t* context);

int32_t contextPlanInputIndex(
    const sdx_context_t* context,
    const std::string& inputName);

NDArray* contextPlanInputArray(
    sdx_context_t* context,
    int32_t planInputIndex);

}  // namespace detail
}  // namespace runtime
}  // namespace dsp
}  // namespace sd

#endif  // LIBND4J_DSP_RUNTIME_DETAIL_DSP_RUNTIME_INTERNAL_H
