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
// Platform-agnostic NativeOps DSP entry points.
//
// These functions are trivial accessors on NativeDynamicShapePlan that have
// identical implementations on CPU and CUDA builds. Keeping them in
// legacy/impl/ means there is exactly one definition shared between both
// backends (BuildCPU.cmake and MainBuildFlow.cmake both glob
// ./include/legacy/impl/*.cpp into the source list).
//

#include <dsp/NativeOpsDsp.h>
#include <legacy/NativeOps.h>
#include <graph/NativeDynamicShapePlan.h>

using sd::graph::NativeDynamicShapePlan;


int isPlanCompilationSealed(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return plan->isCompilationSealed() ? 1 : 0;
}

long long getPlanMidExecutionCompileCount(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return static_cast<long long>(plan->getMidExecutionCompileCount());
}

void resetPlanMidExecutionCompileCount(sd::Pointer planHandle) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  plan->resetMidExecutionCompileCount();
}

