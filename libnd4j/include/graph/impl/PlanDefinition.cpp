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

#include <graph/PlanDefinition.h>
#include <cstring>

namespace sd {
namespace graph {

PlanDefinition::~PlanDefinition() {
  delete[] requestedOutputSlotIndices_;
}

// ── Builder ─────────────────────────────────────────────────────────────────

PlanDefinition::Builder& PlanDefinition::Builder::setRequestedOutputSlotIndices(
    const int* indices, int count) {
  numRequestedOutputs_ = count;
  delete[] requestedOutputSlotIndices_;
  if (count > 0 && indices != nullptr) {
    requestedOutputSlotIndices_ = new int[count];
    std::memcpy(requestedOutputSlotIndices_, indices, count * sizeof(int));
  } else {
    requestedOutputSlotIndices_ = nullptr;
  }
  return *this;
}

PlanDefinition* PlanDefinition::Builder::build() {
  auto* def = new PlanDefinition();
  def->numSlots_ = numSlots_;
  def->totalOutputSlots_ = totalOutputSlots_;
  def->numExternalInputs_ = numExternalInputs_;
  def->numRequestedOutputs_ = numRequestedOutputs_;
  def->requestedOutputSlotIndices_ = requestedOutputSlotIndices_;
  requestedOutputSlotIndices_ = nullptr;  // Transfer ownership
  def->externalInputNames_ = std::move(externalInputNames_);
  def->externalInputIsVariable_ = std::move(externalInputIsVariable_);
  def->hasControlFlow_ = hasControlFlow_;
  def->numLoopRegions_ = numLoopRegions_;
  def->backendPriority_ = std::move(backendPriority_);
  return def;
}

}  // namespace graph
}  // namespace sd
