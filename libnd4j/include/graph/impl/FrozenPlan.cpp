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

#include <graph/FrozenPlan.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>

namespace sd {
namespace graph {

FrozenPlan::FrozenPlan() : plan_(nullptr), buildPassCount_(0) {}

FrozenPlan::~FrozenPlan() {
  // Plan is owned by us (when not in a cache) or by the cache.
  // In the cache path, the cache deletes the FrozenPlan which deletes the plan.
  // In the standalone path (compileDynamicShapePlan), we own it.
  delete plan_;
  plan_ = nullptr;
}

FrozenPlan* FrozenPlan::create(const void* serializedBytes, LongType numBytes,
                               int graphExecutionMode) {
  if (serializedBytes == nullptr || numBytes <= 0) {
    return nullptr;
  }

  auto mode = static_cast<GraphExecutionMode>(graphExecutionMode);
  auto* nativePlan = NativeDynamicShapePlan::fromSerializedPlan(serializedBytes, numBytes, mode);
  if (nativePlan == nullptr) {
    return nullptr;
  }

  auto* frozen = new FrozenPlan();
  frozen->plan_ = nativePlan;
  frozen->buildPassCount_ = 0;

  DSP_DIAG(LIFECYCLE, "created: %d slots, %d inputs, %d outputs, mode=%d",
           nativePlan->getNumSlots(), nativePlan->getNumExternalInputs(),
           nativePlan->getNumRequestedOutputs(), graphExecutionMode);

  return frozen;
}

int FrozenPlan::execute(Context* context, void* stream) {
  if (plan_ == nullptr) return 1;
  if (context == nullptr) return 1;

  int numInputs = static_cast<int>(context->width());
  int numOutputs = static_cast<int>(context->outputWidth());

  // Validate counts
  if (numInputs != plan_->getNumExternalInputs()) return 2;
  if (numOutputs != plan_->getNumRequestedOutputs()) return 3;

  // Extract input/output NDArrays from context
  std::vector<NDArray*> inputPtrs(numInputs);
  for (int i = 0; i < numInputs; i++) {
    inputPtrs[i] = context->array(i);
    if (inputPtrs[i] == nullptr) return 4;
  }

  std::vector<NDArray*> outputPtrs(numOutputs);
  for (int i = 0; i < numOutputs; i++) {
    outputPtrs[i] = context->outputArray(i);
    // Output arrays may be null (plan allocates them)
  }

  // Delegate to NativeDynamicShapePlan::execute()
  auto status = plan_->execute(
      inputPtrs.data(), numInputs,
      outputPtrs.data(), numOutputs,
      stream);

  if (status == Status::OK) {
    // Track build pass for diagnostics
    buildPassCount_++;

    // Write outputs back to context
    for (int i = 0; i < numOutputs; i++) {
      if (outputPtrs[i] != nullptr && outputPtrs[i] != context->outputArray(i)) {
        context->setOutputArray(i, outputPtrs[i]);
      }
    }
    return 0;
  }

  return static_cast<int>(status);
}

int FrozenPlan::releaseGpuIntermediates() {
  if (plan_ == nullptr) return 0;
  return plan_->releaseGpuIntermediates();
}

bool FrozenPlan::isSealed() const {
  if (plan_ == nullptr) return false;
  return plan_->getPlanPhaseCode() >= 3;  // REPLAYING
}

void FrozenPlan::emitStateReport() const {
  if (plan_ == nullptr) return;

  auto& diag = DspDiagnostics::getInstance();
  if (!diag.isEnabled(DSP_DIAG_LIFECYCLE)) return;

  const auto& segments = plan_->getSegments();
  int numSegs = static_cast<int>(segments.size());
  int numSlots = plan_->getNumSlots();
  const char* phaseName = plan_->planLifecycle().displayName();
  bool sealed = isSealed();

  diag.recordEvent(DSP_DIAG_LIFECYCLE, -1, -1, -1, "FrozenPlan", 0,
                   "[DSP_PLAN_STATE] buildPassCount=%d sealed=%s phase=%s segments=%d slots=%d",
                   buildPassCount_, sealed ? "true" : "false", phaseName, numSegs, numSlots);

  // Per-segment state dump
  for (int i = 0; i < numSegs; i++) {
    const auto& seg = segments[i];
    const char* segPhaseName = seg.exec.segPhase.displayName();
    int segStart = seg.def.startSlot;
    int segEnd = seg.def.endSlot;

    diag.recordEvent(DSP_DIAG_LIFECYCLE, -1, i, -1, "FrozenPlan", 0,
                     "  seg[%d: %d-%d] phase=%s backend=%s",
                     i, segStart, segEnd, segPhaseName,
                     seg.exec.compiledByBackend.empty() ? "none" : seg.exec.compiledByBackend.c_str());
  }
}

void FrozenPlan::emitResourceReport() const {
  if (plan_ == nullptr) return;

  auto& diag = DspDiagnostics::getInstance();
  if (!diag.isEnabled(DSP_DIAG_MEMORY)) return;

  int numSlots = plan_->getNumSlots();
  int numExternalInputs = plan_->getNumExternalInputs();
  int numOutputs = plan_->getNumRequestedOutputs();

  diag.recordEvent(DSP_DIAG_MEMORY, -1, -1, -1, "FrozenPlan", 0,
                   "[DSP_RESOURCES] slots=%d extInputs=%d outputs=%d",
                   numSlots, numExternalInputs, numOutputs);
}

uint64_t FrozenPlan::identityFingerprint() const {
  if (plan_ == nullptr) return 0;
  return plan_->identityFingerprint();
}

int FrozenPlan::getPlanPhaseCode() const {
  if (plan_ == nullptr) return -1;
  return plan_->getPlanPhaseCode();
}

void FrozenPlan::setShapesFrozen(bool frozen) {
  if (plan_ == nullptr) return;
  plan_->setShapesFrozen(frozen);
}

void FrozenPlan::clearShapeCaches() {
  if (plan_ == nullptr) return;
  plan_->clearShapeCaches();
}

void FrozenPlan::clearAllShapeCachesForce() {
  if (plan_ == nullptr) return;
  plan_->clearAllShapeCachesForce();
}

unsigned long long FrozenPlan::getReplaySignatureHash(int segIdx) const {
  if (plan_ == nullptr) return 0;
  const auto& segments = plan_->getSegments();
  if (segIdx < 0 || segIdx >= static_cast<int>(segments.size())) return 0;
  return segments[segIdx].exec.replaySignatureHash;
}

int FrozenPlan::getReplayUnitCount(int segIdx) const {
  if (plan_ == nullptr) return 0;
  const auto& segments = plan_->getSegments();
  if (segIdx < 0 || segIdx >= static_cast<int>(segments.size())) return 0;
  return segments[segIdx].exec.replayUnitCount;
}

}  // namespace graph
}  // namespace sd
