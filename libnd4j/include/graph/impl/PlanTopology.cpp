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

#include <graph/PlanTopology.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>

#include <cstring>
#include <cassert>

namespace sd {
namespace graph {

// Static empty vector for out-of-range releaseAtStep queries
const std::vector<int> PlanTopology::EMPTY_RELEASE_;

PlanTopology::PlanTopology()
    : numSlots_(0),
      numExternalInputs_(0),
      numRequestedOutputs_(0),
      requestedOutputIndices_(nullptr),
      externalInputNames_(nullptr),
      externalInputIsVariable_(nullptr),
      numSegments_(0),
      segmentDefs_(nullptr),
      fingerprint_(0),
      mode_(GraphExecutionMode::GEM_AUTO) {}

PlanTopology::~PlanTopology() {
  delete[] requestedOutputIndices_;
  delete[] externalInputNames_;
  delete[] externalInputIsVariable_;
  delete[] segmentDefs_;
}

std::shared_ptr<const PlanTopology> PlanTopology::create(
    int numSlots,
    int numExternalInputs,
    int numRequestedOutputs,
    const int* requestedOutputIndices,
    const std::string* externalInputNames,
    const bool* externalInputIsVariable,
    int numSegments,
    const GraphSegmentDef* segmentDefs,
    const std::vector<std::vector<int>>& releaseSchedule,
    const std::vector<LoopRegion>& loopRegions,
    uint64_t fingerprint,
    GraphExecutionMode mode) {

  auto* topo = new PlanTopology();

  topo->numSlots_ = numSlots;
  topo->numExternalInputs_ = numExternalInputs;
  topo->numRequestedOutputs_ = numRequestedOutputs;
  topo->fingerprint_ = fingerprint;
  topo->mode_ = mode;

  // Copy requested output indices
  if (numRequestedOutputs > 0 && requestedOutputIndices != nullptr) {
    topo->requestedOutputIndices_ = new int[numRequestedOutputs];
    std::memcpy(topo->requestedOutputIndices_, requestedOutputIndices,
                sizeof(int) * numRequestedOutputs);
  }

  // Copy external input names
  if (numExternalInputs > 0 && externalInputNames != nullptr) {
    topo->externalInputNames_ = new std::string[numExternalInputs];
    for (int i = 0; i < numExternalInputs; i++) {
      topo->externalInputNames_[i] = externalInputNames[i];
    }
  }

  // Copy external input isVariable flags
  if (numExternalInputs > 0 && externalInputIsVariable != nullptr) {
    topo->externalInputIsVariable_ = new bool[numExternalInputs];
    std::memcpy(topo->externalInputIsVariable_, externalInputIsVariable,
                sizeof(bool) * numExternalInputs);
  }

  // Copy segment definitions
  if (numSegments > 0 && segmentDefs != nullptr) {
    topo->numSegments_ = numSegments;
    topo->segmentDefs_ = new GraphSegmentDef[numSegments];
    for (int i = 0; i < numSegments; i++) {
      // GraphSegmentDef is a POD-like struct — copy fields manually
      auto& dst = topo->segmentDefs_[i];
      const auto& src = segmentDefs[i];
      dst.startSlot = src.startSlot;
      dst.endSlot = src.endSlot;
      dst.isCapturable = src.isCapturable;
      dst.hasValueDepOps = src.hasValueDepOps;
      dst.shapeKeyState = src.shapeKeyState;
      dst.backendOverride = src.backendOverride;
      dst.selectedBackend = src.selectedBackend;
      dst.contract = src.contract;
    }
  }

  // Copy release schedule
  topo->releaseSchedule_ = releaseSchedule;

  // Copy loop regions
  topo->loopRegions_ = loopRegions;

  DSP_DIAG(LIFECYCLE, "PlanTopology created: %d slots, %d extInputs, %d outputs, %d segments, fingerprint=%llu",
           numSlots, numExternalInputs, numRequestedOutputs, numSegments,
           (unsigned long long)fingerprint);

  return std::shared_ptr<const PlanTopology>(topo);
}

int PlanTopology::segmentForSlot(int slotIdx) const {
  for (int i = 0; i < numSegments_; i++) {
    if (slotIdx >= segmentDefs_[i].startSlot && slotIdx <= segmentDefs_[i].endSlot) {
      return i;
    }
  }
  return -1;
}

const std::vector<int>& PlanTopology::releaseAtStep(int stepIdx) const {
  if (stepIdx < 0 || stepIdx >= static_cast<int>(releaseSchedule_.size())) {
    return EMPTY_RELEASE_;
  }
  return releaseSchedule_[stepIdx];
}

void PlanTopology::emitReport() const {
  DSP_DIAG(LIFECYCLE, "[TOPOLOGY] slots=%d extInputs=%d outputs=%d segments=%d loops=%d fingerprint=%llu mode=%d",
           numSlots_, numExternalInputs_, numRequestedOutputs_,
           numSegments_, numLoopRegions(),
           (unsigned long long)fingerprint_, static_cast<int>(mode_));

  for (int i = 0; i < numSegments_; i++) {
    const auto& seg = segmentDefs_[i];
    DSP_DIAG(LIFECYCLE, "  seg[%d] slots=[%d-%d] capturable=%s backend=%d valueDepOps=%s",
             i, seg.startSlot, seg.endSlot,
             seg.isCapturable ? "true" : "false",
             static_cast<int>(seg.selectedBackend),
             seg.hasValueDepOps ? "true" : "false");
  }
}

}  // namespace graph
}  // namespace sd
