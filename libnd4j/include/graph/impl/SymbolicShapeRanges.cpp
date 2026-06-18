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

#include <graph/gpu/SymbolicShapeRanges.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspHashUtils.h>
#include <helpers/shape.h>

#include <vector>
#include <algorithm>

namespace sd {
namespace graph {

// Per-dimension observation record
struct DimRecord {
  LongType minSeen;
  LongType maxSeen;
  bool isStatic;  // true if dimension never changed across observations

  DimRecord() : minSeen(0), maxSeen(0), isStatic(true) {}
};

// Per-input shape profile: tracks rank and per-dimension stats
struct InputShapeRecord {
  int rank;
  DataType dtype;
  std::vector<DimRecord> dims;
  bool initialized;

  InputShapeRecord() : rank(0), dtype(DataType::FLOAT32), initialized(false) {}
};

// Full profile for a segment
struct SegmentShapeProfile {
  int warmupSteps;       // Total warmup steps to observe
  int observationCount;  // Current count of observations
  std::vector<InputShapeRecord> inputRecords;

  SegmentShapeProfile(int warmup) : warmupSteps(warmup), observationCount(0) {}
};

SegmentShapeProfile* createSegmentShapeProfile(int warmupSteps) {
  return new SegmentShapeProfile(std::max(1, warmupSteps));
}

void freeSegmentShapeProfile(SegmentShapeProfile* profile) {
  delete profile;
}

void recordObservedShapes(SegmentShapeProfile* profile,
                          NDArray** inputArrays, int numInputs) {
  if (profile == nullptr || inputArrays == nullptr) return;

  DSP_DIAG(SHAPE, "recordObservedShapes: obs=%d/%d numInputs=%d",
           profile->observationCount + 1, profile->warmupSteps, numInputs);

  // Resize input records if needed
  if (static_cast<int>(profile->inputRecords.size()) < numInputs) {
    profile->inputRecords.resize(numInputs);
  }

  for (int i = 0; i < numInputs; i++) {
    NDArray* arr = inputArrays[i];
    if (arr == nullptr) continue;

    auto& rec = profile->inputRecords[i];
    const LongType* si = arr->shapeInfo();
    LongType rank = shape::rank(si);
    DataType dt = arr->dataType();

    if (!rec.initialized) {
      // First observation: initialize
      rec.rank = static_cast<int>(rank);
      rec.dtype = dt;
      rec.dims.resize(rank);
      for (int d = 0; d < rank; d++) {
        rec.dims[d].minSeen = si[d + 1];
        rec.dims[d].maxSeen = si[d + 1];
        rec.dims[d].isStatic = true;
      }
      rec.initialized = true;
    } else {
      // Subsequent observations: update min/max, detect changes
      if (static_cast<int>(rank) != rec.rank) {
        // Rank changed — mark ALL dimensions as dynamic
        // (this is unusual but handle it gracefully)
        DSP_DIAG(SHAPE, "recordObservedShapes: input[%d] RANK CHANGED %d→%lld — all dims dynamic",
                 i, rec.rank, (long long)rank);
        rec.rank = static_cast<int>(rank);
        rec.dims.resize(rank);
        for (int d = 0; d < rank; d++) {
          rec.dims[d].isStatic = false;
          rec.dims[d].minSeen = si[d + 1];
          rec.dims[d].maxSeen = si[d + 1];
        }
      } else {
        for (int d = 0; d < rank; d++) {
          LongType dimVal = si[d + 1];
          if (dimVal != rec.dims[d].minSeen || dimVal != rec.dims[d].maxSeen) {
            rec.dims[d].isStatic = false;
          }
          rec.dims[d].minSeen = std::min(rec.dims[d].minSeen, dimVal);
          rec.dims[d].maxSeen = std::max(rec.dims[d].maxSeen, dimVal);
        }
      }
    }
  }

  profile->observationCount++;
}

bool isWarmupComplete(const SegmentShapeProfile* profile) {
  return profile != nullptr && profile->observationCount >= profile->warmupSteps;
}

int getObservationCount(const SegmentShapeProfile* profile) {
  return profile != nullptr ? profile->observationCount : 0;
}

int getWarmupSteps(const SegmentShapeProfile* profile) {
  return profile != nullptr ? profile->warmupSteps : 0;
}

LongType computeRangeBasedShapeKey(SegmentShapeProfile* profile,
                                    NDArray** inputArrays, int numInputs,
                                    int startSlot, int endSlot) {
  if (profile == nullptr) return 0;

  DSP_DIAG(SHAPE, "computeRangeBasedShapeKey: [%d-%d] numInputs=%d warmupComplete=%d",
           startSlot, endSlot, numInputs, isWarmupComplete(profile) ? 1 : 0);

  uint64_t key = dsp::FNV1A64_OFFSET_BASIS;
  auto mix = [&key](LongType val) {
    dsp::fnv1aMixValue(key, static_cast<uint64_t>(val));
  };

  mix(startSlot);
  mix(endSlot);

  for (int i = 0; i < numInputs; i++) {
    NDArray* arr = inputArrays[i];
    if (arr == nullptr) continue;

    const LongType* si = arr->shapeInfo();
    LongType rank = shape::rank(si);

    // Always hash rank and dtype
    mix(rank);
    mix(static_cast<LongType>(arr->dataType()));

    if (i < static_cast<int>(profile->inputRecords.size()) &&
        profile->inputRecords[i].initialized &&
        isWarmupComplete(profile)) {
      auto& rec = profile->inputRecords[i];
      for (int d = 0; d < rank && d < static_cast<int>(rec.dims.size()); d++) {
        if (rec.dims[d].isStatic) {
          // Static dimension: hash exact value (standard behavior)
          mix(si[d + 1]);
        }
        // Dynamic dimension: skip hashing the exact value.
        // rank + dtype already hashed above, so same kernel works for any value.
      }
    } else {
      // Not enough observations yet — hash everything (standard behavior)
      for (int d = 0; d < rank; d++) {
        mix(si[d + 1]);
      }
      mix(static_cast<LongType>(arr->lengthOf()));
    }
  }

  DSP_DIAG(SHAPE, "computeRangeBasedShapeKey: [%d-%d] key=0x%llx", startSlot, endSlot, (long long)key);

  return key;
}

}  // namespace graph
}  // namespace sd
