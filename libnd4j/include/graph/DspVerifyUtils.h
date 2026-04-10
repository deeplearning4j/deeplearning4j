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

#ifndef LIBND4J_DSP_VERIFY_UTILS_H
#define LIBND4J_DSP_VERIFY_UTILS_H

#include <array/DataTypeUtils.h>
#include <array/NDArray.h>
#include <graph/DspDiagnostics.h>
#include <system/common.h>

// Portable buffer accessor: specialBuffer() on CUDA, buffer() on CPU.
#ifndef DSP_BUF
#ifdef SD_CUDA
#define DSP_BUF(arr) ((arr)->specialBuffer())
#else
#define DSP_BUF(arr) ((arr)->buffer())
#endif
#endif
#include <types/types.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

namespace sd {
namespace graph {

#ifdef SD_CUDA

// ─── Templated D2H copy: read first N elements as float ──────────────────
template <typename T>
inline int dspVerifyCopyValuesT(void* devicePtr, LongType length, float* hostOut, int maxVals) {
  int n = std::min(static_cast<int>(length), maxVals);
  // std::vector<bool> is bit-packed (no .data()), so use raw buffer for all types
  std::unique_ptr<T[]> tmp(new T[n]);
  cudaMemcpy(tmp.get(), devicePtr, n * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) hostOut[i] = static_cast<float>(tmp[i]);
  return n;
}

// Dispatch D2H copy by runtime DataType.
inline int dspVerifyCopyValues(void* devicePtr, DataType dtype, LongType length,
                               float* hostOut, int maxVals) {
  if (devicePtr == nullptr || length <= 0) return 0;
  BUILD_SINGLE_SELECTOR(dtype, return dspVerifyCopyValuesT,
      (devicePtr, length, hostOut, maxVals), SD_COMMON_TYPES);
  return 0;
}

// ─── Templated raw-bytes-to-float: convert from host byte buffer ─────────
template <typename T>
inline void dspBytesToFloatT(const uint8_t* raw, float* out, int count) {
  for (int i = 0; i < count; i++) {
    T val;
    memcpy(&val, raw + i * sizeof(T), sizeof(T));
    out[i] = static_cast<float>(val);
  }
}

// Dispatch bytes-to-float by runtime DataType.
inline void dspBytesToFloat(const uint8_t* raw, DataType dtype, float* out, int count) {
  BUILD_SINGLE_SELECTOR(dtype, dspBytesToFloatT, (raw, out, count), SD_COMMON_TYPES);
}

// ─── Templated comparison: compute max absolute diff between two raw buffers ─
template <typename T>
inline float dspMaxDiffT(const uint8_t* bufA, const uint8_t* bufB, int count) {
  auto* a = reinterpret_cast<const T*>(bufA);
  auto* b = reinterpret_cast<const T*>(bufB);
  float maxDiff = 0.0f;
  for (int i = 0; i < count; i++) {
    float diff = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
    if (diff > maxDiff) maxDiff = diff;
  }
  return maxDiff;
}

// Dispatch max-diff comparison by runtime DataType.
inline float dspMaxDiff(const uint8_t* bufA, const uint8_t* bufB, DataType dtype, int count) {
  BUILD_SINGLE_SELECTOR(dtype, return dspMaxDiffT, (bufA, bufB, count), SD_COMMON_TYPES);
  return 0.0f;
}

// ─── Format float array to string ────────────────────────────────────────
inline std::string dspFormatValues(const float* vals, int count) {
  std::string s = "[";
  for (int i = 0; i < count; i++) {
    if (i > 0) s += ",";
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6g", vals[i]);
    s += buf;
  }
  s += "]";
  return s;
}

// ─── D2H copy + format as string (convenience) ──────────────────────────
// IMPORTANT: D2H copies are ILLEGAL during CUDA graph capture.
// Returns "[capture]" when called during capture to avoid invalidating the graph.
inline std::string dspDumpSlotValues(void* devicePtr, DataType dtype, LongType length, int maxVals = 4) {
  if (devicePtr == nullptr || length <= 0) return "[]";
  // Guard: D2H transfers during graph capture create illegal dependencies
  if (sd::tl_graphExecutionActive) return "[capture]";
  float vals[16];
  int n = dspVerifyCopyValues(devicePtr, dtype, length, vals, std::min(maxVals, 16));
  return dspFormatValues(vals, n);
}

// ─── Shape string for an NDArray ─────────────────────────────────────────
inline std::string dspShapeStr(NDArray* arr) {
  if (arr == nullptr) return "null";
  auto rank = arr->rankOf();
  std::string s = "[";
  for (int d = 0; d < rank; d++) {
    if (d > 0) s += ",";
    s += std::to_string(arr->sizeAt(d));
  }
  s += "]";
  return s;
}

// ─── Compute argmax from device buffer ───────────────────────────────────
// D2H copies the full buffer and finds the index of the maximum value.
// Returns -1 if the buffer is null/empty or during graph capture.
inline int dspArgmax(void* devicePtr, DataType dtype, LongType length) {
  if (devicePtr == nullptr || length <= 0) return -1;
  if (sd::tl_graphExecutionActive) return -1;
  int n = static_cast<int>(length);
  std::unique_ptr<float[]> vals(new float[n]);
  dspVerifyCopyValues(devicePtr, dtype, length, vals.get(), n);
  int bestIdx = 0;
  float bestVal = vals[0];
  for (int i = 1; i < n; i++) {
    if (vals[i] > bestVal) { bestVal = vals[i]; bestIdx = i; }
  }
  return bestIdx;
}

// ─── Read host buffer values as float (no CUDA, no D2H) ─────────────────
// Reads directly from the primary (host) buffer of a DataBuffer.
inline int dspReadHostValues(DataBuffer* db, DataType dtype, LongType length,
                             float* hostOut, int maxVals) {
  if (db == nullptr || db->primary() == nullptr || length <= 0) return 0;
  int n = std::min(static_cast<int>(length), maxVals);
  auto* raw = reinterpret_cast<const uint8_t*>(db->primary());
  dspBytesToFloat(raw, dtype, hostOut, n);
  return n;
}

// ─── Dump NDArray host + device buffer values to diagnostic string ────────
// Reads host buffer directly and device buffer via D2H copy.
// Returns a formatted string: "dtype=FLOAT32 len=5 pAct=1 sAct=1 host=[...] dev=[...]"
inline std::string dspDumpHostDeviceValues(NDArray* arr, int maxVals = 8) {
  if (arr == nullptr) return "null";
  auto* db = arr->dataBuffer();
  auto dtype = arr->dataType();
  auto len = arr->lengthOf();
  int n = std::min(static_cast<int>(len), std::min(maxVals, 16));

  std::string hostStr = "null", devStr = "null";
  float vals[16];

  if (db && db->primary() != nullptr && n > 0) {
    int hn = dspReadHostValues(db, dtype, len, vals, n);
    hostStr = dspFormatValues(vals, hn);
  }
  if (db && db->special() != nullptr && n > 0 && !sd::tl_graphExecutionActive) {
    int dn = dspVerifyCopyValues(db->special(), dtype, len, vals, n);
    devStr = dspFormatValues(vals, dn);
  }

  char header[256];
  snprintf(header, sizeof(header), "dtype=%s len=%lld pAct=%d sAct=%d hostAddr=%p devAddr=%p",
            DataTypeUtils::asString(dtype).c_str(), (long long)len,
            db ? (db->isPrimaryActual() ? 1 : 0) : -1,
            db ? (db->isSpecialActual() ? 1 : 0) : -1,
            db ? db->primary() : nullptr, db ? db->special() : nullptr);
  return std::string(header) + " host=" + hostStr + " dev=" + devStr;
}

// ─── Dump all external inputs that are VARIABLE/PLACEHOLDER ──────────────
// Logs host and device values for each non-constant external input.
inline void dspDumpVariableExternals(NDArray** externalArrays, int numExt,
                                     const std::vector<bool>& isVariable,
                                     const std::vector<std::string>& names,
                                     const char* tag, int maxPerExt = 8) {
  for (int ei = 0; ei < numExt; ei++) {
    if (ei >= (int)isVariable.size() || !isVariable[ei]) continue;
    if (externalArrays[ei] == nullptr) continue;
    std::string name = (ei < (int)names.size() && !names[ei].empty()) ? names[ei] : "?";
    std::string info = dspDumpHostDeviceValues(externalArrays[ei], maxPerExt);
    DSP_DIAG(VERIFY, "EXT_DUMP(%s) ext#%d:\"%s\" %s", tag, ei, name.c_str(), info.c_str());
  }
}

// ─── Log output slot state after execution ───────────────────────────────
inline void dspLogSlotOutput(int stepIdx, const char* opName, const char* tag,
                             NDArray** outputSlots, int* outputSlotIndices,
                             int numOutputs, int totalOutputSlots) {
  for (int i = 0; i < numOutputs; i++) {
    int si = outputSlotIndices[i];
    if (si < 0 || si >= totalOutputSlots) continue;
    NDArray* out = outputSlots[si];
    if (out == nullptr) continue;
    std::string vals = dspDumpSlotValues(DSP_BUF(out), out->dataType(), out->lengthOf());
    DSP_DIAG(VERIFY, "SLOT_EXEC step=%d op=%s tag=%s -> output slot#%d dtype=%s len=%lld shape=%s vals=%s",
              stepIdx, opName, tag, si,
              DataTypeUtils::asString(out->dataType()).c_str(),
              (long long)out->lengthOf(), dspShapeStr(out).c_str(), vals.c_str());
  }
}

// ─── Source type name helper ─────────────────────────────────────────────
inline const char* dspSourceTypeName(int8_t sourceType) {
  switch (sourceType) {
    case 0: return "CONSTANT";
    case 1: return "VARIABLE";
    case 2: return "PLACEHOLDER";
    case 3: return "OP_OUTPUT";
    default: return "UNKNOWN";
  }
}

// ─── Dump a single external input by index ───────────────────────────────
// Returns a formatted diagnostic string with name, source type, dtype, length,
// actuality flags, addresses, and host/device values.
// Parameters:
//   externalArrays - array of external NDArray pointers
//   numExt         - total number of external inputs
//   isVariable     - per-ext flag: true if VARIABLE/PLACEHOLDER
//   names          - per-ext name strings
//   extIdx         - which external input index to dump
//   maxVals        - max number of leading values to include (default 8)
inline std::string dspDumpExternalInput(NDArray** externalArrays, int numExt,
                                        const std::vector<bool>& isVariable,
                                        const std::vector<std::string>& names,
                                        int extIdx, int maxVals = 8) {
  if (extIdx < 0 || extIdx >= numExt) {
    char buf[64];
    snprintf(buf, sizeof(buf), "ext#%d: OUT_OF_RANGE (numExt=%d)", extIdx, numExt);
    return std::string(buf);
  }
  NDArray* arr = externalArrays[extIdx];
  if (arr == nullptr) {
    char buf[64];
    snprintf(buf, sizeof(buf), "ext#%d: null", extIdx);
    return std::string(buf);
  }

  std::string name = (extIdx < (int)names.size() && !names[extIdx].empty())
                         ? names[extIdx] : "?";
  bool isVar = (extIdx < (int)isVariable.size()) ? isVariable[extIdx] : false;
  std::string info = dspDumpHostDeviceValues(arr, maxVals);

  char header[512];
  snprintf(header, sizeof(header), "ext#%d:\"%s\" srcType=%s shape=%s %s",
            extIdx, name.c_str(),
            isVar ? "VARIABLE" : "CONSTANT",
            dspShapeStr(arr).c_str(), info.c_str());
  return std::string(header);
}

// ─── Dump ALL external inputs ────────────────────────────────────────────
// Logs every external input (constants AND variables) with a one-line summary.
// tag: caller-provided label for context (e.g. "pre-capture", "replay-entry")
inline void dspDumpAllExternals(NDArray** externalArrays, int numExt,
                                const std::vector<bool>& isVariable,
                                const std::vector<std::string>& names,
                                const char* tag, int maxPerExt = 4) {
  DSP_DIAG(VERIFY, "ALL_EXTERNALS(%s) numExt=%d", tag, numExt);
  for (int ei = 0; ei < numExt; ei++) {
    if (externalArrays[ei] == nullptr) continue;
    std::string line = dspDumpExternalInput(externalArrays, numExt, isVariable, names, ei, maxPerExt);
    DSP_DIAG(VERIFY, "  %s", line.c_str());
  }
}

// ─── Compare host vs device buffer values for any NDArray ────────────────
// Returns the max absolute difference between host and device buffer values.
// This is the key function for detecting stale buffers (host or device out of date).
// Returns -1.0 if comparison is impossible (null arr, null buffers, during capture).
// Optionally returns per-element details via diffReport string pointer.
inline float dspCompareHostDevice(NDArray* arr, int maxVals = 16, std::string* diffReport = nullptr) {
  if (arr == nullptr) {
    if (diffReport) *diffReport = "null array";
    return -1.0f;
  }
  auto* db = arr->dataBuffer();
  if (db == nullptr) {
    if (diffReport) *diffReport = "null DataBuffer";
    return -1.0f;
  }
  if (db->primary() == nullptr) {
    if (diffReport) *diffReport = "null host buffer";
    return -1.0f;
  }
  if (db->special() == nullptr) {
    if (diffReport) *diffReport = "null device buffer";
    return -1.0f;
  }
  // D2H copies are illegal during CUDA graph capture
  if (sd::tl_graphExecutionActive) {
    if (diffReport) *diffReport = "capture active — cannot compare";
    return -1.0f;
  }

  auto dtype = arr->dataType();
  auto len = arr->lengthOf();
  int n = std::min(static_cast<int>(len), std::min(maxVals, 1024));

  // Read host values
  std::unique_ptr<float[]> hostVals(new float[n]);
  int hn = dspReadHostValues(db, dtype, len, hostVals.get(), n);

  // Read device values via D2H copy
  std::unique_ptr<float[]> devVals(new float[n]);
  int dn = dspVerifyCopyValues(db->special(), dtype, len, devVals.get(), n);

  int count = std::min(hn, dn);
  if (count <= 0) {
    if (diffReport) *diffReport = "no values to compare";
    return -1.0f;
  }

  float maxDiff = 0.0f;
  int maxDiffIdx = 0;
  int numDiffs = 0;
  for (int i = 0; i < count; i++) {
    float diff = std::abs(hostVals[i] - devVals[i]);
    if (diff > 0.0f) numDiffs++;
    if (diff > maxDiff) {
      maxDiff = diff;
      maxDiffIdx = i;
    }
  }

  if (diffReport) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "compared=%d diffs=%d maxDiff=%.6g@[%d] (host=%.6g dev=%.6g) pAct=%d sAct=%d",
             count, numDiffs, maxDiff, maxDiffIdx,
             hostVals[maxDiffIdx], devVals[maxDiffIdx],
             db->isPrimaryActual() ? 1 : 0,
             db->isSpecialActual() ? 1 : 0);
    *diffReport = std::string(buf);
  }
  return maxDiff;
}

// ─── Dump all inputs for a given slot (step) ─────────────────────────────
// For a given NativeSlot, dumps each input's source (slot reference or external),
// the source array's host/device values, actuality flags, and addresses.
// Requires the NativeSlot struct fields: numInputs, inputSourceIndices,
// inputSourceTypes, opName.
// Parameters:
//   stepIdx              - step index in the plan (for logging)
//   opName               - op name for this step
//   numInputs            - number of inputs for this slot
//   inputSourceIndices   - array of source indices (>=0 = slot, <0 = external -(idx+1))
//   inputSourceTypes     - array of NativeSourceType values
//   outputSlots          - the flat output slot array (NDArray**)
//   totalOutputSlots     - size of outputSlots array
//   externalArrays       - array of external NDArray pointers
//   numExternals         - total number of external inputs
//   externalNames        - per-external name strings (may be empty)
//   tag                  - caller-provided label
//   maxVals              - max values to dump per input (default 4)
inline void dspDumpSlotInputs(int stepIdx, const char* opName,
                              int numInputs, const int* inputSourceIndices,
                              const int8_t* inputSourceTypes,
                              NDArray** outputSlots, int totalOutputSlots,
                              NDArray** externalArrays, int numExternals,
                              const std::vector<std::string>& externalNames,
                              const char* tag, int maxVals = 4) {
  DSP_DIAG(VERIFY, "SLOT_INPUTS(%s) step=%d op=%s numInputs=%d", tag, stepIdx, opName, numInputs);
  for (int i = 0; i < numInputs; i++) {
    int srcIdx = inputSourceIndices[i];
    int8_t srcType = (inputSourceTypes != nullptr) ? inputSourceTypes[i] : -1;
    NDArray* srcArr = nullptr;
    std::string srcLabel;

    if (srcIdx >= 0) {
      // Slot reference (output of a prior op)
      if (srcIdx < totalOutputSlots) {
        srcArr = outputSlots[srcIdx];
      }
      char buf[128];
      snprintf(buf, sizeof(buf), "input#%d: slot[%d] srcType=%s",
                i, srcIdx, dspSourceTypeName(srcType));
      srcLabel = buf;
    } else {
      // External input: -(srcIdx + 1)
      int extIdx = -(srcIdx + 1);
      std::string name = "?";
      if (extIdx >= 0 && extIdx < numExternals) {
        srcArr = externalArrays[extIdx];
        if (extIdx < (int)externalNames.size() && !externalNames[extIdx].empty()) {
          name = externalNames[extIdx];
        }
      }
      char buf[256];
      snprintf(buf, sizeof(buf), "input#%d: ext[%d]=\"%s\" srcType=%s",
                i, extIdx, name.c_str(), dspSourceTypeName(srcType));
      srcLabel = buf;
    }

    if (srcArr != nullptr) {
      std::string info = dspDumpHostDeviceValues(srcArr, maxVals);
      DSP_DIAG(VERIFY, "  %s shape=%s %s", srcLabel.c_str(), dspShapeStr(srcArr).c_str(), info.c_str());
    } else {
      DSP_DIAG(VERIFY, "  %s -> null array", srcLabel.c_str());
    }
  }
}

// ─── Dump external input index -> device buffer address map ──────────────
// Builds and logs a map of external input index to its device buffer address.
// Useful for comparing capture-time vs replay-time addresses to detect
// address instability that would break CUDA graph replay.
// Returns the map for programmatic use.
// Parameters:
//   externalArrays - array of external NDArray pointers
//   numExt         - total number of external inputs
//   names          - per-external name strings (may be empty)
//   tag            - caller-provided label (e.g. "capture", "replay-3")
inline std::unordered_map<int, void*> dspDumpAddressMap(NDArray** externalArrays, int numExt,
                                                        const std::vector<std::string>& names,
                                                        const char* tag) {
  std::unordered_map<int, void*> addrMap;
  DSP_DIAG(VERIFY, "ADDR_MAP(%s) numExt=%d", tag, numExt);
  for (int ei = 0; ei < numExt; ei++) {
    NDArray* arr = externalArrays[ei];
    if (arr == nullptr) continue;
    void* devAddr = DSP_BUF(arr);
    addrMap[ei] = devAddr;
    std::string name = (ei < (int)names.size() && !names[ei].empty()) ? names[ei] : "?";
    auto* db = arr->dataBuffer();
    DSP_DIAG(VERIFY, "  ext#%d:\"%s\" devAddr=%p hostAddr=%p len=%lld dtype=%s pAct=%d sAct=%d",
              ei, name.c_str(), devAddr,
              db ? db->primary() : nullptr,
              (long long)arr->lengthOf(),
              DataTypeUtils::asString(arr->dataType()).c_str(),
              db ? (db->isPrimaryActual() ? 1 : 0) : -1,
              db ? (db->isSpecialActual() ? 1 : 0) : -1);
  }
  return addrMap;
}

// ─── Compare two address maps and report mismatches ──────────────────────
// Compares two address maps (from dspDumpAddressMap) and logs differences.
// Returns the number of mismatches.
// Parameters:
//   mapA, mapB     - address maps to compare
//   names          - per-external name strings (for readable output)
//   tagA, tagB     - labels identifying each map (e.g. "capture", "replay")
inline int dspCompareAddressMaps(const std::unordered_map<int, void*>& mapA,
                                 const std::unordered_map<int, void*>& mapB,
                                 const std::vector<std::string>& names,
                                 const char* tagA, const char* tagB) {
  int mismatches = 0;
  // Check all entries in mapA against mapB
  for (const auto& entry : mapA) {
    int ei = entry.first;
    void* addrA = entry.second;
    auto it = mapB.find(ei);
    if (it == mapB.end()) {
      std::string name = (ei < (int)names.size() && !names[ei].empty()) ? names[ei] : "?";
      DSP_DIAG(VERIFY, "ADDR_MISMATCH ext#%d:\"%s\" present in %s(%p) but MISSING in %s",
                ei, name.c_str(), tagA, addrA, tagB);
      mismatches++;
    } else if (it->second != addrA) {
      std::string name = (ei < (int)names.size() && !names[ei].empty()) ? names[ei] : "?";
      DSP_DIAG(VERIFY, "ADDR_MISMATCH ext#%d:\"%s\" %s=%p vs %s=%p",
                ei, name.c_str(), tagA, addrA, tagB, it->second);
      mismatches++;
    }
  }
  // Check for entries in mapB that are missing from mapA
  for (const auto& entry : mapB) {
    if (mapA.find(entry.first) == mapA.end()) {
      int ei = entry.first;
      std::string name = (ei < (int)names.size() && !names[ei].empty()) ? names[ei] : "?";
      DSP_DIAG(VERIFY, "ADDR_MISMATCH ext#%d:\"%s\" MISSING in %s but present in %s(%p)",
                ei, name.c_str(), tagA, tagB, entry.second);
      mismatches++;
    }
  }
  if (mismatches == 0) {
    DSP_DIAG(VERIFY, "ADDR_COMPARE %s vs %s: ALL %d addresses match", tagA, tagB, (int)mapA.size());
  } else {
    DSP_DIAG(VERIFY, "ADDR_COMPARE %s vs %s: %d MISMATCHES out of %d entries",
              tagA, tagB, mismatches, (int)mapA.size());
  }
  return mismatches;
}

#endif  // SD_CUDA

// ═══════════════════════════════════════════════════════════════════════════
// Output validation flags (returned as bitmask per output)
// ═══════════════════════════════════════════════════════════════════════════
constexpr int DSP_VALIDATE_OK         = 0;
constexpr int DSP_VALIDATE_NULL       = (1 << 0);
constexpr int DSP_VALIDATE_NAN        = (1 << 1);
constexpr int DSP_VALIDATE_INF        = (1 << 2);
constexpr int DSP_VALIDATE_ALL_ZERO   = (1 << 3);

/**
 * Validate plan outputs after execution. For each output, checks:
 * - null pointer
 * - contains NaN (via sum-based detection: sum != sum)
 * - contains Inf
 * - all zeros (norm2 == 0)
 *
 * @param outputs     array of output NDArray pointers
 * @param numOutputs  number of outputs
 * @param flagsOut    int array of size numOutputs, filled with DSP_VALIDATE_* bitmask per output
 * @return number of outputs with validation issues (non-zero flags)
 */
SD_INLINE int dspValidateOutputs(NDArray** outputs, int numOutputs, int* flagsOut) {
  int issues = 0;
  for (int i = 0; i < numOutputs; i++) {
    flagsOut[i] = DSP_VALIDATE_OK;
    if (outputs[i] == nullptr) {
      flagsOut[i] = DSP_VALIDATE_NULL;
      issues++;
      continue;
    }
    NDArray* arr = outputs[i];
    auto len = arr->lengthOf();
    if (len == 0) continue;

    // Sum-based NaN/Inf detection (works for both CPU and CUDA)
    auto sum = arr->sumNumber();
    float sumVal = sum.e<float>(0);
    if (std::isnan(sumVal)) {
      flagsOut[i] |= DSP_VALIDATE_NAN;
    }
    if (std::isinf(sumVal)) {
      flagsOut[i] |= DSP_VALIDATE_INF;
    }

    // All-zeros detection via norm2
    auto norm = arr->reduceNumber(reduce::Norm2);
    float normVal = norm->e<float>(0);
    delete norm;
    if (normVal == 0.0f && len > 0) {
      flagsOut[i] |= DSP_VALIDATE_ALL_ZERO;
    }

    if (flagsOut[i] != DSP_VALIDATE_OK) issues++;
  }
  return issues;
}

/**
 * Detect stale outputs by comparing current outputs against previously stored fingerprints.
 * A "stale" output is one that is identical (within epsilon) to the previous step's output.
 *
 * @param outputs       array of current output NDArray pointers
 * @param numOutputs    number of outputs
 * @param prevNorms     array of norm2 values from previous step (updated in-place)
 * @param staleOut      bool array of size numOutputs, set to true for stale outputs
 * @param epsilon       threshold below which norm2 difference is considered stale (default 1e-10)
 * @return number of stale outputs detected
 */
SD_INLINE int dspDetectStaleOutputs(NDArray** outputs, int numOutputs,
                                     float* prevNorms, bool* staleOut, float epsilon = 1e-10f) {
  int staleCount = 0;
  for (int i = 0; i < numOutputs; i++) {
    staleOut[i] = false;
    if (outputs[i] == nullptr) {
      prevNorms[i] = 0.0f;
      continue;
    }
    auto norm = outputs[i]->reduceNumber(reduce::Norm2);
    float normVal = norm->e<float>(0);
    delete norm;
    float diff = std::abs(normVal - prevNorms[i]);
    if (diff < epsilon && prevNorms[i] != 0.0f) {
      staleOut[i] = true;
      staleCount++;
    }
    prevNorms[i] = normVal;
  }
  return staleCount;
}

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_VERIFY_UTILS_H
