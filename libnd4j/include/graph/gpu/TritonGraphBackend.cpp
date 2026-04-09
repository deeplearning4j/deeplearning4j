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
// TritonGraphBackend core: singleton, configuration, availability, fusibility.
//
// Method implementations are split across focused compilation units:
//   TritonGraphBackend_cache.cpp   — disk cache (read/write PTX + metadata)
//   TritonGraphBackend_compile.cpp — segment compilation (splitting, parallel work-stealing)
//   TritonGraphBackend_kernel.cpp  — single kernel launch + arg table + replay
//   TritonGraphBackend_execute.cpp — segment execution (sub-kernel loop, gaps, diagnostics)
//   TritonGraphBackend_binary.cpp  — GPU binary compilation (IR → PTX → module)
//
// Shared internal helpers live in TritonGraphBackend_internal.h.
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonGraphBackend.h>
#include <graph/gpu/TritonIRBuilder.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <mutex>

namespace sd {
namespace graph {

// ─── Static member initialization ───────────────────────────────────────────

int TritonGraphBackend::maxParallelCompilations_ = DEFAULT_MAX_PARALLEL_COMPILATIONS;
std::mutex TritonGraphBackend::configMtx_;
thread_local TritonGraphBackend::OrderedRangeExecutor TritonGraphBackend::orderedRangeExecutor_ = nullptr;

// ─── Parallel compilation configuration ─────────────────────────────────────

int TritonGraphBackend::getMaxParallelCompilations() {
  std::lock_guard<std::mutex> lock(configMtx_);

  int configuredThreads = sd::Environment::getInstance().tritonBuildThreads();
  if (configuredThreads > 0 && configuredThreads <= 16) {
    maxParallelCompilations_ = configuredThreads;
  } else {
    maxParallelCompilations_ = DEFAULT_MAX_PARALLEL_COMPILATIONS;
  }

  static int lastReported = -1;
  if (lastReported != maxParallelCompilations_) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: Using %d parallel compilation threads (Environment)",
             maxParallelCompilations_);
    lastReported = maxParallelCompilations_;
  }

  return maxParallelCompilations_;
}

void TritonGraphBackend::setMaxParallelCompilations(int maxThreads) {
  std::lock_guard<std::mutex> lock(configMtx_);
  if (maxThreads > 0 && maxThreads <= 16) {
    maxParallelCompilations_ = maxThreads;
    DSP_DIAG(COMPILE, "TritonGraphBackend: Set max parallel compilations to %d", maxThreads);
  } else {
    DSP_DIAG(COMPILE, "TritonGraphBackend: Invalid maxThreads=%d (must be 1-16), keeping %d",
             maxThreads, maxParallelCompilations_);
  }
}

// ─── Singleton ──────────────────────────────────────────────────────────────

TritonGraphBackend& TritonGraphBackend::getInstance() {
  static TritonGraphBackend instance;
  return instance;
}

void TritonGraphBackend::setOrderedRangeExecutor(OrderedRangeExecutor executor) {
  orderedRangeExecutor_ = std::move(executor);
}

void TritonGraphBackend::clearOrderedRangeExecutor() {
  orderedRangeExecutor_ = nullptr;
}

TritonGraphBackend::TritonGraphBackend() = default;

TritonGraphBackend::~TritonGraphBackend() {
  invalidateCache();
}

// ─── Availability ───────────────────────────────────────────────────────────

bool TritonGraphBackend::isAvailable() const {
  return TritonTargetDispatch::isReady();
}

// ─── Check if all ops in a range are Triton-mappable ────────────────────────

bool TritonGraphBackend::areAllOpsMappable(NativeSlot* slots, int start, int end) {
  for (int i = start; i <= end; i++) {
    if (!TritonIRBuilder::isTritonMappable(slots[i].opName)) {
      return false;
    }
  }
  return true;
}

// ─── Segment fusibility check ───────────────────────────────────────────────
//
// A segment is fusible if it contains at least one Triton-mappable op.
// Non-mappable ops (matmul, gather, etc.) become native ordered ranges inside
// compileSegment() — they run in program order while the mappable
// chains get compiled into Triton kernels.  The previous version required
// ALL ops to be mappable, which caused the post-freeze mega-segment
// (1 segment, 3407 ops) to always fail canFuseSegment → launches=0.

bool TritonGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) return false;

  int totalOps = end - start + 1;
  if (totalOps < MIN_MAPPABLE_OPS) return false;

  // Check if at least one op is Triton-mappable.
  // compileSegment handles mixed segments via isFallbackSection.
  for (int i = start; i <= end; i++) {
    if (TritonIRBuilder::isTritonMappable(slots[i].opName)) {
      return true;
    }
  }

  return false;  // No mappable ops at all
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
