/* ******************************************************************************
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
// Split from NativeOps.cu to reduce object file size for SD_GCC_FUNCTRACE builds
// Contains: execCustomOp2
//

#include <cuda.h>
#include <cstdio>
#include <string>
#include <exceptions/cuda_exception.h>
#include <execution/LaunchContext.h>
#include <graph/DspLifecycleContext.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/OpExecutionLogger.h>
#include <graph/OpContextLifecycleTracker.h>
#include <system/common.h>

// Function to execute a custom operation with context
sd::Status execCustomOp2(sd::Pointer *extraPointers, sd::LongType hash, Context *opContext) {
  try {
    // Tripwire: validate Context pointer
    if (opContext == nullptr) {
      THROW_EXCEPTION("execCustomOp2: opContext is null");
    }
    uintptr_t ctxAddr = reinterpret_cast<uintptr_t>(opContext);
    if (ctxAddr < 0x10000 || (ctxAddr & 0x7) != 0) {
      std::string error = "execCustomOp2: opContext pointer appears invalid: 0x";
      char buf[32];
      snprintf(buf, sizeof(buf), "%lx", static_cast<unsigned long>(ctxAddr));
      error += buf;
      THROW_EXCEPTION(error.c_str());
    }

    // Tripwire: capture _context value before op execution
    sd::LaunchContext* contextBefore = opContext->launchContext();

    // Retrieve the operation based on the hash
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    if (op == nullptr) {
      throw std::invalid_argument("Operation not found for the given hash.");
    }

#if defined(SD_GCC_FUNCTRACE)
    // Set op name BEFORE execute() so allocations during execution are tagged
    if (op->getOpName() != nullptr) {
        sd::ops::OpExecutionLogger::setCurrentOpName(*op->getOpName());
        // Also update the already-tracked context with the op name
        sd::graph::OpContextLifecycleTracker::getInstance().updateContextOpName(opContext, *op->getOpName());
    }
#endif

    // Execute the custom operation with the provided context
    auto result = op->execute(opContext);

    // Tripwire: check if _context was corrupted during op execution
    sd::LaunchContext* contextAfter = opContext->launchContext();
    if (contextAfter != contextBefore) {
      std::string error = "execCustomOp2: _context was corrupted during op execution. Op: ";
      if (op->getOpName() != nullptr) {
        error += *op->getOpName();
      } else {
        error += "unknown";
      }
      error += ", before: 0x";
      char buf1[32], buf2[32];
      snprintf(buf1, sizeof(buf1), "%lx", reinterpret_cast<unsigned long>(contextBefore));
      snprintf(buf2, sizeof(buf2), "%lx", reinterpret_cast<unsigned long>(contextAfter));
      error += buf1;
      error += ", after: 0x";
      error += buf2;
      THROW_EXCEPTION(error.c_str());
    }

    // Don't sync here - let CUDA operations run asynchronously
    // The prepareSpecialUse/registerSpecialUse pattern handles data dependencies

    // After GPU execution completes, update actuality tracking:
    // - Outputs were WRITTEN on device, so mark device as having latest data
    // - Inputs were READ on device
    //
    // IMPORTANT: We must NOT call syncToDevice() here - that copies FROM host TO device,
    // but the data is already on device from the kernel execution. Instead, we need to
    // update the actuality counters so that subsequent syncToHost() calls will know
    // to copy the data from device to host.
    //
    // This is equivalent to calling registerSpecialUse({outputs}, {inputs}).
    //
    // DSP gate: when the current thread is executing under a live DSP capture
    // or replay, ticking these counters here would clobber the actuality state
    // DSP already established for the captured graph. Defer to DSP's own
    // reconciler (NativeDynamicShapePlan_slotexec::reconcileExecutedOutputActuality).
    // COEXIST_SAFE (default) is the mode that enables this gate — LEGACY_UNAWARE
    // falls through for bisecting regressions. See graph/DspLifecycleContext.h.
    const bool skipTick = sd::graph::DspLifecycleContext::shouldSkipTick();
    if (!skipTick) {
      for (auto v : opContext->fastpath_in()) {
        if (v != nullptr && !v->isEmpty()) v->tickReadDevice();
      }

      for (auto v : opContext->fastpath_out()) {
        if (v != nullptr && !v->isEmpty()) v->tickWriteDevice();
      }
    }

#if defined(SD_GCC_FUNCTRACE)
    sd::ops::OpExecutionLogger::clearCurrentOpName();
#endif

    return result;
  }
  catch (std::exception &e) {
#if defined(SD_GCC_FUNCTRACE)
    sd::ops::OpExecutionLogger::clearCurrentOpName();
#endif
    // Handle exceptions by setting error codes and messages
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::KERNEL_FAILURE;
  }
}
