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
// Created by raver119 on 16.10.2017.
//
#include <array/NDArrayFactory.h>
#include <execution/AffinityManager.h>
#include <graph/DspDiagnostics.h>
#include <helpers/DebugHelper.h>
#include <ops/declarable/LegacyScalarOp.h>

#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

#include <cstring>

namespace sd {
#ifdef SD_CUDA
SD_LIB_EXPORT bool isCudaGraphCaptureActiveForScalarOps(void *stream);
#endif
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
namespace {
// The cached scalar outlives the device it was built on: a DSP capture rehome
// warms a plan up on one device and captures it on another, and a kernel that
// reads a non-peer device's buffer faults (err700). Outside capture migrate()
// moves the cached buffer in place. During capture migrate() declines so the
// buffer never moves under recorded work; the call then gets a replica built
// on the current device, whose capture allocation and H2D staging are
// graph-owned and stay valid for every replay.
NDArray *scalarOperandOnCurrentDevice(NDArray *cached, LaunchContext *context) {
  if (cached == nullptr || cached->dataBuffer() == nullptr)
    THROW_EXCEPTION("LegacyScalarOp: cached scalar has no valid DataBuffer");

  const int device = AffinityManager::currentDeviceId();
  auto *cachedBuffer = cached->dataBuffer();
  const int cachedDevice = cachedBuffer->deviceId();
  auto *stream = context != nullptr ? context->getCudaStream() : nullptr;
#ifdef SD_CUDA
  const bool capturing = isCudaGraphCaptureActiveForScalarOps(static_cast<void *>(stream));
#else
  const bool capturing = false;
#endif
  if (cachedDevice == device) {
    // The scalar buffer was created (or last synced) on this device, with its
    // H2D enqueued on the caller's LC stream while this op's kernel runs on the
    // context stream. syncToSpecial recorded the write event; make the kernel
    // stream wait on it so the kernel cannot read pre-upload device memory —
    // under compute-sanitizer this deterministically produced x + 0 instead of
    // x + scalar. On later executions the event is long-complete and this wait
    // is a cheap no-op. Stream-order only; the exported scalar-op helper
    // delegates to the shared capture authority (plain op TUs cannot see
    // DebugHelper's CUDA inline overloads), and waits are never recorded
    // inside a capture region.
    auto *kernelStreamPtr = context != nullptr ? context->getCudaStream() : nullptr;
    void *kernelStreamValue = (kernelStreamPtr != nullptr) ? *kernelStreamPtr : nullptr;
    if (kernelStreamValue != nullptr &&
        !isCudaGraphCaptureActiveForScalarOps(kernelStreamPtr)) {
      cachedBuffer->waitForSpecialWriteEvent(kernelStreamValue);
      DSP_DIAG(MULTI_DEVICE,
               "SCALAR_OPERAND: kernel stream %p waits for cached scalar H2D write event db=%p",
               kernelStreamValue, (void *)cachedBuffer);
    }
    DSP_DIAG(MULTI_DEVICE,
             "SCALAR_OPERAND: cachedDb=%p device=%d targetDevice=%d capture=%d special=%p replica=0",
             static_cast<void *>(cachedBuffer), cachedDevice, device, capturing ? 1 : 0,
             cachedBuffer->special());
    return cached;
  }

  // Never ask DataBuffer::migrate to move an address while a graph is being
  // recorded. The migration guard preserves the old allocation for live plans,
  // but NDArray::syncToDevice still updates its local affinity. During capture,
  // make a capture-workspace replica from the scalar's authoritative host copy.
  if (!capturing) {
    cached->syncToDevice();
    if (cachedBuffer->deviceId() == device) {
      // The H2D above is enqueued on the caller's LC stream (DataBuffer routes
      // away from tl_dspExecutionStream when tl_graphExecutionActive and
      // tl_dspReplayActive are both false), while this op's kernel runs on the
      // context stream. syncToSpecial recorded a write event; make the kernel
      // stream wait on it so the kernel cannot read pre-upload device memory —
      // under compute-sanitizer this deterministically produced x + 0 instead
      // of x + scalar. Stream-order only, and never inside a capture region
      // (the exported scalar-op helper delegates to the shared capture
      // authority; plain op TUs cannot see DebugHelper's CUDA inline
      // overloads).
      auto *kernelStreamPtr = context != nullptr ? context->getCudaStream() : nullptr;
      void *kernelStreamValue = (kernelStreamPtr != nullptr) ? *kernelStreamPtr : nullptr;
      if (kernelStreamValue != nullptr &&
          !isCudaGraphCaptureActiveForScalarOps(kernelStreamPtr)) {
        cachedBuffer->waitForSpecialWriteEvent(kernelStreamValue);
        DSP_DIAG(MULTI_DEVICE,
                 "SCALAR_OPERAND: kernel stream %p waits for cached scalar H2D write event db=%p",
                 kernelStreamValue, (void *)cachedBuffer);
      }
      DSP_DIAG(MULTI_DEVICE,
               "SCALAR_OPERAND: cachedDb=%p device=%d targetDevice=%d capture=0 special=%p replica=0",
               static_cast<void *>(cachedBuffer), cachedBuffer->deviceId(), device,
               cachedBuffer->special());
      return cached;
    }
    THROW_EXCEPTION("LegacyScalarOp: cached scalar could not be migrated to the execution device");
  }

  if (!cached->isActualOnHostSide())
    THROW_EXCEPTION("LegacyScalarOp: cannot rehome a stale host scalar during graph capture");
  auto replica = new NDArray(cached->dataType(), context);
  try {
    std::memcpy(replica->buffer(), cached->buffer(), cached->sizeOfT());
    replica->tickWriteHost();
    replica->syncToDevice();
    auto *replicaBuffer = replica->dataBuffer();
    if (replicaBuffer == nullptr || replicaBuffer->deviceId() != device)
      THROW_EXCEPTION("LegacyScalarOp: capture scalar replica was not allocated on the execution device");
    DSP_DIAG(MULTI_DEVICE,
             "SCALAR_OPERAND: cachedDb=%p cachedDevice=%d targetDevice=%d capture=1 replicaDb=%p replicaDevice=%d special=%p replica=1",
             static_cast<void *>(cachedBuffer), cachedDevice, device,
             static_cast<void *>(replicaBuffer), replicaBuffer->deviceId(), replicaBuffer->special());
  } catch (...) {
    delete replica;
    throw;
  }
  return replica;
}

struct ScalarReplica {
  NDArray *array;
  ~ScalarReplica() { delete array; }
};
}  // namespace

LegacyScalarOp::LegacyScalarOp() : LegacyOp(1) {
  this->getOpDescriptor()->allowInplace(true);
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

LegacyScalarOp::LegacyScalarOp(int opNum) : LegacyOp(1, opNum) {
  this->getOpDescriptor()->allowInplace(true);
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

LegacyOp *LegacyScalarOp::clone() {
  return _scalar == nullptr ? new LegacyScalarOp(this->_opNum) : new LegacyScalarOp(this->_opNum, *this->_scalar);
}

LegacyScalarOp::LegacyScalarOp(int opNum, NDArray &scalar) : LegacyOp(1, opNum) {
  this->getOpDescriptor()->allowInplace(true);
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
  _scalar = scalar.dup(scalar.ordering(), false);
}

ShapeList *LegacyScalarOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);

  LongType *newShape;
  COPY_SHAPE(inShape, newShape);

  return SHAPELIST(CONSTANT(newShape));
}

Status LegacyScalarOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  ExtraArguments extras(*block.getTArguments());

  if (block.width() > 1) {
    auto y = INPUT_VARIABLE(1);

    NDArray::prepareSpecialUse({z}, {x, y});

    NativeOpExecutioner::execScalar(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                    x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                    z->specialShapeInfo(), y->buffer(), y->shapeInfo(), y->specialBuffer(),
                                    y->specialShapeInfo(), extras.argumentsAsT(z->dataType()));

    NDArray::registerSpecialUse({z}, {x, y});
  } else if (block.getTArguments()->size() > 0) {
    // Cache the scalar NDArray in _scalar to avoid creating and destroying a
    // temporary on every call.  During CUDA graph capture the kernel records
    // the device address of the scalar buffer; if that buffer is freed before
    // replay the graph reads stale/garbage memory for the scalar value.
    // Caching keeps the buffer alive for the entire op lifetime, which covers
    // both capture and all subsequent replays.
    double scalarVal = T_ARG(0);
    auto xDt = x->dataType();
    if (_scalar == nullptr || !_cachedScalarValid ||
        _cachedScalarType != xDt || _cachedScalarValue != scalarVal) {
      delete _scalar;
      _scalar = NDArrayFactory::create(xDt, scalarVal, block.launchContext());
      _cachedScalarValid = true;
      _cachedScalarValue = scalarVal;
      _cachedScalarType = xDt;
    }

    auto scalar = scalarOperandOnCurrentDevice(_scalar, block.launchContext());
    ScalarReplica replica{scalar == _scalar ? nullptr : scalar};

    NDArray::prepareSpecialUse({z}, {x, scalar});

    NativeOpExecutioner::execScalar(
        block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), scalar->buffer(), scalar->shapeInfo(),
        scalar->specialBuffer(), scalar->specialShapeInfo(),
        extras.length() > 1 ? extras.argumentsAsT(z->dataType(), 1) : nullptr);

    NDArray::registerSpecialUse({z}, {x, scalar});
  } else {
    REQUIRE_TRUE(_scalar != nullptr, 0,
                 "LegacyScalarOp: no scalar value provided (neither via tArgs, input[1], nor pre-set _scalar). "
                 "OpNum=%d. This typically means the DSP plan compiler did not extract the scalar value.", opNum);
    auto scalar = scalarOperandOnCurrentDevice(_scalar, block.launchContext());
    ScalarReplica replica{scalar == _scalar ? nullptr : scalar};

    NDArray::prepareSpecialUse({z}, {x, scalar});

    NativeOpExecutioner::execScalar(
        block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), scalar->buffer(), scalar->shapeInfo(),
        scalar->specialBuffer(), scalar->specialShapeInfo(), extras.argumentsAsT(z->dataType()));

    NDArray::registerSpecialUse({z}, {x, scalar});
  }


  traceExecIfNeeded(block);

  return Status::OK;
}
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd
