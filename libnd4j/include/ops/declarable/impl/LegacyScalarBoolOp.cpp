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
#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/LegacyScalarBoolOp.h>

#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

#include <cstring>

namespace sd {
#ifdef SD_CUDA
SD_LIB_EXPORT bool isCudaGraphCaptureActiveForScalarOps(void *stream);
SD_LIB_EXPORT bool orderScalarH2DForKernelStream(void *kernelStreamPtr);
#endif
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
namespace {
NDArray *scalarOperandOnCurrentDevice(NDArray *cached, LaunchContext *context) {
  if (cached == nullptr || cached->dataBuffer() == nullptr)
    THROW_EXCEPTION("LegacyScalarBoolOp: cached scalar has no valid DataBuffer");

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
    // Mirror the LegacyScalarOp fix: order the scalar's H2D (LC stream) into
    // the kernel's context stream directly via the exported DataBuffer helper.
    // Capture-guarded and no-op when both work is already on one stream.
    auto *kernelStreamPtr = context != nullptr ? context->getCudaStream() : nullptr;
    if (kernelStreamPtr != nullptr &&
        orderScalarH2DForKernelStream(*kernelStreamPtr)) {
      DSP_DIAG(MULTI_DEVICE,
               "SCALAR_BOOL_OPERAND: kernel stream %p ordered after scalar H2D on LC stream",
               (void *)*kernelStreamPtr);
    }
    DSP_DIAG(MULTI_DEVICE,
             "SCALAR_BOOL_OPERAND: cachedDb=%p device=%d targetDevice=%d capture=%d special=%p replica=0",
             static_cast<void *>(cachedBuffer), cachedDevice, device, capturing ? 1 : 0,
             cachedBuffer->special());
    return cached;
  }

  if (!capturing) {
    cached->syncToDevice();
    if (cachedBuffer->deviceId() == device) {
      // Mirror the LegacyScalarOp fix: order the scalar's H2D (LC stream) into
      // the kernel's context stream via its recorded write event, so the
      // kernel cannot read pre-upload memory. The exported scalar-op helper
      // delegates to the shared capture authority (plain op TUs cannot see
      // DebugHelper's CUDA inline overloads); stream-order only.
      auto *kernelStreamPtr = context != nullptr ? context->getCudaStream() : nullptr;
      void *kernelStreamValue = (kernelStreamPtr != nullptr) ? *kernelStreamPtr : nullptr;
      if (kernelStreamValue != nullptr &&
          !isCudaGraphCaptureActiveForScalarOps(kernelStreamPtr)) {
        cachedBuffer->waitForSpecialWriteEvent(kernelStreamValue);
        DSP_DIAG(MULTI_DEVICE,
                 "SCALAR_BOOL_OPERAND: kernel stream %p waits for cached scalar H2D write event db=%p",
                 kernelStreamValue, (void *)cachedBuffer);
      }
      DSP_DIAG(MULTI_DEVICE,
               "SCALAR_BOOL_OPERAND: cachedDb=%p device=%d targetDevice=%d capture=0 special=%p replica=0",
               static_cast<void *>(cachedBuffer), cachedBuffer->deviceId(), device,
               cachedBuffer->special());
      return cached;
    }
    THROW_EXCEPTION("LegacyScalarBoolOp: cached scalar could not be migrated to the execution device");
  }

  if (!cached->isActualOnHostSide())
    THROW_EXCEPTION("LegacyScalarBoolOp: cannot rehome a stale host scalar during graph capture");
  auto replica = new NDArray(cached->dataType(), context);
  try {
    std::memcpy(replica->buffer(), cached->buffer(), cached->sizeOfT());
    replica->tickWriteHost();
    replica->syncToDevice();
    auto *replicaBuffer = replica->dataBuffer();
    if (replicaBuffer == nullptr || replicaBuffer->deviceId() != device)
      THROW_EXCEPTION("LegacyScalarBoolOp: capture scalar replica was not allocated on the execution device");
    DSP_DIAG(MULTI_DEVICE,
             "SCALAR_BOOL_OPERAND: cachedDb=%p cachedDevice=%d targetDevice=%d capture=1 replicaDb=%p replicaDevice=%d special=%p replica=1",
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

LegacyScalarBoolOp::LegacyScalarBoolOp() : LegacyOp(1) {
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_COMPARISON |
      OP_TRAIT_FULLY_WRITING);
}

LegacyScalarBoolOp::LegacyScalarBoolOp(int opNum) : LegacyOp(1, opNum) {
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_COMPARISON |
      OP_TRAIT_FULLY_WRITING);
}

LegacyOp *LegacyScalarBoolOp::clone() {
  return _scalar == nullptr ? new LegacyScalarBoolOp(this->_opNum)
                            : new LegacyScalarBoolOp(this->_opNum, *this->_scalar);
}

LegacyScalarBoolOp::LegacyScalarBoolOp(int opNum, NDArray &scalar) : LegacyOp(1, opNum) {
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_COMPARISON |
      OP_TRAIT_FULLY_WRITING);
  _scalar = scalar.dup(scalar.ordering(), false);
}

void LegacyScalarBoolOp::registerTypes() {
  // Bool ops produce BOOL output regardless of input type — NOT same mode.
  this->getOpDescriptor()->setSameMode(false);
  this->getOpDescriptor()->setAllowedOutputTypes({BOOL});
  this->getOpDescriptor()->setAllowedInputTypes(ANY);
}

ShapeList *LegacyScalarBoolOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);
  // Bool ops always produce BOOL output regardless of input type
  return SHAPELIST(ConstantShapeHelper::getInstance().castToDataType(inShape, BOOL));
}

Status LegacyScalarBoolOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  ExtraArguments extras(*block.getTArguments());

  if (block.width() > 1) {
    auto y = INPUT_VARIABLE(1);

    NDArray::prepareSpecialUse({z}, {x, y});

    NativeOpExecutioner::execScalarBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                        x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                        z->specialShapeInfo(), y->buffer(), y->shapeInfo(), y->specialBuffer(),
                                        y->specialShapeInfo(), extras.argumentsAsT(x->dataType()));

    NDArray::registerSpecialUse({z}, {x, y});
  } else if (block.getTArguments()->size() > 0) {
    // Cache the scalar NDArray in _scalar to avoid creating and destroying a
    // temporary on every call.  During CUDA graph capture the kernel records
    // the device address of the scalar buffer; if that buffer is freed before
    // replay the graph reads stale/garbage memory for the scalar value.
    // Caching keeps the buffer alive for the entire op lifetime.
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

    NativeOpExecutioner::execScalarBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                        x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                        z->specialShapeInfo(), scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(),
                                        scalar->specialShapeInfo(),
                                        extras.length() > 1 ? extras.argumentsAsT(x->dataType(), 1) : nullptr);

    NDArray::registerSpecialUse({z}, {x, scalar});
  } else {
    REQUIRE_TRUE(_scalar != nullptr, 0,
                 "LegacyScalarBoolOp: no scalar value provided (neither via tArgs, input[1], nor pre-set _scalar). "
                 "OpNum=%d. This typically means the DSP plan compiler did not extract the scalar value.", opNum);
    auto scalar = scalarOperandOnCurrentDevice(_scalar, block.launchContext());
    ScalarReplica replica{scalar == _scalar ? nullptr : scalar};
    NDArray::prepareSpecialUse({z}, {x, scalar});

    NativeOpExecutioner::execScalarBool(
        block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), scalar->buffer(), scalar->shapeInfo(),
        scalar->specialBuffer(), scalar->specialShapeInfo(), extras.argumentsAsT(x->dataType()));

    NDArray::registerSpecialUse({z}, {x, scalar});
  }
  STORE_RESULT(*z);
  traceExecIfNeeded(block);

  return Status::OK;
}
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd
