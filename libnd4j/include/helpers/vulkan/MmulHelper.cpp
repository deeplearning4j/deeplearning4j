/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN)

#include <execution/LaunchContext.h>
#include <helpers/MmulHelper.h>
#include <helpers/shape.h>
#include <ops/declarable/headers/blas.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#endif

#include <string>
#include <vector>

namespace sd {
namespace {

void executeMatmulDescriptor(NDArray* a, NDArray* b, NDArray* c,
                             double alpha, double beta) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    THROW_EXCEPTION("Vulkan MmulHelper received a null array");
  }

#if defined(HAVE_VULKAN) && HAVE_VULKAN && NOT_EXCLUDED(OP_matmul)
  std::vector<NDArray*> inputs{a, b};
  std::vector<NDArray*> outputs{c};

  if (a->getDataBuffer() == nullptr || b->getDataBuffer() == nullptr ||
      c->getDataBuffer() == nullptr) {
    THROW_EXCEPTION("Vulkan MmulHelper received an array without a data buffer");
  }

  const int deviceId = a->getDataBuffer()->deviceId();
  LaunchContext* launchContext = a->getContext();
  if (deviceId < 0 || launchContext == nullptr ||
      launchContext->getDeviceID() != deviceId) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper received an invalid input device context");
  }

  for (auto* input : inputs) {
    if (input->getDataBuffer()->deviceId() != deviceId) {
      THROW_EXCEPTION(
          "Vulkan MmulHelper inputs must reside on one physical device");
    }
  }
  if (c->getDataBuffer()->deviceId() != deviceId) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper output must reside on the input device");
  }

  const auto contextStream = graph::vulkanExecutionStream(launchContext);
  auto* stream =
      contextStream == nullptr
          ? graph::VulkanExecutionStream::defaultExecution(deviceId)
          : graph::VulkanExecutionStream::fromOpaque(contextStream, false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    THROW_EXCEPTION(
        contextStream == nullptr
            ? "Vulkan MmulHelper could not resolve the exact-device default "
              "execution stream"
            : "Vulkan MmulHelper received an invalid context-owned execution "
              "stream");
  }

  NDArray::prepareSpecialUse(outputs, inputs, beta != 0.0);

  graph::Context opContext(0);
  opContext.setInputArrays(static_cast<int>(inputs.size()), inputs.data(),
                           false);
  opContext.setOutputArrays(static_cast<int>(outputs.size()), outputs.data(),
                            false);
  if (alpha != 1.0 || beta != 0.0) {
    opContext.setTArguments(std::vector<double>{alpha, beta});
  }

  ops::matmul descriptor;
  std::string error;
  const Status status = graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), opContext, *stream, &error);
  if (status != Status::OK) {
    if (error.empty()) {
      error = "Vulkan MmulHelper descriptor execution failed";
    }
    THROW_EXCEPTION(error.c_str());
  }
#else
  (void)alpha;
  (void)beta;
  THROW_EXCEPTION(
      "Vulkan MmulHelper requires the registered Vulkan matrix-product "
      "descriptor and MLIR execution support");
#endif
}

void executeReshapedMatrixVector(NDArray* a, NDArray* x, NDArray* y,
                                 double alpha, double beta) {
  std::vector<LongType> xShape{x->lengthOf(), 1};
  std::vector<LongType> yShape{y->lengthOf(), 1};
  NDArray* xMatrix = x->reshape(x->ordering(), xShape);
  NDArray* yMatrix = y->reshape(y->ordering(), yShape, false);

  try {
    executeMatmulDescriptor(a, xMatrix, yMatrix, alpha, beta);
  } catch (...) {
    delete xMatrix;
    delete yMatrix;
    throw;
  }

  delete xMatrix;
  delete yMatrix;
}

void executeReshapedDot(NDArray* x, NDArray* y, NDArray* z, double alpha,
                        double beta) {
  std::vector<LongType> xShape{1, x->lengthOf()};
  std::vector<LongType> yShape{y->lengthOf(), 1};
  std::vector<LongType> zShape{1, 1};
  NDArray* xMatrix = x->reshape(x->ordering(), xShape);
  NDArray* yMatrix = y->reshape(y->ordering(), yShape);
  NDArray* zMatrix = z->reshape(z->ordering(), zShape, false);

  try {
    executeMatmulDescriptor(xMatrix, yMatrix, zMatrix, alpha, beta);
  } catch (...) {
    delete xMatrix;
    delete yMatrix;
    delete zMatrix;
    throw;
  }

  delete xMatrix;
  delete yMatrix;
  delete zMatrix;
}

void executeSameRankBatched(NDArray* a, NDArray* b, NDArray* c,
                            double alpha, double beta) {
  const int rank = a->rankOf();
  if (rank <= 3) {
    executeMatmulDescriptor(a, b, c, alpha, beta);
    return;
  }

  LongType batchSize = 1;
  for (int axis = 0; axis < rank - 2; ++axis) {
    batchSize *= a->sizeAt(axis);
  }

  std::vector<LongType> aShape{batchSize, a->sizeAt(-2), a->sizeAt(-1)};
  std::vector<LongType> bShape{batchSize, b->sizeAt(-2), b->sizeAt(-1)};
  std::vector<LongType> cShape{batchSize, c->sizeAt(-2), c->sizeAt(-1)};
  NDArray* aBatched = a->reshape(a->ordering(), aShape);
  NDArray* bBatched = b->reshape(b->ordering(), bShape);
  NDArray* cBatched = c->reshape(c->ordering(), cShape, false);

  try {
    executeMatmulDescriptor(aBatched, bBatched, cBatched, alpha, beta);
  } catch (...) {
    delete aBatched;
    delete bBatched;
    delete cBatched;
    throw;
  }

  delete aBatched;
  delete bBatched;
  delete cBatched;
}

void executeHigherRankLeft(NDArray* a, NDArray* b, NDArray* c,
                           double alpha, double beta) {
  std::vector<LongType> aShape{a->lengthOf() / a->sizeAt(-1),
                               a->sizeAt(-1)};
  std::vector<LongType> cShape{c->lengthOf() / c->sizeAt(-1),
                               c->sizeAt(-1)};
  NDArray* aMatrix = a->reshape(a->ordering(), aShape);
  NDArray* cMatrix = c->reshape(c->ordering(), cShape, false);

  try {
    executeMatmulDescriptor(aMatrix, b, cMatrix, alpha, beta);
  } catch (...) {
    delete aMatrix;
    delete cMatrix;
    throw;
  }

  delete aMatrix;
  delete cMatrix;
}

void executeHigherRankRight(NDArray* a, NDArray* b, NDArray* c,
                            double alpha, double beta) {
  const LongType batchSize =
      b->lengthOf() / (b->sizeAt(-2) * b->sizeAt(-1));
  std::vector<LongType> bShape{batchSize, b->sizeAt(-2), b->sizeAt(-1)};
  std::vector<LongType> cShape{batchSize, c->sizeAt(-2), c->sizeAt(-1)};
  NDArray* bBatched = b->reshape(b->ordering(), bShape);
  NDArray* cBatched = c->reshape(c->ordering(), cShape, false);

  try {
    for (LongType batch = 0; batch < batchSize; ++batch) {
      NDArray* bMatrix = (*bBatched)(batch, {0});
      NDArray* cMatrix = (*cBatched)(batch, {0});
      try {
        executeMatmulDescriptor(a, bMatrix, cMatrix, alpha, beta);
      } catch (...) {
        delete bMatrix;
        delete cMatrix;
        throw;
      }
      delete bMatrix;
      delete cMatrix;
    }
  } catch (...) {
    delete bBatched;
    delete cBatched;
    throw;
  }

  delete bBatched;
  delete cBatched;
}

}  // namespace

NDArray* MmulHelper::mmulMxM(NDArray* a, NDArray* b, NDArray* c,
                             double alpha, double beta,
                             const char outOrder) {
  if (a == nullptr || b == nullptr || a->rankOf() != 2 ||
      b->rankOf() != 2) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper matrix product requires two rank-2 inputs");
  }

  const LongType rows = a->sizeAt(0);
  const LongType inner = a->sizeAt(1);
  const LongType columns = b->sizeAt(1);
  if (b->sizeAt(0) != inner) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper matrix-product inner dimensions do not match");
  }
  if (c != nullptr &&
      (c->rankOf() != 2 || c->sizeAt(0) != rows ||
       c->sizeAt(1) != columns)) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper matrix-product output shape is invalid");
  }

  if (c == nullptr) {
    std::vector<LongType> outputShape{rows, columns};
    c = new NDArray(
        outOrder, outputShape,
        DataTypeUtils::pickPairwiseResultType(a->dataType(), b->dataType()),
        a->getContext());
  }
  if (!c->isEmpty()) {
    executeMatmulDescriptor(a, b, c, alpha, beta);
  }
  return c;
}

NDArray* MmulHelper::mmulMxV(NDArray* a, NDArray* x, NDArray* y,
                             const double alpha, const double beta,
                             const char outOrder) {
  LongType xLengthAxis = 0;
  LongType yLengthAxis = 0;
  if (a == nullptr || x == nullptr || a->rankOf() != 2 ||
      !shape::isCommonVector(x->shapeInfo(), xLengthAxis)) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper matrix-vector product requires a rank-2 matrix "
        "and a vector");
  }

  const LongType rows = a->sizeAt(0);
  const LongType columns = a->sizeAt(1);
  if (x->lengthOf() != columns) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper matrix-vector dimensions do not match");
  }
  if (y != nullptr &&
      (!shape::isCommonVector(y->shapeInfo(), yLengthAxis) ||
       y->lengthOf() != rows)) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper matrix-vector output shape is invalid");
  }

  if (y == nullptr) {
    std::vector<LongType> outputShape{rows};
    y = new NDArray(
        outOrder, outputShape,
        DataTypeUtils::pickPairwiseResultType(a->dataType(), x->dataType()),
        a->getContext());
  }
  if (!y->isEmpty()) {
    executeReshapedMatrixVector(a, x, y, alpha, beta);
  }
  return y;
}

NDArray* MmulHelper::dot(NDArray* x, NDArray* y, NDArray* z,
                         const double alpha, const double beta) {
  LongType xLengthAxis = 0;
  LongType yLengthAxis = 0;
  if (x == nullptr || y == nullptr ||
      !shape::isCommonVector(x->shapeInfo(), xLengthAxis) ||
      !shape::isCommonVector(y->shapeInfo(), yLengthAxis)) {
    THROW_EXCEPTION("Vulkan MmulHelper dot product requires two vectors");
  }
  if (x->lengthOf() != y->lengthOf()) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper dot-product vector lengths do not match");
  }
  if (z != nullptr && z->lengthOf() != 1) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper dot-product output must contain one element");
  }

  if (z == nullptr) {
    z = new NDArray(
        DataTypeUtils::pickPairwiseResultType(x->dataType(), y->dataType()),
        x->getContext());
  }
  if (!z->isEmpty()) {
    executeReshapedDot(x, y, z, alpha, beta);
  }
  return z;
}

NDArray* MmulHelper::mmulNxN(NDArray* a, NDArray* b, NDArray* c,
                             const double alpha, const double beta,
                             const char outOrder) {
  if (a == nullptr || b == nullptr || a->rankOf() < 2 ||
      b->rankOf() < 2) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper batched product requires rank-2 or higher inputs");
  }

  const int aRank = a->rankOf();
  const int bRank = b->rankOf();
  if (aRank > bRank && bRank != 2) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper can broadcast only a rank-2 right input");
  }
  if (bRank > aRank && aRank != 2) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper can broadcast only a rank-2 left input");
  }
  if (aRank == bRank) {
    for (int axis = 0; axis < aRank - 2; ++axis) {
      if (a->sizeAt(axis) != b->sizeAt(axis)) {
        THROW_EXCEPTION(
            "Vulkan MmulHelper batched dimensions do not match");
      }
    }
  }
  if (a->sizeAt(-1) != b->sizeAt(-2)) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper matrix-product inner dimensions do not match");
  }

  NDArray* shapeSource = aRank > bRank ? a : b;
  std::vector<LongType>* expectedShapePointer =
      shapeSource->getShapeAsVector();
  std::vector<LongType> expectedShape = *expectedShapePointer;
  delete expectedShapePointer;
  expectedShape[expectedShape.size() - 2] = a->sizeAt(-2);
  expectedShape[expectedShape.size() - 1] = b->sizeAt(-1);

  if (c != nullptr && !c->isSameShape(expectedShape)) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper batched-product output shape is invalid");
  }
  if (c == nullptr) {
    c = new NDArray(
        outOrder, expectedShape,
        DataTypeUtils::pickPairwiseResultType(a->dataType(), b->dataType()),
        a->getContext());
  }
  if (c->isEmpty()) {
    return c;
  }

  if (aRank == bRank) {
    executeSameRankBatched(a, b, c, alpha, beta);
  } else if (aRank > bRank) {
    executeHigherRankLeft(a, b, c, alpha, beta);
  } else {
    executeHigherRankRight(a, b, c, alpha, beta);
  }
  return c;
}

bool MmulHelper::mmulBatched(NDArray* a, NDArray* b, NDArray* c,
                             double alpha, double beta) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper batched product requires non-null arrays");
  }

  if (tryBlasStridedBatched(a, b, c, alpha, beta)) {
    return true;
  }

  mmulNxN(a, b, c, alpha, beta, c->ordering());
  return true;
}

bool MmulHelper::tryBlasStridedBatched(NDArray* a, NDArray* b, NDArray* c,
                                       double alpha, double beta,
                                       bool transA, bool transB) {
  if (a == nullptr || b == nullptr || c == nullptr || transA || transB) {
    return false;
  }

  const int rank = a->rankOf();
  if (rank != b->rankOf() || rank != c->rankOf() ||
      (rank != 2 && rank != 3)) {
    return false;
  }
  if (c->isEmpty()) {
    return true;
  }

  executeMatmulDescriptor(a, b, c, alpha, beta);
  return true;
}

void MmulHelper::setLtEpilogue(int type, const void* biasPointer,
                               int64_t biasSize) {
  (void)biasPointer;
  (void)biasSize;
  if (type != 0) {
    THROW_EXCEPTION(
        "Vulkan MmulHelper fused epilogues require descriptor emitter support");
  }
}

void MmulHelper::clearLtEpilogue() {}

// CUDA's MmulHelper owns a cache of temporary cast NDArrays because cuBLAS
// graph capture must replay the same device addresses. Vulkan dtype conversion
// belongs to the emitted pipeline, so this backend has no host-side cast arrays
// or cache indices. The shared DSP lifecycle remains complete and reports the
// actual empty state.
void MmulHelper::resetCastCacheIndices() {}

void MmulHelper::resetCastCacheIndicesTo(size_t, size_t) {}

std::pair<size_t, size_t> MmulHelper::getCastCacheHighWaterMark() {
  return {0, 0};
}

void MmulHelper::clearCastCache() {}

void MmulHelper::bumpCastCacheEpoch() {}

}  // namespace sd

#endif  // SD_VULKAN
