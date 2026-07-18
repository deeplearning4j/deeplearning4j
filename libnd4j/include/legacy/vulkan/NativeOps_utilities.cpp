/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/DataBuffer.h>
#include <execution/LaunchContext.h>
#include <helpers/shape.h>
#include <legacy/NativeOps.h>
#include <system/type_boilerplate.h>

#include <string>


namespace {

template <typename T>
void printVulkanBuffer(OpaqueDataBuffer *buffer, sd::LongType offset) {
  auto *dataBuffer = buffer->getDataBuffer();
  dataBuffer->syncToPrimary(sd::LaunchContext::defaultContext(), true);

  const sd::LongType length = dataBuffer->getNumElements();
  const auto *values = dataBuffer->template primaryAsT<T>();
  sd_printf("Data type %s: ", sd::DataTypeUtils::asString(dataBuffer->getDataType()).c_str());
  sd_printf("Vulkan device buffer: ", 0);
  for (sd::LongType index = offset; index < length; ++index) {
    sd_printf("%f ", static_cast<double>(values[index]));
  }
  sd_printf("\n", 0);
}

void requireLiveBuffer(const OpaqueDataBuffer *buffer, const char *operation) {
  if (buffer == nullptr || !buffer->hasValidDataBuffer()) {
    const std::string message =
        std::string("Vulkan ") + operation + " requires a live data buffer";
    THROW_EXCEPTION(message.c_str());
  }
}

}  // namespace

bool isBlasVersionMatches(int /*major*/, int /*minor*/, int /*build*/) {
  // Vulkan has no backend BLAS runtime ABI to compare. Match the CPU backend's
  // compatibility contract while Vulkan kernels remain independent of cuBLAS.
  return true;
}

void saveNpy(std::string fileName, const OpaqueDataBuffer *data,
             const unsigned int *shape, const unsigned int dimensions,
             std::string mode) {
  requireLiveBuffer(data, "saveNpy");
  if (shape == nullptr && dimensions != 0) {
    THROW_EXCEPTION("Vulkan saveNpy received a null shape");
  }

  auto *dataBuffer = data->getDataBuffer();
  dataBuffer->syncToPrimary(sd::LaunchContext::defaultContext(), true);
  const auto dataType = dataBuffer->getDataType();
  BUILD_SINGLE_SELECTOR(
      dataType, cnpy::npy_save,
      (fileName, dataBuffer->primary(), shape, dimensions, mode),
      SD_COMMON_TYPES);
}

void printDeviceBuffer(OpaqueDataBuffer *buffer) {
  printDeviceBuffer(buffer, 0);
}

void printDeviceBuffer(OpaqueDataBuffer *buffer, sd::LongType offset) {
  requireLiveBuffer(buffer, "printDeviceBuffer");
  const sd::LongType length = buffer->getDataBuffer()->getNumElements();
  if (offset < 0 || offset > length) {
    THROW_EXCEPTION("Vulkan printDeviceBuffer offset is outside the buffer");
  }

  const auto dataType = buffer->getDataBuffer()->getDataType();
  BUILD_SINGLE_SELECTOR(dataType, printVulkanBuffer, (buffer, offset),
                        SD_COMMON_TYPES);
}

int lengthForShapeBufferPointer(sd::Pointer buffer) {
  if (buffer == nullptr) {
    THROW_EXCEPTION("Vulkan shape buffer pointer is null");
  }
  auto *shapeBuffer = reinterpret_cast<sd::LongType *>(buffer);
  return shape::shapeInfoLength(shape::rank(shapeBuffer));
}

sd::Pointer pointerForAddress(sd::LongType address) {
  return reinterpret_cast<sd::Pointer>(address);
}


#endif  // SD_VULKAN && HAVE_VULKAN
