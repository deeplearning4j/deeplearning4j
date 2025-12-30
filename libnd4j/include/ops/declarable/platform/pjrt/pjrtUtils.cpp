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

#include "pjrtUtils.h"

#include <sstream>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Utility function implementations
//////////////////////////////////////////////////////////////////////////

int pjrtBufferType(DataType dtype) {
  // Map ND4J DataType to PJRT/XLA PrimitiveType values
  // These values correspond to xla::PrimitiveType enum
  switch (dtype) {
    case DataType::FLOAT32:
      return 11;  // F32
    case DataType::FLOAT64:
      return 12;  // F64
    case DataType::FLOAT16:
      return 10;  // F16
    case DataType::BFLOAT16:
      return 16;  // BF16
    case DataType::INT8:
      return 2;   // S8
    case DataType::INT16:
      return 4;   // S16
    case DataType::INT32:
      return 6;   // S32
    case DataType::INT64:
      return 8;   // S64
    case DataType::UINT8:
      return 3;   // U8
    case DataType::UINT16:
      return 5;   // U16
    case DataType::UINT32:
      return 7;   // U32
    case DataType::UINT64:
      return 9;   // U64
    case DataType::BOOL:
      return 1;   // PRED
    default:
      return -1;  // Invalid
  }
}

DataType fromPjrtBufferType(int pjrtType) {
  switch (pjrtType) {
    case 11:
      return DataType::FLOAT32;
    case 12:
      return DataType::FLOAT64;
    case 10:
      return DataType::FLOAT16;
    case 16:
      return DataType::BFLOAT16;
    case 2:
      return DataType::INT8;
    case 4:
      return DataType::INT16;
    case 6:
      return DataType::INT32;
    case 8:
      return DataType::INT64;
    case 3:
      return DataType::UINT8;
    case 5:
      return DataType::UINT16;
    case 7:
      return DataType::UINT32;
    case 9:
      return DataType::UINT64;
    case 1:
      return DataType::BOOL;
    default:
      return DataType::UNKNOWN;
  }
}

bool isTpuSupportedType(DataType dtype) {
  // TPU v4/v5 support these types natively
  switch (dtype) {
    case DataType::FLOAT32:
    case DataType::BFLOAT16:
    case DataType::FLOAT16:
    case DataType::INT32:
    case DataType::INT8:
    case DataType::BOOL:
      return true;
    default:
      return false;
  }
}

std::string getHLOTypeString(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return "f32";
    case DataType::FLOAT64:
      return "f64";
    case DataType::FLOAT16:
      return "f16";
    case DataType::BFLOAT16:
      return "bf16";
    case DataType::INT8:
      return "s8";
    case DataType::INT16:
      return "s16";
    case DataType::INT32:
      return "s32";
    case DataType::INT64:
      return "s64";
    case DataType::UINT8:
      return "u8";
    case DataType::UINT16:
      return "u16";
    case DataType::UINT32:
      return "u32";
    case DataType::UINT64:
      return "u64";
    case DataType::BOOL:
      return "pred";
    default:
      return "f32";
  }
}

std::string shapeToString(const std::vector<sd::LongType>& shape) {
  std::ostringstream oss;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ",";
    oss << shape[i];
  }
  return oss.str();
}

std::string buildMatmulHLO(const std::vector<sd::LongType>& aShape,
                           const std::vector<sd::LongType>& bShape,
                           DataType dtype, bool transA, bool transB) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  // Determine output shape based on transpose flags
  sd::LongType m = transA ? aShape[1] : aShape[0];
  sd::LongType k = transA ? aShape[0] : aShape[1];
  sd::LongType n = transB ? bShape[0] : bShape[1];

  hlo << "HloModule matmul_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  a = " << typeStr << "[" << shapeToString(aShape) << "] parameter(0)\n";
  hlo << "  b = " << typeStr << "[" << shapeToString(bShape) << "] parameter(1)\n";

  // Apply transposes if needed
  if (transA) {
    hlo << "  a_t = " << typeStr << "[" << aShape[1] << "," << aShape[0] << "] transpose(a), dimensions={1,0}\n";
  }
  if (transB) {
    hlo << "  b_t = " << typeStr << "[" << bShape[1] << "," << bShape[0] << "] transpose(b), dimensions={1,0}\n";
  }

  std::string aRef = transA ? "a_t" : "a";
  std::string bRef = transB ? "b_t" : "b";

  hlo << "  ROOT result = " << typeStr << "[" << m << "," << n << "] dot(" << aRef << ", " << bRef;
  hlo << "), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n";
  hlo << "}\n";

  return hlo.str();
}

std::string buildAddHLO(const std::vector<sd::LongType>& shape, DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);
  std::string shapeStr = shapeToString(shape);

  hlo << "HloModule add_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  a = " << typeStr << "[" << shapeStr << "] parameter(0)\n";
  hlo << "  b = " << typeStr << "[" << shapeStr << "] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[" << shapeStr << "] add(a, b)\n";
  hlo << "}\n";

  return hlo.str();
}

std::string buildConv2dHLO(const std::vector<sd::LongType>& inputShape,
                           const std::vector<sd::LongType>& filterShape,
                           const std::vector<int>& strides,
                           const std::vector<int>& padding,
                           DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  // inputShape: [N, H, W, C] (NHWC format)
  // filterShape: [kH, kW, C, F] (filter format)
  sd::LongType batch = inputShape[0];
  sd::LongType inH = inputShape[1];
  sd::LongType inW = inputShape[2];
  sd::LongType inC = inputShape[3];
  sd::LongType kH = filterShape[0];
  sd::LongType kW = filterShape[1];
  sd::LongType outF = filterShape[3];

  // Calculate output dimensions
  sd::LongType outH = (inH + 2 * padding[0] - kH) / strides[0] + 1;
  sd::LongType outW = (inW + 2 * padding[1] - kW) / strides[1] + 1;

  hlo << "HloModule conv2d_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[" << batch << "," << inH << "," << inW << "," << inC << "] parameter(0)\n";
  hlo << "  filter = " << typeStr << "[" << kH << "," << kW << "," << inC << "," << outF << "] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[" << batch << "," << outH << "," << outW << "," << outF << "] ";
  hlo << "convolution(input, filter), ";
  hlo << "window={size=" << kH << "x" << kW << " stride=" << strides[0] << "x" << strides[1];
  hlo << " pad=" << padding[0] << "_" << padding[0] << "x" << padding[1] << "_" << padding[1] << "}, ";
  hlo << "dim_labels=b01f_01io->b01f\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// PJRT RAII wrapper implementations (when PJRT is available)
//////////////////////////////////////////////////////////////////////////

#if defined(HAVE_PJRT) && HAVE_PJRT

// Global PJRT client instance
static std::unique_ptr<PjrtClient> globalPjrtClient;
static std::once_flag globalClientInitFlag;

PjrtClient& getGlobalPjrtClient() {
  std::call_once(globalClientInitFlag, []() {
    globalPjrtClient = std::make_unique<PjrtClient>();
  });
  return *globalPjrtClient;
}

//////////////////////////////////////////////////////////////////////////
// PjrtClient implementation
//////////////////////////////////////////////////////////////////////////

PjrtClient::PjrtClient() {
  // Get the PJRT API - this is typically loaded from a shared library
  // For TPU, this would be libtpu.so or similar
  // The actual initialization would use PJRT_Api_Create or similar

  // Placeholder - actual implementation depends on how PJRT is loaded
  // This would typically involve:
  // 1. Loading the PJRT plugin shared library
  // 2. Getting the PJRT_Api struct
  // 3. Creating a client

  // For now, we leave this as a stub that will be filled in
  // when the actual PJRT library is available
  api_ = nullptr;
  client_ = nullptr;
}

PjrtClient::~PjrtClient() {
  destroy();
}

PjrtClient::PjrtClient(PjrtClient&& other) noexcept
    : client_(other.client_), api_(other.api_) {
  other.client_ = nullptr;
  other.api_ = nullptr;
}

PjrtClient& PjrtClient::operator=(PjrtClient&& other) noexcept {
  if (this != &other) {
    destroy();
    client_ = other.client_;
    api_ = other.api_;
    other.client_ = nullptr;
    other.api_ = nullptr;
  }
  return *this;
}

void PjrtClient::destroy() {
  if (client_ && api_) {
    PJRT_Client_Destroy_Args args;
    args.struct_size = sizeof(PJRT_Client_Destroy_Args);
    args.client = client_;
    api_->PJRT_Client_Destroy(&args);
  }
  client_ = nullptr;
}

int PjrtClient::deviceCount() const {
  if (!client_ || !api_) return 0;

  PJRT_Client_Devices_Args args;
  args.struct_size = sizeof(PJRT_Client_Devices_Args);
  args.client = client_;

  PJRT_Error* error = api_->PJRT_Client_Devices(&args);
  if (error) {
    // Clean up error and return 0
    PJRT_Error_Destroy_Args destroy_args;
    destroy_args.struct_size = sizeof(PJRT_Error_Destroy_Args);
    destroy_args.error = error;
    api_->PJRT_Error_Destroy(&destroy_args);
    return 0;
  }

  return static_cast<int>(args.num_devices);
}

//////////////////////////////////////////////////////////////////////////
// PjrtBuffer implementation
//////////////////////////////////////////////////////////////////////////

PjrtBuffer::PjrtBuffer(PjrtClient& client, const NDArray* arr) {
  api_ = client.api();
  // Placeholder - actual buffer creation would involve:
  // 1. Getting device from client
  // 2. Creating buffer from host data
  // 3. Transferring data to device
  buffer_ = nullptr;
}

PjrtBuffer::~PjrtBuffer() {
  destroy();
}

PjrtBuffer::PjrtBuffer(PjrtBuffer&& other) noexcept
    : buffer_(other.buffer_), api_(other.api_) {
  other.buffer_ = nullptr;
  other.api_ = nullptr;
}

PjrtBuffer& PjrtBuffer::operator=(PjrtBuffer&& other) noexcept {
  if (this != &other) {
    destroy();
    buffer_ = other.buffer_;
    api_ = other.api_;
    other.buffer_ = nullptr;
    other.api_ = nullptr;
  }
  return *this;
}

void PjrtBuffer::destroy() {
  if (buffer_ && api_) {
    PJRT_Buffer_Destroy_Args args;
    args.struct_size = sizeof(PJRT_Buffer_Destroy_Args);
    args.buffer = buffer_;
    api_->PJRT_Buffer_Destroy(&args);
  }
  buffer_ = nullptr;
}

void PjrtBuffer::toHost(NDArray* arr) {
  if (!buffer_ || !api_ || !arr) return;
  // Placeholder - actual implementation would copy data from device to host
}

//////////////////////////////////////////////////////////////////////////
// PjrtLoadedExecutable implementation
//////////////////////////////////////////////////////////////////////////

PjrtLoadedExecutable::PjrtLoadedExecutable(PjrtClient& client, const char* hloModule, size_t size) {
  api_ = client.api();
  client_ = client.get();
  // Placeholder - actual implementation would:
  // 1. Parse HLO module
  // 2. Compile for TPU
  // 3. Load executable
  executable_ = nullptr;
}

PjrtLoadedExecutable::~PjrtLoadedExecutable() {
  destroy();
}

PjrtLoadedExecutable::PjrtLoadedExecutable(PjrtLoadedExecutable&& other) noexcept
    : executable_(other.executable_), api_(other.api_), client_(other.client_) {
  other.executable_ = nullptr;
  other.api_ = nullptr;
  other.client_ = nullptr;
}

PjrtLoadedExecutable& PjrtLoadedExecutable::operator=(PjrtLoadedExecutable&& other) noexcept {
  if (this != &other) {
    destroy();
    executable_ = other.executable_;
    api_ = other.api_;
    client_ = other.client_;
    other.executable_ = nullptr;
    other.api_ = nullptr;
    other.client_ = nullptr;
  }
  return *this;
}

void PjrtLoadedExecutable::destroy() {
  if (executable_ && api_) {
    PJRT_LoadedExecutable_Destroy_Args args;
    args.struct_size = sizeof(PJRT_LoadedExecutable_Destroy_Args);
    args.executable = executable_;
    api_->PJRT_LoadedExecutable_Destroy(&args);
  }
  executable_ = nullptr;
}

void PjrtLoadedExecutable::execute(const std::vector<PjrtBuffer*>& inputs,
                                    std::vector<PjrtBuffer*>& outputs) {
  if (!executable_ || !api_) return;
  // Placeholder - actual implementation would execute the compiled program
}

//////////////////////////////////////////////////////////////////////////
// HLOExecutableCache implementation
//////////////////////////////////////////////////////////////////////////

HLOExecutableCache& HLOExecutableCache::getInstance() {
  static HLOExecutableCache instance;
  return instance;
}

std::shared_ptr<PjrtLoadedExecutable> HLOExecutableCache::getOrCompile(
    PjrtClient& client, const std::string& hloKey, const std::string& hloModule) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = cache_.find(hloKey);
  if (it != cache_.end()) {
    return it->second;
  }

  // Compile new executable
  auto executable = std::make_shared<PjrtLoadedExecutable>(
      client, hloModule.c_str(), hloModule.size());

  // Evict oldest if cache is full (simple LRU would need more tracking)
  if (cache_.size() >= maxSize_) {
    cache_.erase(cache_.begin());
  }

  cache_[hloKey] = executable;
  return executable;
}

void HLOExecutableCache::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
}

#endif  // HAVE_PJRT

}  // namespace platforms
}  // namespace ops
}  // namespace sd
