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

#include <graph/SdnbReader.h>
#include <graph/DspDiagnostics.h>

#include <cstdio>
#include <cstring>
#include <fstream>

using namespace ::graph;

namespace sd {
namespace graph {

namespace {

uint8_t* prepareWritablePrimary(NDArray* array, size_t bytes) {
  if (array == nullptr) return nullptr;

  auto* dataBuffer = array->getDataBuffer();
  if (dataBuffer == nullptr) return nullptr;

  // Device backends may allocate only special/device storage by default. SDNB
  // payloads are host bytes, so explicitly allocate host staging before copy.
  dataBuffer->allocatePrimary();
  auto* primary = static_cast<uint8_t*>(dataBuffer->primary());
  if (primary == nullptr) return nullptr;

  if (bytes > 0) std::memset(primary, 0, bytes);
  return primary;
}

}  // namespace

SdnbReader::SdnbReader()
    : data_(nullptr), size_(0), ownsData_(false),
      flatGraph_(nullptr), flatBufferOffset_(0) {}

SdnbReader::~SdnbReader() {
  if (ownsData_ && data_) {
    delete[] data_;
  }
}

DataType SdnbReader::convertDType(::graph::DType fbDtype) {
  switch (fbDtype) {
    case DType_FLOAT:   return FLOAT32;
    case DType_DOUBLE:  return DOUBLE;
    case DType_HALF:    return HALF;
    case DType_BFLOAT16: return BFLOAT16;
    case DType_INT8:    return INT8;
    case DType_INT16:   return INT16;
    case DType_INT32:   return INT32;
    case DType_INT64:   return INT64;
    case DType_UINT8:   return UINT8;
    case DType_UINT16:  return UINT16;
    case DType_UINT32:  return UINT32;
    case DType_UINT64:  return UINT64;
    case DType_BOOL:    return BOOL;
    default:            return FLOAT32;
  }
}

// Helper: verify + parse FlatGraph at a given offset within a buffer.
// Returns the FlatGraph pointer on success, nullptr on failure.
static const ::graph::FlatGraph* tryParseFlatGraphAt(const uint8_t* buf, size_t totalSize, size_t offset) {
  if (offset + 4 > totalSize) return nullptr;  // need at least root offset (4 bytes)
  const size_t remaining = totalSize - offset;

  // Use FlatBuffers verifier to prevent out-of-bounds reads / SIGSEGV
  flatbuffers::Verifier verifier(buf + offset, remaining);
  if (!verifier.VerifyBuffer<::graph::FlatGraph>(nullptr)) {
    return nullptr;
  }

  auto* graph = GetFlatGraph(buf + offset);
  if (graph && graph->nodes()) {
    return graph;
  }
  return nullptr;
}

SdnbReader* SdnbReader::open(const void* data, size_t size) {
  if (!data || size < 4) return nullptr;

  auto* reader = new SdnbReader();
  reader->data_ = static_cast<const uint8_t*>(data);
  reader->size_ = size;
  reader->ownsData_ = false;

  // ── SDNB header-aware parsing ──
  // Java's SameDiffSerializer writes SDNB files with a 32-byte header:
  //   MAGIC "SDNB" (4) + VERSION int32 (4) + ManifestOffset int64 (8)
  //   + ManifestLength int64 (8) + MetadataOffset int64 (8) = 32 bytes
  // The FlatBuffer (FlatGraph) starts at MetadataOffset (always 32 for v1).
  //
  // Check for the SDNB magic first and parse the header to find the
  // FlatBuffer offset. Fall back to raw FlatBuffer parse if no magic.
  static constexpr uint8_t kMagic[4] = {'S', 'D', 'N', 'B'};
  static constexpr size_t kHeaderSize = 32;

  if (size >= kHeaderSize && std::memcmp(data, kMagic, 4) == 0) {
    // SDNB format — read header fields (big-endian, matching Java's DataOutputStream)
    const uint8_t* hdr = reader->data_;

    // Version at offset 4 (4 bytes, big-endian int32)
    // ManifestOffset at offset 8 (8 bytes, big-endian int64) — unused here
    // ManifestLength at offset 16 (8 bytes, big-endian int64) — unused here
    // MetadataOffset at offset 24 (8 bytes, big-endian int64) = FlatBuffer start
    auto readBigEndianInt64 = [](const uint8_t* p) -> int64_t {
      return (static_cast<int64_t>(p[0]) << 56) | (static_cast<int64_t>(p[1]) << 48) |
             (static_cast<int64_t>(p[2]) << 40) | (static_cast<int64_t>(p[3]) << 32) |
             (static_cast<int64_t>(p[4]) << 24) | (static_cast<int64_t>(p[5]) << 16) |
             (static_cast<int64_t>(p[6]) << 8)  |  static_cast<int64_t>(p[7]);
    };

    int64_t metadataOffset = readBigEndianInt64(hdr + 24);
    if (metadataOffset < 0 || static_cast<size_t>(metadataOffset) >= size) {
      DSP_DIAG(COMPILE, "SdnbReader::open: invalid metadataOffset %lld in SDNB header",
               static_cast<long long>(metadataOffset));
      delete reader;
      return nullptr;
    }

    size_t fbOffset = static_cast<size_t>(metadataOffset);
    auto* graph = tryParseFlatGraphAt(reader->data_, size, fbOffset);
    if (graph) {
      reader->flatGraph_ = graph;
      reader->flatBufferOffset_ = static_cast<long>(fbOffset);
      return reader;
    }

    DSP_DIAG(COMPILE, "SdnbReader::open: SDNB header found but FlatGraph at offset %zu is invalid",
             fbOffset);
    delete reader;
    return nullptr;
  }

  // No SDNB magic — try as raw FlatBuffer at offset 0
  auto* graph = tryParseFlatGraphAt(reader->data_, size, 0);
  if (graph) {
    reader->flatGraph_ = graph;
    reader->flatBufferOffset_ = 0;
    return reader;
  }

  DSP_DIAG(COMPILE, "SdnbReader::open: could not find valid FlatGraph in data");
  delete reader;
  return nullptr;
}

SdnbReader* SdnbReader::openFile(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    DSP_DIAG(COMPILE, "SdnbReader::openFile: cannot open %s", path);
    return nullptr;
  }

  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  auto* buffer = new uint8_t[fileSize];
  file.read(reinterpret_cast<char*>(buffer), fileSize);
  file.close();

  auto* reader = open(buffer, fileSize);
  if (reader) {
    reader->ownsData_ = true;
  } else {
    delete[] buffer;
  }
  return reader;
}

int SdnbReader::numVariables() const {
  if (!flatGraph_ || !flatGraph_->variables()) return 0;
  return flatGraph_->variables()->size();
}

int SdnbReader::numNodes() const {
  if (!flatGraph_ || !flatGraph_->nodes()) return 0;
  return flatGraph_->nodes()->size();
}

NDArray* SdnbReader::loadFlatArray(const ::graph::FlatArray* fa) const {
  if (!fa) return nullptr;

  // Get shape
  auto* shapeVec = fa->shape();
  if (!shapeVec || shapeVec->size() == 0) return nullptr;

  int rank = 0;
  std::vector<LongType> shape;

  // Shape vector from FlatArray is the full shape info buffer
  // We need to extract rank and dimensions
  // The shape is stored as raw dimensions, not a full shapeInfo buffer
  for (unsigned int i = 0; i < shapeVec->size(); i++) {
    shape.push_back(shapeVec->Get(i));
  }

  if (shape.empty()) return nullptr;
  rank = static_cast<int>(shape.size());

  auto dtype = convertDType(fa->dtype());

  // Check for inline buffer data
  auto* buffer = fa->buffer();
  if (buffer && buffer->size() > 0) {
    auto* arr = new NDArray('c', shape, dtype);
    size_t bytesNeeded = arr->lengthOf() * arr->sizeOfT();
    size_t bytesAvailable = buffer->size();
    size_t bytesToCopy = std::min(bytesNeeded, bytesAvailable);
    auto* destination = prepareWritablePrimary(arr, bytesNeeded);
    if (destination == nullptr) {
      delete arr;
      return nullptr;
    }
    std::memcpy(destination, buffer->Data(), bytesToCopy);
    arr->tickWriteHost();
    return arr;
  }

  // Check for chunked buffer data
  auto* chunks = fa->bufferChunks();
  if (chunks && chunks->size() > 0) {
    auto* arr = new NDArray('c', shape, dtype);
    size_t bytesNeeded = arr->lengthOf() * arr->sizeOfT();
    auto* destination = prepareWritablePrimary(arr, bytesNeeded);
    if (destination == nullptr) {
      delete arr;
      return nullptr;
    }

    size_t offset = 0;
    for (unsigned int i = 0; i < chunks->size(); i++) {
      auto* chunk = chunks->Get(i);
      if (chunk && chunk->data()) {
        size_t chunkSize = chunk->data()->size();
        if (offset + chunkSize <= bytesNeeded) {
          std::memcpy(destination + offset, chunk->data()->Data(), chunkSize);
          offset += chunkSize;
        }
      }
    }
    arr->tickWriteHost();
    return arr;
  }

  // Check for appended data (new format via appendedDataOffset/appendedDataLength)
  long appendedOffset = fa->appendedDataOffset();
  long appendedLength = fa->appendedDataLength();
  if (appendedOffset > 0 && appendedLength > 0) {
    if (static_cast<size_t>(appendedOffset + appendedLength) <= size_) {
      auto* arr = new NDArray('c', shape, dtype);
      size_t bytesNeeded = arr->lengthOf() * arr->sizeOfT();
      size_t bytesToCopy = std::min(bytesNeeded, static_cast<size_t>(appendedLength));
      auto* destination = prepareWritablePrimary(arr, bytesNeeded);
      if (destination == nullptr) {
        delete arr;
        return nullptr;
      }
      std::memcpy(destination, data_ + appendedOffset, bytesToCopy);
      arr->tickWriteHost();
      return arr;
    }
  }

  // Empty array (no data)
  return new NDArray('c', shape, dtype);
}

NDArray* SdnbReader::loadVariable(const char* name) const {
  if (!flatGraph_ || !flatGraph_->variables() || !name) return nullptr;

  auto* vars = flatGraph_->variables();
  for (unsigned int i = 0; i < vars->size(); i++) {
    auto* fv = vars->Get(i);
    if (fv && fv->name() && fv->name()->str() == name) {
      return loadFlatArray(fv->ndarray());
    }
  }
  return nullptr;
}

NDArray* SdnbReader::loadVariable(int index) const {
  if (!flatGraph_ || !flatGraph_->variables()) return nullptr;

  auto* vars = flatGraph_->variables();
  if (index < 0 || index >= static_cast<int>(vars->size())) return nullptr;

  auto* fv = vars->Get(index);
  return fv ? loadFlatArray(fv->ndarray()) : nullptr;
}

SdnbReader::LoadedModel SdnbReader::loadAll() const {
  LoadedModel model;
  model.graph = flatGraph_;

  if (!flatGraph_ || !flatGraph_->variables()) return model;

  auto* vars = flatGraph_->variables();
  for (unsigned int i = 0; i < vars->size(); i++) {
    auto* fv = vars->Get(i);
    if (!fv || !fv->name()) continue;

    std::string name = fv->name()->str();
    auto vtype = fv->variabletype();

    if (vtype == VarType_PLACEHOLDER) {
      model.placeholderNames.push_back(name);
    } else if (vtype == VarType_CONSTANT || vtype == VarType_VARIABLE) {
      auto* arr = loadFlatArray(fv->ndarray());
      if (arr) {
        model.variables[name] = arr;
      }
    }
  }

  return model;
}

}  // namespace graph
}  // namespace sd
