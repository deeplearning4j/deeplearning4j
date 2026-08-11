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

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace ::graph;

namespace sd {
namespace graph {

namespace {

constexpr size_t kLargeTensorBytes = 1024U * 1024U;

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
    : data_(nullptr), size_(0), backingOwner_(), fileBacked_(false),
      flatGraph_(nullptr), flatBufferOffset_(0), appendedDataBaseOffset_(0),
      hasAppendedDataBaseOffset_(false) {}

SdnbReader::~SdnbReader() = default;

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
  return openOwned(data, size, std::shared_ptr<void>(), false);
}

SdnbReader* SdnbReader::openOwned(const void* data, size_t size,
                                  std::shared_ptr<void> owner,
                                  bool fileBacked) {
  if (!data || size < 4) return nullptr;

  auto* reader = new SdnbReader();
  reader->data_ = static_cast<const uint8_t*>(data);
  reader->size_ = size;
  reader->backingOwner_ = std::move(owner);
  reader->fileBacked_ = fileBacked;

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
    // ManifestOffset at offset 8 (8 bytes, big-endian int64)
    // ManifestLength at offset 16 (8 bytes, big-endian int64)
    // MetadataOffset at offset 24 (8 bytes, big-endian int64) = FlatBuffer start
    auto readBigEndianInt64 = [](const uint8_t* p) -> int64_t {
      return (static_cast<int64_t>(p[0]) << 56) | (static_cast<int64_t>(p[1]) << 48) |
             (static_cast<int64_t>(p[2]) << 40) | (static_cast<int64_t>(p[3]) << 32) |
             (static_cast<int64_t>(p[4]) << 24) | (static_cast<int64_t>(p[5]) << 16) |
             (static_cast<int64_t>(p[6]) << 8)  |  static_cast<int64_t>(p[7]);
    };

    const int64_t manifestOffset = readBigEndianInt64(hdr + 8);
    const int64_t manifestLength = readBigEndianInt64(hdr + 16);
    const int64_t metadataOffset = readBigEndianInt64(hdr + 24);
    if (metadataOffset < 0 || static_cast<size_t>(metadataOffset) >= size ||
        manifestOffset < metadataOffset || manifestLength < 0 ||
        static_cast<uint64_t>(manifestOffset) > size ||
        static_cast<uint64_t>(manifestLength) >
            size - static_cast<size_t>(manifestOffset)) {
      DSP_DIAG(COMPILE,
               "SdnbReader::open: invalid SDNB offsets metadata=%lld manifest=%lld length=%lld size=%llu",
               static_cast<long long>(metadataOffset),
               static_cast<long long>(manifestOffset),
               static_cast<long long>(manifestLength),
               static_cast<unsigned long long>(size));
      delete reader;
      return nullptr;
    }

    const size_t fbOffset = static_cast<size_t>(metadataOffset);
    auto* graph = tryParseFlatGraphAt(reader->data_, size, fbOffset);
    if (graph) {
      size_t maxAppendedExtent = 0;
      bool hasAppendedData = false;
      if (graph->variables() != nullptr) {
        for (unsigned int i = 0; i < graph->variables()->size(); i++) {
          const auto* variable = graph->variables()->Get(i);
          const auto* array = variable == nullptr ? nullptr : variable->ndarray();
          if (array == nullptr || array->appendedDataLength() <= 0) continue;

          const int64_t relativeOffset = array->appendedDataOffset();
          const int64_t length = array->appendedDataLength();
          if (relativeOffset < 0 ||
              static_cast<uint64_t>(relativeOffset) >
                  std::numeric_limits<size_t>::max() ||
              static_cast<uint64_t>(length) >
                  std::numeric_limits<size_t>::max() -
                      static_cast<size_t>(relativeOffset)) {
            DSP_DIAG(COMPILE,
                     "SdnbReader::open: invalid appended range offset=%lld length=%lld",
                     static_cast<long long>(relativeOffset),
                     static_cast<long long>(length));
            delete reader;
            return nullptr;
          }
          hasAppendedData = true;
          maxAppendedExtent = std::max(
              maxAppendedExtent,
              static_cast<size_t>(relativeOffset) + static_cast<size_t>(length));
        }
      }

      if (hasAppendedData) {
        const size_t manifestStart = static_cast<size_t>(manifestOffset);
        if (maxAppendedExtent > manifestStart) {
          DSP_DIAG(COMPILE,
                   "SdnbReader::open: appended data extent %llu exceeds manifest offset %llu",
                   static_cast<unsigned long long>(maxAppendedExtent),
                   static_cast<unsigned long long>(manifestStart));
          delete reader;
          return nullptr;
        }
        const size_t appendedDataBase = manifestStart - maxAppendedExtent;
        if (appendedDataBase < fbOffset) {
          DSP_DIAG(COMPILE,
                   "SdnbReader::open: appended data base %llu precedes metadata offset %llu",
                   static_cast<unsigned long long>(appendedDataBase),
                   static_cast<unsigned long long>(fbOffset));
          delete reader;
          return nullptr;
        }
        reader->appendedDataBaseOffset_ = appendedDataBase;
        reader->hasAppendedDataBaseOffset_ = true;
      }

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
#if !defined(_WIN32)
  const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    DSP_DIAG(COMPILE, "SdnbReader::openFile: cannot open %s", path);
    return nullptr;
  }

  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    DSP_DIAG(COMPILE, "SdnbReader::openFile: cannot stat %s", path);
    return nullptr;
  }

  const size_t fileSize = static_cast<size_t>(st.st_size);
  void* mapping = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (mapping == MAP_FAILED) {
    DSP_DIAG(COMPILE, "SdnbReader::openFile: mmap failed for %s", path);
    return nullptr;
  }

  std::shared_ptr<void> owner(mapping, [fileSize](void* ptr) {
    if (ptr != nullptr) ::munmap(ptr, fileSize);
  });
  return openOwned(mapping, fileSize, std::move(owner), true);
#else
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    DSP_DIAG(COMPILE, "SdnbReader::openFile: cannot open %s", path);
    return nullptr;
  }

  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  const size_t size = static_cast<size_t>(fileSize);
  auto* buffer = new uint8_t[size];
  file.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(size));
  file.close();

  std::shared_ptr<void> owner(buffer, [](void* ptr) {
    delete[] static_cast<uint8_t*>(ptr);
  });
  return openOwned(buffer, size, std::move(owner), false);
#endif
}

int SdnbReader::numVariables() const {
  if (!flatGraph_ || !flatGraph_->variables()) return 0;
  return flatGraph_->variables()->size();
}

int SdnbReader::numNodes() const {
  if (!flatGraph_ || !flatGraph_->nodes()) return 0;
  return flatGraph_->nodes()->size();
}

NDArray* SdnbReader::loadFlatArray(const ::graph::FlatArray* fa,
                                   bool allowBorrowed,
                                   bool requireFileBacked,
                                   size_t* fileBackedBytes,
                                   size_t* heapOwnedBytes,
                                   size_t* largeHeapOwnedBytes) const {
  if (!fa) return nullptr;

  // Get shape
  auto* shapeVec = fa->shape();
  std::vector<LongType> shape;

  // Shape vector from FlatArray is the full shape info buffer
  // We need to extract rank and dimensions
  // The shape is stored as raw dimensions, not a full shapeInfo buffer
  if (shapeVec != nullptr) {
    for (unsigned int i = 0; i < shapeVec->size(); i++) {
      shape.push_back(shapeVec->Get(i));
    }
  }

  auto dtype = convertDType(fa->dtype());
  size_t elements = 1;
  for (LongType dim : shape) {
    if (dim < 0 || (dim > 0 &&
        elements > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim))) {
      return nullptr;
    }
    elements *= static_cast<size_t>(dim);
  }
  const size_t elementSize = DataTypeUtils::sizeOfElement(dtype);
  if (elementSize == 0 ||
      elements > std::numeric_limits<size_t>::max() / elementSize) {
    return nullptr;
  }
  const size_t bytesNeeded = elements * elementSize;

  auto recordHeapCopy = [&](size_t bytes) {
    if (heapOwnedBytes != nullptr) *heapOwnedBytes += bytes;
    if (largeHeapOwnedBytes != nullptr && bytes >= kLargeTensorBytes) {
      *largeHeapOwnedBytes += bytes;
    }
  };
  auto borrowedArray = [&](const uint8_t* bytes, size_t available) -> NDArray* {
    if (!allowBorrowed || !fileBacked_ || !backingOwner_ ||
        available < bytesNeeded || bytes == nullptr) {
      return nullptr;
    }
    if (fileBackedBytes != nullptr) *fileBackedBytes += bytesNeeded;
    return new NDArray(const_cast<uint8_t*>(bytes), 'c', shape, dtype,
                       LaunchContext::defaultContext(), false);
  };
  auto strictCopyRejected = [&]() {
    return requireFileBacked && bytesNeeded >= kLargeTensorBytes;
  };

  // Check for inline buffer data
  auto* buffer = fa->buffer();
  if (buffer && buffer->size() > 0) {
    if (auto* borrowed = borrowedArray(buffer->Data(), buffer->size())) {
      return borrowed;
    }
    if (strictCopyRejected()) return nullptr;
    auto* arr = new NDArray('c', shape, dtype);
    size_t bytesAvailable = buffer->size();
    size_t bytesToCopy = std::min(bytesNeeded, bytesAvailable);
    auto* destination = prepareWritablePrimary(arr, bytesNeeded);
    if (destination == nullptr) {
      delete arr;
      return nullptr;
    }
    std::memcpy(destination, buffer->Data(), bytesToCopy);
    arr->tickWriteHost();
    recordHeapCopy(bytesNeeded);
    return arr;
  }

  // Check for chunked buffer data
  auto* chunks = fa->bufferChunks();
  if (chunks && chunks->size() > 0) {
    if (strictCopyRejected()) return nullptr;
    auto* arr = new NDArray('c', shape, dtype);
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
    recordHeapCopy(bytesNeeded);
    return arr;
  }

  // Appended offsets in Java SDNB metadata are relative to the beginning of
  // the raw-data section. Offset zero is valid and identifies its first tensor.
  const int64_t appendedOffset = fa->appendedDataOffset();
  const int64_t appendedLength = fa->appendedDataLength();
  if (appendedOffset >= 0 && appendedLength > 0 &&
      hasAppendedDataBaseOffset_) {
    const size_t relativeOffset = static_cast<size_t>(appendedOffset);
    const size_t length = static_cast<size_t>(appendedLength);
    if (relativeOffset <=
            std::numeric_limits<size_t>::max() - appendedDataBaseOffset_) {
      const size_t absoluteOffset = appendedDataBaseOffset_ + relativeOffset;
      if (absoluteOffset <= size_ && length <= size_ - absoluteOffset) {
        const auto* appended = data_ + absoluteOffset;
        if (auto* borrowed = borrowedArray(appended, length)) {
          return borrowed;
        }
        if (strictCopyRejected()) return nullptr;
        auto* arr = new NDArray('c', shape, dtype);
        const size_t bytesToCopy = std::min(bytesNeeded, length);
        auto* destination = prepareWritablePrimary(arr, bytesNeeded);
        if (destination == nullptr) {
          delete arr;
          return nullptr;
        }
        std::memcpy(destination, appended, bytesToCopy);
        arr->tickWriteHost();
        recordHeapCopy(bytesNeeded);
        return arr;
      }
    }
  }

  // Empty array (no data)
  if (strictCopyRejected()) return nullptr;
  auto* empty = new NDArray('c', shape, dtype);
  recordHeapCopy(bytesNeeded);
  return empty;
}

NDArray* SdnbReader::loadVariable(const char* name) const {
  if (!flatGraph_ || !flatGraph_->variables() || !name) return nullptr;

  auto* vars = flatGraph_->variables();
  for (unsigned int i = 0; i < vars->size(); i++) {
    auto* fv = vars->Get(i);
    if (fv && fv->name() && fv->name()->str() == name) {
      return loadFlatArray(fv->ndarray(), false, false, nullptr, nullptr,
                           nullptr);
    }
  }
  return nullptr;
}

NDArray* SdnbReader::loadVariable(int index) const {
  if (!flatGraph_ || !flatGraph_->variables()) return nullptr;

  auto* vars = flatGraph_->variables();
  if (index < 0 || index >= static_cast<int>(vars->size())) return nullptr;

  auto* fv = vars->Get(index);
  return fv ? loadFlatArray(fv->ndarray(), false, false, nullptr, nullptr,
                            nullptr)
            : nullptr;
}

SdnbReader::LoadedModel SdnbReader::loadAll() const {
  return loadAllOwned(false, false);
}

SdnbReader::LoadedModel SdnbReader::loadAllOwned(
    bool inferenceOnly, bool requireFileBacked) const {
  LoadedModel model;
  if (requireFileBacked && (!fileBacked_ || !backingOwner_)) {
    DSP_DIAG(COMPILE,
             "SdnbReader::loadAllOwned: strict file-backed loading requested "
             "for non-mapped storage");
    return model;
  }

  model.graph = flatGraph_;
  if (backingOwner_) model.backingOwners.push_back(backingOwner_);

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
      auto* arr = loadFlatArray(
          fv->ndarray(), inferenceOnly, requireFileBacked,
          &model.fileBackedBytes, &model.heapOwnedBytes,
          &model.largeHeapOwnedBytes);
      if (arr) {
        model.variables[name] = arr;
      } else if (requireFileBacked && fv->ndarray() != nullptr) {
        DSP_DIAG(COMPILE,
                 "SdnbReader::loadAllOwned: strict loading rejected variable %s",
                 name.c_str());
        model.graph = nullptr;
        return model;
      }
    }
  }

  DSP_DIAG(COMPILE,
           "SdnbReader::loadAllOwned: vars=%d mappedBytes=%llu heapBytes=%llu "
           "largeHeapBytes=%llu",
           static_cast<int>(model.variables.size()),
           static_cast<unsigned long long>(model.fileBackedBytes),
           static_cast<unsigned long long>(model.heapOwnedBytes),
           static_cast<unsigned long long>(model.largeHeapOwnedBytes));
  return model;
}

}  // namespace graph
}  // namespace sd
