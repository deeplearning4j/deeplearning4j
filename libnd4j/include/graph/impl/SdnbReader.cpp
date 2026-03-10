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

SdnbReader* SdnbReader::open(const void* data, size_t size) {
  if (!data || size < 4) return nullptr;

  auto* reader = new SdnbReader();
  reader->data_ = static_cast<const uint8_t*>(data);
  reader->size_ = size;
  reader->ownsData_ = false;

  // Try to parse as FlatBuffer directly first
  // The SDNB format may have a header before the FlatBuffer, or the data
  // may be a raw FlatBuffer (FlatGraph).
  // Try direct FlatBuffer parse at offset 0
  reader->flatGraph_ = GetFlatGraph(data);
  if (reader->flatGraph_ && reader->flatGraph_->nodes()) {
    reader->flatBufferOffset_ = 0;
    return reader;
  }

  // Try scanning for FlatBuffer start
  // SDNB header format varies — try common offsets
  // The Java serializer writes the FlatBuffer at a known offset stored in the file
  for (size_t offset = 0; offset < std::min(size, (size_t)4096); offset += 4) {
    auto* candidate = GetFlatGraph(reader->data_ + offset);
    if (candidate && candidate->nodes() && candidate->nodes()->size() > 0) {
      reader->flatGraph_ = candidate;
      reader->flatBufferOffset_ = static_cast<long>(offset);
      return reader;
    }
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
    std::memcpy(arr->buffer(), buffer->Data(), bytesToCopy);
    return arr;
  }

  // Check for chunked buffer data
  auto* chunks = fa->bufferChunks();
  if (chunks && chunks->size() > 0) {
    auto* arr = new NDArray('c', shape, dtype);
    size_t offset = 0;
    for (unsigned int i = 0; i < chunks->size(); i++) {
      auto* chunk = chunks->Get(i);
      if (chunk && chunk->data()) {
        size_t chunkSize = chunk->data()->size();
        if (offset + chunkSize <= arr->lengthOf() * arr->sizeOfT()) {
          std::memcpy(static_cast<uint8_t*>(arr->buffer()) + offset,
                      chunk->data()->Data(), chunkSize);
          offset += chunkSize;
        }
      }
    }
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
      std::memcpy(arr->buffer(), data_ + appendedOffset, bytesToCopy);
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
