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

#include <graph/SdzReader.h>
#include <graph/DspDiagnostics.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#ifdef HAVE_ZLIB
#include <zlib.h>
#endif

namespace {

bool endsWithIgnoreCase(const std::string& value, const char* suffix) {
  const size_t suffixLen = std::strlen(suffix);
  if (value.size() < suffixLen) return false;

  const size_t start = value.size() - suffixLen;
  for (size_t i = 0; i < suffixLen; i++) {
    const unsigned char lhs = static_cast<unsigned char>(value[start + i]);
    const unsigned char rhs = static_cast<unsigned char>(suffix[i]);
    if (std::tolower(lhs) != std::tolower(rhs)) return false;
  }
  return true;
}

template <typename T>
bool readPod(const std::vector<uint8_t>& data, size_t offset, T* out) {
  if (offset + sizeof(T) > data.size()) return false;
  std::memcpy(out, data.data() + offset, sizeof(T));
  return true;
}

#ifdef HAVE_ZLIB
bool inflateDeflateRaw(const uint8_t* src, size_t srcSize, size_t expectedSize,
                       std::vector<uint8_t>* out, std::string* errorOut) {
  if (src == nullptr || out == nullptr) {
    if (errorOut) *errorOut = "inflateDeflateRaw: null argument";
    return false;
  }

  if (srcSize > static_cast<size_t>(std::numeric_limits<uInt>::max()) ||
      expectedSize > static_cast<size_t>(std::numeric_limits<uInt>::max())) {
    if (errorOut) *errorOut = "inflateDeflateRaw: ZIP entry exceeds zlib single-call limits";
    return false;
  }

  out->assign(expectedSize, 0);

  z_stream stream{};
  stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(src));
  stream.avail_in = static_cast<uInt>(srcSize);
  stream.next_out = reinterpret_cast<Bytef*>(out->data());
  stream.avail_out = static_cast<uInt>(expectedSize);

  int rc = inflateInit2(&stream, -MAX_WBITS);  // Raw DEFLATE stream in ZIP entries.
  if (rc != Z_OK) {
    if (errorOut) *errorOut = "inflateInit2 failed";
    return false;
  }

  rc = inflate(&stream, Z_FINISH);
  const bool ok = (rc == Z_STREAM_END && stream.total_out == expectedSize);
  if (!ok && errorOut) {
    *errorOut = "inflate failed with code " + std::to_string(rc);
  }

  inflateEnd(&stream);
  return ok;
}
#endif

}  // namespace

namespace sd {
namespace graph {

#pragma pack(push, 1)
struct ZipEndOfCentralDir {
  uint32_t signature;        // 0x06054b50
  uint16_t diskNum;
  uint16_t centralDirDisk;
  uint16_t numEntriesDisk;
  uint16_t numEntriesTotal;
  uint32_t centralDirSize;
  uint32_t centralDirOffset;
  uint16_t commentLen;
};

struct ZipCentralDirEntry {
  uint32_t signature;        // 0x02014b50
  uint16_t versionMadeBy;
  uint16_t versionNeeded;
  uint16_t flags;
  uint16_t compression;
  uint16_t modTime;
  uint16_t modDate;
  uint32_t crc32;
  uint32_t compressedSize;
  uint32_t uncompressedSize;
  uint16_t filenameLen;
  uint16_t extraLen;
  uint16_t commentLen;
  uint16_t diskStart;
  uint16_t internalAttrs;
  uint32_t externalAttrs;
  uint32_t localHeaderOffset;
};

struct ZipLocalFileHeader {
  uint32_t signature;        // 0x04034b50
  uint16_t versionNeeded;
  uint16_t flags;
  uint16_t compression;
  uint16_t modTime;
  uint16_t modDate;
  uint32_t crc32;
  uint32_t compressedSize;
  uint32_t uncompressedSize;
  uint16_t filenameLen;
  uint16_t extraLen;
};
#pragma pack(pop)

SdzReader::~SdzReader() = default;

SdzReader* SdzReader::openFile(const char* zipPath) {
  std::ifstream file(zipPath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    DSP_DIAG(COMPILE, "SdzReader: cannot open %s", zipPath);
    return nullptr;
  }

  const auto fileSize = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  if (fileSize < sizeof(ZipEndOfCentralDir)) {
    DSP_DIAG(COMPILE, "SdzReader: file too small for ZIP footer: %s", zipPath);
    return nullptr;
  }

  std::vector<uint8_t> fileData(fileSize);
  file.read(reinterpret_cast<char*>(fileData.data()), fileSize);
  file.close();

  // ZIP comment can be up to 65535 bytes; EOCD must be within that trailing window.
  const size_t maxCommentLen = 0xFFFF;
  const size_t searchStart =
      (fileSize > sizeof(ZipEndOfCentralDir) + maxCommentLen)
          ? fileSize - (sizeof(ZipEndOfCentralDir) + maxCommentLen)
          : 0;

  size_t eocdOffset = std::numeric_limits<size_t>::max();
  for (size_t i = fileSize - sizeof(ZipEndOfCentralDir) + 1; i > searchStart;) {
    --i;
    uint32_t sig = 0;
    std::memcpy(&sig, fileData.data() + i, sizeof(sig));
    if (sig == 0x06054b50) {
      eocdOffset = i;
      break;
    }
  }

  if (eocdOffset == std::numeric_limits<size_t>::max()) {
    DSP_DIAG(COMPILE, "SdzReader: not a valid ZIP file: %s", zipPath);
    return nullptr;
  }

  ZipEndOfCentralDir eocd{};
  if (!readPod(fileData, eocdOffset, &eocd)) {
    DSP_DIAG(COMPILE, "SdzReader: failed to read ZIP footer: %s", zipPath);
    return nullptr;
  }

  if (eocd.centralDirOffset + eocd.centralDirSize > fileSize) {
    DSP_DIAG(COMPILE, "SdzReader: invalid central directory bounds in %s", zipPath);
    return nullptr;
  }

  auto* reader = new SdzReader();

  size_t cdOffset = eocd.centralDirOffset;
  for (int i = 0; i < eocd.numEntriesTotal; i++) {
    ZipCentralDirEntry entry{};
    if (!readPod(fileData, cdOffset, &entry)) break;
    if (entry.signature != 0x02014b50) break;

    const size_t filenameOffset = cdOffset + sizeof(ZipCentralDirEntry);
    const size_t filenameEnd = filenameOffset + entry.filenameLen;
    if (filenameEnd > fileSize) break;

    std::string filename(reinterpret_cast<const char*>(fileData.data() + filenameOffset),
                         entry.filenameLen);

    if (endsWithIgnoreCase(filename, ".sdnb")) {
      if ((entry.flags & 0x1) != 0) {
        DSP_DIAG(COMPILE, "SdzReader: encrypted entry '%s' is not supported", filename.c_str());
      } else {
        const size_t localOffset = entry.localHeaderOffset;
        ZipLocalFileHeader local{};

        if (readPod(fileData, localOffset, &local) && local.signature == 0x04034b50) {
          const size_t dataOffset =
              localOffset + sizeof(ZipLocalFileHeader) + local.filenameLen + local.extraLen;
          const size_t compressedSize = entry.compressedSize;
          const size_t uncompressedSize = entry.uncompressedSize;

          if (dataOffset <= fileSize && compressedSize <= (fileSize - dataOffset)) {
            std::vector<uint8_t> entryData;
            bool extracted = false;

            if (entry.compression == 0) {
              // STORED (no compression)
              entryData.assign(fileData.data() + dataOffset,
                               fileData.data() + dataOffset + compressedSize);
              extracted = true;
            } else if (entry.compression == 8) {
#ifdef HAVE_ZLIB
              std::string inflateError;
              extracted = inflateDeflateRaw(fileData.data() + dataOffset, compressedSize,
                                            uncompressedSize, &entryData, &inflateError);
              if (!extracted) {
                DSP_DIAG(COMPILE, "SdzReader: failed to inflate '%s': %s", filename.c_str(),
                         inflateError.c_str());
              }
#else
              DSP_DIAG(COMPILE, "SdzReader: deflated entry '%s' requires zlib support (HAVE_ZLIB)",
                       filename.c_str());
#endif
            } else {
              DSP_DIAG(COMPILE, "SdzReader: unsupported ZIP compression method %u for '%s'",
                       static_cast<unsigned>(entry.compression), filename.c_str());
            }

            if (extracted) {
              reader->sdnbEntries_.emplace_back(filename, std::move(entryData));
            }
          }
        }
      }
    }

    const size_t nextOffset = cdOffset + sizeof(ZipCentralDirEntry) + entry.filenameLen +
                              entry.extraLen + entry.commentLen;
    if (nextOffset <= cdOffset || nextOffset > fileSize) break;
    cdOffset = nextOffset;
  }

  if (reader->sdnbEntries_.empty()) {
    DSP_DIAG(COMPILE, "SdzReader: no .sdnb entries found in %s", zipPath);
    delete reader;
    return nullptr;
  }

  return reader;
}

SdnbReader::LoadedModel SdzReader::load() const {
  SdnbReader::LoadedModel combined;

  for (auto& entry : sdnbEntries_) {
    auto* sdnb = SdnbReader::open(entry.second.data(), entry.second.size());
    if (!sdnb) {
      DSP_DIAG(COMPILE, "SdzReader: failed to parse SDNB entry '%s'", entry.first.c_str());
      continue;
    }

    auto model = sdnb->loadAll();

    // Merge variables
    for (auto& pair : model.variables) {
      combined.variables[pair.first] = pair.second;
      pair.second = nullptr;  // Transfer ownership
    }

    // Merge placeholders
    for (auto& ph : model.placeholderNames) {
      combined.placeholderNames.push_back(ph);
    }

    // Use the graph from the first SDNB entry
    if (!combined.graph) {
      combined.graph = model.graph;
    }

    delete sdnb;
  }

  return combined;
}

}  // namespace graph
}  // namespace sd
