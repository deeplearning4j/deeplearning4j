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

// We use a simple ZIP reading approach.
// For a production implementation, miniz would be vendored here.
// For now, we implement basic ZIP central directory parsing.

#include <cstdio>
#include <cstring>
#include <fstream>

namespace sd {
namespace graph {

// ─── Minimal ZIP parsing ────────────────────────────────────────────────────
// ZIP files store a central directory at the end. We parse just enough to
// extract entries. A full implementation would use miniz.

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
    sd_printf("SdzReader: cannot open %s\n", zipPath);
    return nullptr;
  }

  auto fileSize = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> fileData(fileSize);
  file.read(reinterpret_cast<char*>(fileData.data()), fileSize);
  file.close();

  // Find End of Central Directory (search backwards for signature 0x06054b50)
  const ZipEndOfCentralDir* eocd = nullptr;
  for (size_t i = fileSize - sizeof(ZipEndOfCentralDir); i > 0; i--) {
    uint32_t sig;
    std::memcpy(&sig, fileData.data() + i, 4);
    if (sig == 0x06054b50) {
      eocd = reinterpret_cast<const ZipEndOfCentralDir*>(fileData.data() + i);
      break;
    }
  }

  if (!eocd) {
    sd_printf("SdzReader: not a valid ZIP file: %s\n", zipPath);
    return nullptr;
  }

  auto* reader = new SdzReader();

  // Parse central directory entries
  size_t cdOffset = eocd->centralDirOffset;
  for (int i = 0; i < eocd->numEntriesTotal; i++) {
    if (cdOffset + sizeof(ZipCentralDirEntry) > fileSize) break;

    const auto* entry = reinterpret_cast<const ZipCentralDirEntry*>(fileData.data() + cdOffset);
    if (entry->signature != 0x02014b50) break;

    // Get filename
    std::string filename(reinterpret_cast<const char*>(fileData.data() + cdOffset + sizeof(ZipCentralDirEntry)),
                         entry->filenameLen);

    // Only process .sdnb files and stored (uncompressed) entries
    bool isSdnb = filename.size() > 5 &&
                  filename.substr(filename.size() - 5) == ".sdnb";

    if (isSdnb && entry->compression == 0) {
      // Read from local file header
      size_t localOffset = entry->localHeaderOffset;
      if (localOffset + sizeof(ZipLocalFileHeader) < fileSize) {
        const auto* local = reinterpret_cast<const ZipLocalFileHeader*>(
            fileData.data() + localOffset);
        if (local->signature == 0x04034b50) {
          size_t dataOffset = localOffset + sizeof(ZipLocalFileHeader) +
                              local->filenameLen + local->extraLen;
          size_t dataSize = entry->uncompressedSize;
          if (dataOffset + dataSize <= fileSize) {
            std::vector<uint8_t> entryData(fileData.data() + dataOffset,
                                           fileData.data() + dataOffset + dataSize);
            reader->sdnbEntries_.emplace_back(filename, std::move(entryData));
          }
        }
      }
    } else if (isSdnb && entry->compression != 0) {
      sd_printf("SdzReader: compressed entry '%s' not supported (use STORED compression)\n",
                filename.c_str());
    }

    cdOffset += sizeof(ZipCentralDirEntry) + entry->filenameLen +
                entry->extraLen + entry->commentLen;
  }

  if (reader->sdnbEntries_.empty()) {
    sd_printf("SdzReader: no .sdnb entries found in %s\n", zipPath);
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
      sd_printf("SdzReader: failed to parse SDNB entry '%s'\n", entry.first.c_str());
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
