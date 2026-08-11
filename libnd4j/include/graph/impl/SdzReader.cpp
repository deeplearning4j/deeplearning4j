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
#include <memory>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef HAVE_ZLIB
#include <zlib.h>
#endif

// std::filesystem availability — same detection as DspRuntimeC.cpp / ReplayCacheManager.cpp.
#if defined(SD_FILESYSTEM_AVAILABLE)
#define SDZ_HAS_FILESYSTEM 1
#elif defined(__has_include)
#  if __has_include(<filesystem>) && __cplusplus >= 201703L
#    if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 9
#      define SDZ_HAS_FILESYSTEM 0
#    elif defined(__APPLE__)
#      if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500
#        define SDZ_HAS_FILESYSTEM 1
#      else
#        define SDZ_HAS_FILESYSTEM 0
#      endif
#    else
#      define SDZ_HAS_FILESYSTEM 1
#    endif
#  else
#    define SDZ_HAS_FILESYSTEM 0
#  endif
#else
#define SDZ_HAS_FILESYSTEM 0
#endif

#if SDZ_HAS_FILESYSTEM
#include <filesystem>
#endif

namespace {

struct ByteView {
  const uint8_t* bytes = nullptr;
  size_t length = 0;
  const uint8_t* data() const { return bytes; }
  size_t size() const { return length; }
};

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

// The SDNB binary format starts with the 4-byte magic "SDNB".
// Java's SDZSerializer may write entries without the .sdnb extension (e.g. "model")
// for single-shard saves. This helper checks raw data for the magic prefix.
static constexpr uint8_t kSdnbMagic[4] = {'S', 'D', 'N', 'B'};

bool hasSdnbMagic(const uint8_t* data, size_t size) {
  return size >= 4 && std::memcmp(data, kSdnbMagic, 4) == 0;
}

template <typename Container, typename T>
bool readPod(const Container& data, size_t offset, T* out) {
  if (offset + sizeof(T) > data.size()) return false;
  std::memcpy(out, data.data() + offset, sizeof(T));
  return true;
}

// Parse the ZIP64 extended-information extra field (header ID 0x0001) of a
// central-directory entry. Per APPNOTE 4.5.3, only the fields saturated at
// 0xFFFFFFFF (or 0xFFFF for diskStart) in the fixed record are present, in
// the fixed order: uncompressed size, compressed size, local header offset.
template <typename Container>
bool parseZip64ExtraField(const Container& data, size_t extraOffset, uint16_t extraLen,
                          uint32_t rawUncompressed, uint32_t rawCompressed, uint32_t rawLocalOffset,
                          uint64_t* uncompressedSize, uint64_t* compressedSize,
                          uint64_t* localHeaderOffset) {
  size_t pos = extraOffset;
  const size_t end = extraOffset + extraLen;
  if (end > data.size()) return false;

  while (pos + 4 <= end) {
    uint16_t fieldId = 0;
    uint16_t fieldSize = 0;
    std::memcpy(&fieldId, data.data() + pos, sizeof(fieldId));
    std::memcpy(&fieldSize, data.data() + pos + 2, sizeof(fieldSize));
    pos += 4;
    if (pos + fieldSize > end) return false;

    if (fieldId == 0x0001) {
      size_t p = pos;
      const size_t fieldEnd = pos + fieldSize;
      if (rawUncompressed == 0xFFFFFFFFu) {
        if (p + 8 > fieldEnd) return false;
        std::memcpy(uncompressedSize, data.data() + p, 8);
        p += 8;
      }
      if (rawCompressed == 0xFFFFFFFFu) {
        if (p + 8 > fieldEnd) return false;
        std::memcpy(compressedSize, data.data() + p, 8);
        p += 8;
      }
      if (rawLocalOffset == 0xFFFFFFFFu) {
        if (p + 8 > fieldEnd) return false;
        std::memcpy(localHeaderOffset, data.data() + p, 8);
        p += 8;
      }
      return true;
    }
    pos += fieldSize;
  }
  return false;
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

// ZIP64 support: archives over 4GB (or >65535 entries) saturate the classic
// EOCD/central-directory fields at 0xFFFF/0xFFFFFFFF and carry the real
// 64-bit values in these records. Java's SDZSerializer (java.util.zip)
// produces ZIP64 automatically for large models.
struct Zip64EndOfCentralDirLocator {
  uint32_t signature;        // 0x07064b50
  uint32_t zip64EocdDisk;
  uint64_t zip64EocdOffset;
  uint32_t totalDisks;
};

struct Zip64EndOfCentralDir {
  uint32_t signature;        // 0x06064b50
  uint64_t recordSize;
  uint16_t versionMadeBy;
  uint16_t versionNeeded;
  uint32_t diskNum;
  uint32_t centralDirDisk;
  uint64_t numEntriesDisk;
  uint64_t numEntriesTotal;
  uint64_t centralDirSize;
  uint64_t centralDirOffset;
};
#pragma pack(pop)

SdzReader::~SdzReader() = default;

SdzReader* SdzReader::openFile(const char* zipPath) {
#if !defined(_WIN32)
  const int fd = ::open(zipPath, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    DSP_DIAG(COMPILE, "SdzReader: cannot open %s", zipPath);
    return nullptr;
  }
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    DSP_DIAG(COMPILE, "SdzReader: cannot stat %s", zipPath);
    return nullptr;
  }
  const size_t fileSize = static_cast<size_t>(st.st_size);
  void* mapping = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (mapping == MAP_FAILED) {
    DSP_DIAG(COMPILE, "SdzReader: mmap failed for %s", zipPath);
    return nullptr;
  }
  std::shared_ptr<void> fileOwner(mapping, [fileSize](void* ptr) {
    if (ptr != nullptr) ::munmap(ptr, fileSize);
  });
  ByteView fileData{static_cast<const uint8_t*>(mapping), fileSize};
#else
  std::ifstream file(zipPath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    DSP_DIAG(COMPILE, "SdzReader: cannot open %s", zipPath);
    return nullptr;
  }

  const size_t fileSize = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  if (fileSize < sizeof(ZipEndOfCentralDir)) {
    DSP_DIAG(COMPILE, "SdzReader: file too small for ZIP footer: %s", zipPath);
    return nullptr;
  }

  auto fileBytes = std::make_shared<std::vector<uint8_t>>(fileSize);
  file.read(reinterpret_cast<char*>(fileBytes->data()),
            static_cast<std::streamsize>(fileSize));
  file.close();
  std::shared_ptr<void> fileOwner = fileBytes;
  ByteView fileData{fileBytes->data(), fileBytes->size()};
#endif

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

  // ZIP64: when the classic EOCD saturates (0xFFFF entries / 0xFFFFFFFF
  // offset or size), the real values live in the ZIP64 EOCD record, found
  // via the locator that sits immediately before the classic EOCD.
  uint64_t numEntriesTotal = eocd.numEntriesTotal;
  uint64_t centralDirSize = eocd.centralDirSize;
  uint64_t centralDirOffset = eocd.centralDirOffset;

  const bool eocdSaturated = eocd.numEntriesTotal == 0xFFFFu ||
                             eocd.centralDirSize == 0xFFFFFFFFu ||
                             eocd.centralDirOffset == 0xFFFFFFFFu;
  if (eocdSaturated || eocdOffset >= sizeof(Zip64EndOfCentralDirLocator)) {
    const size_t locatorOffset = eocdOffset - sizeof(Zip64EndOfCentralDirLocator);
    Zip64EndOfCentralDirLocator locator{};
    if (eocdOffset >= sizeof(Zip64EndOfCentralDirLocator) &&
        readPod(fileData, locatorOffset, &locator) && locator.signature == 0x07064b50) {
      Zip64EndOfCentralDir eocd64{};
      if (locator.zip64EocdOffset < fileSize &&
          readPod(fileData, static_cast<size_t>(locator.zip64EocdOffset), &eocd64) &&
          eocd64.signature == 0x06064b50) {
        numEntriesTotal = eocd64.numEntriesTotal;
        centralDirSize = eocd64.centralDirSize;
        centralDirOffset = eocd64.centralDirOffset;
        DSP_DIAG(COMPILE, "SdzReader: ZIP64 archive (%llu entries) in %s",
                 static_cast<unsigned long long>(numEntriesTotal), zipPath);
      } else if (eocdSaturated) {
        DSP_DIAG(COMPILE, "SdzReader: saturated EOCD but invalid ZIP64 record in %s", zipPath);
        return nullptr;
      }
    } else if (eocdSaturated) {
      DSP_DIAG(COMPILE, "SdzReader: saturated EOCD but no ZIP64 locator in %s", zipPath);
      return nullptr;
    }
  }

  if (centralDirOffset + centralDirSize > fileSize) {
    DSP_DIAG(COMPILE, "SdzReader: invalid central directory bounds in %s", zipPath);
    return nullptr;
  }

  auto* reader = new SdzReader();

  size_t cdOffset = static_cast<size_t>(centralDirOffset);
  for (uint64_t i = 0; i < numEntriesTotal; i++) {
    ZipCentralDirEntry entry{};
    if (!readPod(fileData, cdOffset, &entry)) break;
    if (entry.signature != 0x02014b50) break;

    const size_t filenameOffset = cdOffset + sizeof(ZipCentralDirEntry);
    const size_t filenameEnd = filenameOffset + entry.filenameLen;
    if (filenameEnd > fileSize) break;

    std::string filename(reinterpret_cast<const char*>(fileData.data() + filenameOffset),
                         entry.filenameLen);

    // Accept entries with .sdnb extension OR entries whose raw data starts with "SDNB" magic.
    // Java's SDZSerializer writes single-shard models with entry name "model" (no extension)
    // but the binary content still has the SDNB format header.
    bool nameMatchesSdnb = endsWithIgnoreCase(filename, ".sdnb");
    // Skip directory entries and encrypted entries early
    if ((entry.flags & 0x1) != 0) {
      if (nameMatchesSdnb) {
        DSP_DIAG(COMPILE, "SdzReader: encrypted entry '%s' is not supported", filename.c_str());
      }
    } else {
      // ZIP64: saturated per-entry fields carry their real 64-bit values in
      // the 0x0001 extended-information extra field.
      uint64_t entryUncompressed = entry.uncompressedSize;
      uint64_t entryCompressed = entry.compressedSize;
      uint64_t entryLocalOffset = entry.localHeaderOffset;
      if (entry.uncompressedSize == 0xFFFFFFFFu || entry.compressedSize == 0xFFFFFFFFu ||
          entry.localHeaderOffset == 0xFFFFFFFFu) {
        if (!parseZip64ExtraField(fileData, filenameEnd, entry.extraLen,
                                  entry.uncompressedSize, entry.compressedSize,
                                  entry.localHeaderOffset, &entryUncompressed,
                                  &entryCompressed, &entryLocalOffset)) {
          DSP_DIAG(COMPILE, "SdzReader: saturated entry '%s' missing ZIP64 extra field",
                   filename.c_str());
          const size_t skipOffset = cdOffset + sizeof(ZipCentralDirEntry) + entry.filenameLen +
                                    entry.extraLen + entry.commentLen;
          if (skipOffset <= cdOffset || skipOffset > fileSize) break;
          cdOffset = skipOffset;
          continue;
        }
      }

      const size_t localOffset = static_cast<size_t>(entryLocalOffset);
      ZipLocalFileHeader local{};

      if (readPod(fileData, localOffset, &local) && local.signature == 0x04034b50) {
        const size_t dataOffset =
            localOffset + sizeof(ZipLocalFileHeader) + local.filenameLen + local.extraLen;
        const size_t compressedSize = static_cast<size_t>(entryCompressed);
        const size_t uncompressedSize = static_cast<size_t>(entryUncompressed);

        if (dataOffset <= fileSize && compressedSize <= (fileSize - dataOffset)) {
          // For STORED entries, check the SDNB magic in-place without extraction.
          // For DEFLATE entries, we must decompress first before checking magic.
          bool magicMatchesSdnb = false;
          if (!nameMatchesSdnb && entry.compression == 0 && compressedSize >= 4) {
            magicMatchesSdnb = hasSdnbMagic(fileData.data() + dataOffset, compressedSize);
          }

          if (nameMatchesSdnb || magicMatchesSdnb || (!nameMatchesSdnb && entry.compression != 0)) {
            SdnbEntry indexedEntry;
            indexedEntry.filename = filename;
            indexedEntry.compression = entry.compression;
            bool extracted = false;

            if (entry.compression == 0) {
              // STORED entries borrow directly from the read-only archive map.
              indexedEntry.data = fileData.data() + dataOffset;
              indexedEntry.size = compressedSize;
              indexedEntry.fileBacked = true;
              indexedEntry.backingOwner = fileOwner;
              extracted = true;
            } else if (entry.compression == 8) {
              // Index compressed model shards without inflating them. Strict
              // callers can reject them before allocating uncompressed bytes;
              // compatibility callers inflate lazily in load().
              indexedEntry.data = fileData.data() + dataOffset;
              indexedEntry.size = compressedSize;
              indexedEntry.uncompressedSize = uncompressedSize;
              indexedEntry.backingOwner = fileOwner;
              extracted = nameMatchesSdnb || filename == "model";
            } else {
              DSP_DIAG(COMPILE, "SdzReader: unsupported ZIP compression method %u for '%s'",
                       static_cast<unsigned>(entry.compression), filename.c_str());
            }

            if (extracted) {
              // Final check: verify SDNB magic if we weren't sure by filename
              if (nameMatchesSdnb || filename == "model" ||
                  hasSdnbMagic(indexedEntry.data, indexedEntry.size)) {
                reader->sdnbEntries_.push_back(std::move(indexedEntry));
                auto& stored = reader->sdnbEntries_.back();
              }
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

bool SdzReader::extractArchive(const char* zipPath, const char* destDir, std::string* errorOut) {
#if !SDZ_HAS_FILESYSTEM
  (void)zipPath;
  (void)destDir;
  if (errorOut) *errorOut = "SdzReader::extractArchive requires std::filesystem support";
  return false;
#else
  auto fail = [&](const std::string& msg) {
    if (errorOut) *errorOut = msg;
    return false;
  };

  std::ifstream file(zipPath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) return fail(std::string("cannot open archive: ") + zipPath);

  const auto fileSize = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  if (fileSize < sizeof(ZipEndOfCentralDir)) return fail("archive too small for ZIP footer");

  std::vector<uint8_t> fileData(fileSize);
  file.read(reinterpret_cast<char*>(fileData.data()), fileSize);
  file.close();

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
  if (eocdOffset == std::numeric_limits<size_t>::max()) return fail("not a valid ZIP archive");

  ZipEndOfCentralDir eocd{};
  if (!readPod(fileData, eocdOffset, &eocd)) return fail("failed to read ZIP footer");

  uint64_t numEntriesTotal = eocd.numEntriesTotal;
  uint64_t centralDirOffset = eocd.centralDirOffset;
  uint64_t centralDirSize = eocd.centralDirSize;
  const bool eocdSaturated = eocd.numEntriesTotal == 0xFFFFu ||
                             eocd.centralDirSize == 0xFFFFFFFFu ||
                             eocd.centralDirOffset == 0xFFFFFFFFu;
  if (eocdOffset >= sizeof(Zip64EndOfCentralDirLocator)) {
    Zip64EndOfCentralDirLocator locator{};
    if (readPod(fileData, eocdOffset - sizeof(Zip64EndOfCentralDirLocator), &locator) &&
        locator.signature == 0x07064b50) {
      Zip64EndOfCentralDir eocd64{};
      if (locator.zip64EocdOffset < fileSize &&
          readPod(fileData, static_cast<size_t>(locator.zip64EocdOffset), &eocd64) &&
          eocd64.signature == 0x06064b50) {
        numEntriesTotal = eocd64.numEntriesTotal;
        centralDirOffset = eocd64.centralDirOffset;
        centralDirSize = eocd64.centralDirSize;
      } else if (eocdSaturated) {
        return fail("saturated EOCD but invalid ZIP64 record");
      }
    } else if (eocdSaturated) {
      return fail("saturated EOCD but no ZIP64 locator");
    }
  } else if (eocdSaturated) {
    return fail("saturated EOCD but no room for a ZIP64 locator");
  }
  if (centralDirOffset + centralDirSize > fileSize) return fail("invalid central directory bounds");

  std::error_code ec;
  const std::filesystem::path destRoot(destDir);
  std::filesystem::create_directories(destRoot, ec);
  if (ec) return fail("cannot create destination directory: " + destRoot.string());

  size_t cdOffset = static_cast<size_t>(centralDirOffset);
  for (uint64_t i = 0; i < numEntriesTotal; i++) {
    ZipCentralDirEntry entry{};
    if (!readPod(fileData, cdOffset, &entry)) break;
    if (entry.signature != 0x02014b50) break;

    const size_t filenameOffset = cdOffset + sizeof(ZipCentralDirEntry);
    const size_t filenameEnd = filenameOffset + entry.filenameLen;
    if (filenameEnd > fileSize) break;
    std::string filename(reinterpret_cast<const char*>(fileData.data() + filenameOffset),
                         entry.filenameLen);

    const size_t nextOffset = cdOffset + sizeof(ZipCentralDirEntry) + entry.filenameLen +
                              entry.extraLen + entry.commentLen;

    // Zip-slip guard: reject absolute paths, drive letters, and traversal.
    const bool unsafe = filename.empty() || filename.front() == '/' || filename.front() == '\\' ||
                        filename.find("..") != std::string::npos ||
                        filename.find(':') != std::string::npos;
    if (unsafe || (entry.flags & 0x1) != 0) {
      if ((entry.flags & 0x1) != 0) return fail("encrypted entry not supported: " + filename);
      if (unsafe && !filename.empty()) return fail("unsafe entry name rejected: " + filename);
      if (nextOffset <= cdOffset || nextOffset > fileSize) break;
      cdOffset = nextOffset;
      continue;
    }

    uint64_t entryUncompressed = entry.uncompressedSize;
    uint64_t entryCompressed = entry.compressedSize;
    uint64_t entryLocalOffset = entry.localHeaderOffset;
    if (entry.uncompressedSize == 0xFFFFFFFFu || entry.compressedSize == 0xFFFFFFFFu ||
        entry.localHeaderOffset == 0xFFFFFFFFu) {
      if (!parseZip64ExtraField(fileData, filenameEnd, entry.extraLen, entry.uncompressedSize,
                                entry.compressedSize, entry.localHeaderOffset, &entryUncompressed,
                                &entryCompressed, &entryLocalOffset)) {
        return fail("saturated entry missing ZIP64 extra field: " + filename);
      }
    }

    const std::filesystem::path target = destRoot / std::filesystem::path(filename);
    if (!filename.empty() && (filename.back() == '/' || filename.back() == '\\')) {
      std::filesystem::create_directories(target, ec);
      if (ec) return fail("cannot create directory entry: " + target.string());
    } else {
      ZipLocalFileHeader local{};
      const size_t localOffset = static_cast<size_t>(entryLocalOffset);
      if (!readPod(fileData, localOffset, &local) || local.signature != 0x04034b50) {
        return fail("corrupt local header for entry: " + filename);
      }
      const size_t dataOffset =
          localOffset + sizeof(ZipLocalFileHeader) + local.filenameLen + local.extraLen;
      const size_t compressedSize = static_cast<size_t>(entryCompressed);
      const size_t uncompressedSize = static_cast<size_t>(entryUncompressed);
      if (dataOffset > fileSize || compressedSize > fileSize - dataOffset) {
        return fail("entry data out of bounds: " + filename);
      }

      std::vector<uint8_t> entryData;
      if (entry.compression == 0) {
        entryData.assign(fileData.data() + dataOffset, fileData.data() + dataOffset + compressedSize);
      } else if (entry.compression == 8) {
#ifdef HAVE_ZLIB
        std::string inflateError;
        if (!inflateDeflateRaw(fileData.data() + dataOffset, compressedSize, uncompressedSize,
                               &entryData, &inflateError)) {
          return fail("failed to inflate entry '" + filename + "': " + inflateError);
        }
#else
        return fail("deflated entry '" + filename + "' requires zlib support (HAVE_ZLIB)");
#endif
      } else {
        return fail("unsupported ZIP compression method " +
                    std::to_string(static_cast<unsigned>(entry.compression)) + " for entry: " +
                    filename);
      }

      std::filesystem::create_directories(target.parent_path(), ec);
      std::ofstream outFile(target, std::ios::binary | std::ios::trunc);
      if (!outFile.is_open()) return fail("cannot write extracted entry: " + target.string());
      outFile.write(reinterpret_cast<const char*>(entryData.data()),
                    static_cast<std::streamsize>(entryData.size()));
      if (!outFile.good()) return fail("short write for extracted entry: " + target.string());
    }

    if (nextOffset <= cdOffset || nextOffset > fileSize) break;
    cdOffset = nextOffset;
  }

  return true;
#endif  // SDZ_HAS_FILESYSTEM
}

SdnbReader::LoadedModel SdzReader::load(bool inferenceOnly,
                                        bool requireFileBacked) const {
  SdnbReader::LoadedModel combined;

  for (const auto& entry : sdnbEntries_) {
    if (requireFileBacked && !entry.fileBacked) {
      DSP_DIAG(COMPILE,
               "SdzReader: strict file-backed load rejected compressed SDNB "
               "entry '%s'",
               entry.filename.c_str());
      combined.graph = nullptr;
      return combined;
    }

    const uint8_t* entryData = entry.data;
    size_t entrySize = entry.size;
    std::shared_ptr<void> owner = entry.backingOwner;
    if (entry.compression == 8) {
#ifdef HAVE_ZLIB
      auto inflated = std::make_shared<std::vector<uint8_t>>();
      std::string inflateError;
      if (!inflateDeflateRaw(entry.data, entry.size, entry.uncompressedSize,
                             inflated.get(), &inflateError)) {
        DSP_DIAG(COMPILE, "SdzReader: failed to inflate '%s': %s",
                 entry.filename.c_str(), inflateError.c_str());
        continue;
      }
      entryData = inflated->data();
      entrySize = inflated->size();
      owner = inflated;
#else
      DSP_DIAG(COMPILE,
               "SdzReader: deflated entry '%s' requires zlib support",
               entry.filename.c_str());
      continue;
#endif
    }
    auto* sdnb = SdnbReader::openOwned(entryData, entrySize, owner,
                                       entry.fileBacked);
    if (!sdnb) {
      DSP_DIAG(COMPILE, "SdzReader: failed to parse SDNB entry '%s'",
               entry.filename.c_str());
      continue;
    }

    auto model = sdnb->loadAllOwned(inferenceOnly, requireFileBacked);

    // Merge variables
    for (auto& pair : model.variables) {
      auto existing = combined.variables.find(pair.first);
      if (existing != combined.variables.end()) delete existing->second;
      combined.variables[pair.first] = pair.second;
      pair.second = nullptr;  // Transfer ownership
    }

    // Merge placeholders
    for (auto& ph : model.placeholderNames) {
      combined.placeholderNames.push_back(ph);
    }

    // Variable-only shards also contain a valid FlatGraph envelope, but no
    // executable nodes. Archive entry order is not a format guarantee, so retain
    // the first graph-bearing shard instead of blindly retaining the first shard.
    const bool combinedHasNodes =
        combined.graph != nullptr && combined.graph->nodes() != nullptr &&
        combined.graph->nodes()->size() > 0;
    const bool modelHasNodes =
        model.graph != nullptr && model.graph->nodes() != nullptr &&
        model.graph->nodes()->size() > 0;
    if (combined.graph == nullptr || (!combinedHasNodes && modelHasNodes)) {
      combined.graph = model.graph;
    }

    combined.fileBackedBytes += model.fileBackedBytes;
    combined.heapOwnedBytes += model.heapOwnedBytes;
    combined.largeHeapOwnedBytes += model.largeHeapOwnedBytes;
    for (auto& backing : model.backingOwners) {
      combined.backingOwners.push_back(std::move(backing));
    }

    delete sdnb;
  }

  return combined;
}

}  // namespace graph
}  // namespace sd
