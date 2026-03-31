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

/**
 * Triton kernel cache bundle export/import.
 *
 * Bundles compiled PTX + metadata into a STORED-only ZIP archive (.tkcache)
 * that can be distributed across machines with compatible GPU architectures.
 *
 * Import target is the override directory (triton_override/), which has highest
 * priority in the kernel lookup chain, keeping the user's cache dir clean.
 */

#include <config.h>
#include <mutex>

#if HAVE_TRITON

#include <graph/gpu/TritonGraphBackend_internal.h>
#include <graph/gpu/TritonTargetDispatch.h>
#include <graph/DspDiagnostics.h>
#include <system/Environment.h>

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <process.h>
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#include <dirent.h>
#endif
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace sd::graph {

using namespace triton_internal;

// ─── ZIP format constants (STORED-only, no compression) ─────────────────────

static constexpr uint32_t ZIP_LOCAL_FILE_HEADER_SIG = 0x04034b50;
static constexpr uint32_t ZIP_CENTRAL_DIR_SIG       = 0x02014b50;
static constexpr uint32_t ZIP_EOCD_SIG              = 0x06054b50;

// ─── ZIP writing helpers ────────────────────────────────────────────────────

static void writeU16(std::ostream& out, uint16_t v) {
  out.put(static_cast<char>(v & 0xFF));
  out.put(static_cast<char>((v >> 8) & 0xFF));
}

static void writeU32(std::ostream& out, uint32_t v) {
  out.put(static_cast<char>(v & 0xFF));
  out.put(static_cast<char>((v >> 8) & 0xFF));
  out.put(static_cast<char>((v >> 16) & 0xFF));
  out.put(static_cast<char>((v >> 24) & 0xFF));
}

// ─── ZIP reading helpers ────────────────────────────────────────────────────

static uint16_t readU16(const unsigned char* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t readU32(const unsigned char* p) {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

// ─── CRC-32 (lookup table) ─────────────────────────────────────────────────

static uint32_t crc32Table[256];
static std::once_flag crc32TableOnce;

static void initCrc32Table() {
  std::call_once(crc32TableOnce, []() {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int j = 0; j < 8; j++) {
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      crc32Table[i] = c;
    }
  });
}

static uint32_t computeCrc32(const void* data, size_t len) {
  initCrc32Table();
  uint32_t crc = 0xFFFFFFFF;
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < len; i++) {
    crc = crc32Table[(crc ^ bytes[i]) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFF;
}

// ─── Internal entry struct ──────────────────────────────────────────────────

struct ZipEntry {
  std::string name;
  std::string data;
  uint32_t crc32;
  uint32_t localHeaderOffset;
};

// ─── Manifest generation ────────────────────────────────────────────────────

struct KernelEntryInfo {
  std::string hash;
  std::string kernelName;
  size_t ptxBytes;
  int numWarps;
  int sharedMemBytes;
};

static std::string generateManifest(const std::string& targetArch,
                                     const std::vector<KernelEntryInfo>& entries,
                                     size_t totalPtxBytes) {
  std::ostringstream json;
  json << "{\n";
  json << "  \"formatVersion\": 1,\n";

  // ISO 8601 timestamp
  std::time_t now = std::time(nullptr);
  std::tm tm = {};
#ifdef _WIN32
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char timeBuf[32];
  std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  json << "  \"createdAt\": \"" << timeBuf << "\",\n";

  json << "  \"targetArch\": \"" << targetArch << "\",\n";
  json << "  \"numKernels\": " << entries.size() << ",\n";
  json << "  \"totalPtxBytes\": " << totalPtxBytes << ",\n";
  json << "  \"compilerInfo\": {\n";
  json << "    \"tritonBackend\": \"NVIDIA\"\n";
  json << "  },\n";
  json << "  \"entries\": [\n";

  for (size_t i = 0; i < entries.size(); i++) {
    const auto& e = entries[i];
    json << "    {\n";
    json << "      \"hash\": \"" << e.hash << "\",\n";
    json << "      \"kernelName\": \"" << e.kernelName << "\",\n";
    json << "      \"ptxBytes\": " << e.ptxBytes << ",\n";
    json << "      \"numWarps\": " << e.numWarps << ",\n";
    json << "      \"sharedMemBytes\": " << e.sharedMemBytes << "\n";
    json << "    }";
    if (i + 1 < entries.size()) json << ",";
    json << "\n";
  }

  json << "  ]\n";
  json << "}\n";
  return json.str();
}

// ─── Architecture compatibility check ───────────────────────────────────────

static bool isArchCompatible(const std::string& bundleArch, const std::string& hostArch) {
  // Parse sm_XX format
  auto parseSm = [](const std::string& arch, int& major, int& minor) -> bool {
    if (arch.size() < 4 || arch.substr(0, 3) != "sm_") return false;
    const std::string num = arch.substr(3);
    if (num.size() < 2) return false;
    major = num[0] - '0';
    minor = std::atoi(num.c_str() + 1);
    return major >= 1 && major <= 9;
  };

  int bMaj = 0, bMin = 0, hMaj = 0, hMin = 0;
  if (!parseSm(bundleArch, bMaj, bMin) || !parseSm(hostArch, hMaj, hMin)) {
    DSP_DIAG(JIT, "TritonCacheBundle: cannot parse arch strings: bundle='%s' host='%s'",
             bundleArch.c_str(), hostArch.c_str());
    return false;
  }

  // PTX forward compat: bundle arch must be <= host arch within same major
  if (bMaj != hMaj) return false;
  return bMin <= hMin;
}

// ─── Directory helpers ──────────────────────────────────────────────────────

static bool ensureDir(const std::string& path) {
  if (path.empty()) return false;
  std::string currentPath;
  size_t start = 0;
  if (path[0] == '/') { currentPath = "/"; start = 1; }

  while (start <= path.size()) {
    size_t slashPos = path.find('/', start);
    std::string part = (slashPos == std::string::npos)
                           ? path.substr(start)
                           : path.substr(start, slashPos - start);
    start = (slashPos == std::string::npos) ? (path.size() + 1) : (slashPos + 1);
    if (part.empty()) continue;

    if (!currentPath.empty() && currentPath.back() != '/') currentPath += "/";
    currentPath += part;

    struct stat st;
    if (stat(currentPath.c_str(), &st) == 0) {
      if (!S_ISDIR(st.st_mode)) return false;
      continue;
    }
    if (errno != ENOENT) return false;
#ifdef _WIN32
    if (_mkdir(currentPath.c_str()) != 0 && errno != EEXIST) return false;
#else
    if (mkdir(currentPath.c_str(), 0755) != 0 && errno != EEXIST) return false;
#endif
  }
  return true;
}

// ─── Metadata parsing helpers ───────────────────────────────────────────────

static std::string getMetaValue(const std::string& metaText, const char* key) {
  std::string needle = std::string(key) + "=";
  size_t pos = metaText.find(needle);
  if (pos == std::string::npos) return "";
  size_t valStart = pos + needle.size();
  size_t lineEnd = metaText.find('\n', valStart);
  if (lineEnd == std::string::npos) return metaText.substr(valStart);
  return metaText.substr(valStart, lineEnd - valStart);
}

static int getMetaInt(const std::string& metaText, const char* key, int defaultVal) {
  std::string val = getMetaValue(metaText, key);
  if (val.empty()) return defaultVal;
  int result = defaultVal;
  parseIntValue(val, result);
  return result;
}

// =============================================================================
// Public API
// =============================================================================

int exportTritonCacheBundleImpl(const char* outputPath) {
  if (outputPath == nullptr || outputPath[0] == '\0') {
    DSP_DIAG(JIT, "TritonCacheBundle: export failed — null output path");
    return -1;
  }

  const auto& env = sd::Environment::getInstance();
  const std::string cacheDir = configuredOrDefaultTritonDir(
      env.tritonCacheDir(), env.homeDirectory(), "triton_cache");

  // Scan cache directory for .ptx + .meta pairs
  std::vector<std::string> hashes;
  std::vector<std::string> ptxContents;
  std::vector<std::string> metaContents;

  // Use opendir/readdir for portability
#ifdef _WIN32
  // Windows directory enumeration
  WIN32_FIND_DATAA findData;
  std::string searchPath = cacheDir + "\\*.ptx";
  HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);
  if (hFind == INVALID_HANDLE_VALUE) {
    DSP_DIAG(JIT, "TritonCacheBundle: no PTX files found in cache dir: %s", cacheDir.c_str());
    return -1;
  }
  do {
    std::string fname(findData.cFileName);
    if (fname.size() > 9 && fname.substr(0, 5) == "ttir_" && fname.substr(fname.size() - 4) == ".ptx") {
      std::string hash = fname.substr(5, fname.size() - 9);
      std::string ptxPath = cacheDir + "\\" + fname;
      std::string metaPath = cacheDir + "\\ttir_" + hash + ".meta";

      std::ifstream ptxFile(ptxPath, std::ios::binary);
      std::ifstream metaFile(metaPath);
      if (ptxFile.good() && metaFile.good()) {
        std::string ptxData((std::istreambuf_iterator<char>(ptxFile)), std::istreambuf_iterator<char>());
        std::string metaData((std::istreambuf_iterator<char>(metaFile)), std::istreambuf_iterator<char>());
        if (!ptxData.empty()) {
          hashes.push_back(hash);
          ptxContents.push_back(std::move(ptxData));
          metaContents.push_back(std::move(metaData));
        }
      }
    }
  } while (FindNextFileA(hFind, &findData));
  FindClose(hFind);
#else
  // POSIX directory enumeration
  DIR* dir = opendir(cacheDir.c_str());
  if (dir == nullptr) {
    DSP_DIAG(JIT, "TritonCacheBundle: cannot open cache dir: %s (errno=%d)", cacheDir.c_str(), errno);
    return -1;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string fname(entry->d_name);
    if (fname.size() > 9 && fname.substr(0, 5) == "ttir_" && fname.substr(fname.size() - 4) == ".ptx") {
      std::string hash = fname.substr(5, fname.size() - 9);
      std::string ptxPath = cacheDir + "/" + fname;
      std::string metaPath = cacheDir + "/ttir_" + hash + ".meta";

      std::ifstream ptxFile(ptxPath, std::ios::binary);
      std::ifstream metaFile(metaPath);
      if (ptxFile.good() && metaFile.good()) {
        std::string ptxData((std::istreambuf_iterator<char>(ptxFile)), std::istreambuf_iterator<char>());
        std::string metaData((std::istreambuf_iterator<char>(metaFile)), std::istreambuf_iterator<char>());
        if (!ptxData.empty()) {
          hashes.push_back(hash);
          ptxContents.push_back(std::move(ptxData));
          metaContents.push_back(std::move(metaData));
        }
      }
    }
  }
  closedir(dir);
#endif

  if (hashes.empty()) {
    DSP_DIAG(JIT, "TritonCacheBundle: no kernel pairs found in cache dir: %s", cacheDir.c_str());
    return -1;
  }

  // Get target architecture
  std::string targetArch = TritonTargetDispatch::getTargetArch();
  const std::string archOverride = env.tritonOverrideArch();
  if (!archOverride.empty()) targetArch = archOverride;

  // Build manifest
  std::vector<KernelEntryInfo> entryInfos;
  size_t totalPtxBytes = 0;
  for (size_t i = 0; i < hashes.size(); i++) {
    KernelEntryInfo info;
    info.hash = hashes[i];
    info.ptxBytes = ptxContents[i].size();
    info.kernelName = getMetaValue(metaContents[i], "kernelName");
    info.numWarps = getMetaInt(metaContents[i], "numWarps", 4);
    info.sharedMemBytes = getMetaInt(metaContents[i], "sharedMemBytes", 0);
    totalPtxBytes += info.ptxBytes;
    entryInfos.push_back(std::move(info));
  }

  std::string manifest = generateManifest(targetArch, entryInfos, totalPtxBytes);

  // Build ZIP entries
  std::vector<ZipEntry> zipEntries;

  // Manifest
  {
    ZipEntry e;
    e.name = "manifest.json";
    e.data = manifest;
    e.crc32 = computeCrc32(e.data.data(), e.data.size());
    zipEntries.push_back(std::move(e));
  }

  // Kernel files
  for (size_t i = 0; i < hashes.size(); i++) {
    {
      ZipEntry e;
      e.name = "kernels/ttir_" + hashes[i] + ".ptx";
      e.data = ptxContents[i];
      e.crc32 = computeCrc32(e.data.data(), e.data.size());
      zipEntries.push_back(std::move(e));
    }
    {
      ZipEntry e;
      e.name = "kernels/ttir_" + hashes[i] + ".meta";
      e.data = metaContents[i];
      e.crc32 = computeCrc32(e.data.data(), e.data.size());
      zipEntries.push_back(std::move(e));
    }
  }

  // Write ZIP file (STORED-only)
  std::ofstream out(outputPath, std::ios::binary | std::ios::trunc);
  if (!out.good()) {
    DSP_DIAG(JIT, "TritonCacheBundle: cannot open output file: %s", outputPath);
    return -1;
  }

  // DOS date/time for current time
  std::time_t nowTime = std::time(nullptr);
  std::tm tmNow = {};
#ifdef _WIN32
  gmtime_s(&tmNow, &nowTime);
#else
  gmtime_r(&nowTime, &tmNow);
#endif
  uint16_t dosTime = static_cast<uint16_t>((tmNow.tm_hour << 11) | (tmNow.tm_min << 5) | (tmNow.tm_sec / 2));
  uint16_t dosDate = static_cast<uint16_t>(((tmNow.tm_year - 80) << 9) | ((tmNow.tm_mon + 1) << 5) | tmNow.tm_mday);

  // Write local file headers + data
  for (auto& entry : zipEntries) {
    entry.localHeaderOffset = static_cast<uint32_t>(out.tellp());

    writeU32(out, ZIP_LOCAL_FILE_HEADER_SIG);
    writeU16(out, 20);      // version needed to extract
    writeU16(out, 0);       // general purpose bit flag
    writeU16(out, 0);       // compression method: STORED
    writeU16(out, dosTime);
    writeU16(out, dosDate);
    writeU32(out, entry.crc32);
    writeU32(out, static_cast<uint32_t>(entry.data.size())); // compressed size
    writeU32(out, static_cast<uint32_t>(entry.data.size())); // uncompressed size
    writeU16(out, static_cast<uint16_t>(entry.name.size())); // filename length
    writeU16(out, 0);       // extra field length

    out.write(entry.name.data(), static_cast<std::streamsize>(entry.name.size()));
    out.write(entry.data.data(), static_cast<std::streamsize>(entry.data.size()));
  }

  // Write central directory
  uint32_t centralDirStart = static_cast<uint32_t>(out.tellp());

  for (const auto& entry : zipEntries) {
    writeU32(out, ZIP_CENTRAL_DIR_SIG);
    writeU16(out, 20);      // version made by
    writeU16(out, 20);      // version needed to extract
    writeU16(out, 0);       // general purpose bit flag
    writeU16(out, 0);       // compression method: STORED
    writeU16(out, dosTime);
    writeU16(out, dosDate);
    writeU32(out, entry.crc32);
    writeU32(out, static_cast<uint32_t>(entry.data.size())); // compressed size
    writeU32(out, static_cast<uint32_t>(entry.data.size())); // uncompressed size
    writeU16(out, static_cast<uint16_t>(entry.name.size())); // filename length
    writeU16(out, 0);       // extra field length
    writeU16(out, 0);       // file comment length
    writeU16(out, 0);       // disk number start
    writeU16(out, 0);       // internal file attributes
    writeU32(out, 0);       // external file attributes
    writeU32(out, entry.localHeaderOffset);
    out.write(entry.name.data(), static_cast<std::streamsize>(entry.name.size()));
  }

  uint32_t centralDirEnd = static_cast<uint32_t>(out.tellp());
  uint32_t centralDirSize = centralDirEnd - centralDirStart;

  // Write end of central directory
  writeU32(out, ZIP_EOCD_SIG);
  writeU16(out, 0);       // disk number
  writeU16(out, 0);       // disk number with central dir
  writeU16(out, static_cast<uint16_t>(zipEntries.size()));
  writeU16(out, static_cast<uint16_t>(zipEntries.size()));
  writeU32(out, centralDirSize);
  writeU32(out, centralDirStart);
  writeU16(out, 0);       // comment length

  out.flush();
  if (!out.good()) {
    DSP_DIAG(JIT, "TritonCacheBundle: write failed for: %s", outputPath);
    return -1;
  }

  DSP_DIAG(JIT, "TritonCacheBundle: exported %zu kernels (%zu bytes PTX) to %s",
           hashes.size(), totalPtxBytes, outputPath);

  return static_cast<int>(hashes.size());
}

// ─── ZIP reading for import/inspect ─────────────────────────────────────────

struct ReadZipEntry {
  std::string name;
  std::string data;
};

static bool readZipEntries(const char* bundlePath, std::vector<ReadZipEntry>& entries) {
  std::ifstream file(bundlePath, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    DSP_DIAG(JIT, "TritonCacheBundle: cannot open bundle: %s", bundlePath);
    return false;
  }

  auto fileSize = file.tellg();
  if (fileSize < 22) {
    DSP_DIAG(JIT, "TritonCacheBundle: file too small: %s", bundlePath);
    return false;
  }

  file.seekg(0);
  std::vector<unsigned char> fileData(static_cast<size_t>(fileSize));
  file.read(reinterpret_cast<char*>(fileData.data()), fileSize);

  // Find EOCD (scan from end, handle up to 65535 byte comment)
  int64_t eocdPos = -1;
  int64_t scanStart = std::max(static_cast<int64_t>(0), static_cast<int64_t>(fileSize) - 65557);
  for (int64_t pos = static_cast<int64_t>(fileSize) - 22; pos >= scanStart; pos--) {
    if (readU32(fileData.data() + pos) == ZIP_EOCD_SIG) {
      eocdPos = pos;
      break;
    }
  }

  if (eocdPos < 0) {
    DSP_DIAG(JIT, "TritonCacheBundle: no EOCD found in: %s", bundlePath);
    return false;
  }

  uint16_t numEntries = readU16(fileData.data() + eocdPos + 10);
  uint32_t centralDirOffset = readU32(fileData.data() + eocdPos + 16);

  if (centralDirOffset >= static_cast<uint32_t>(fileSize)) {
    DSP_DIAG(JIT, "TritonCacheBundle: invalid central dir offset in: %s", bundlePath);
    return false;
  }

  // Parse central directory
  size_t cdPos = centralDirOffset;
  for (int i = 0; i < numEntries; i++) {
    if (cdPos + 46 > static_cast<size_t>(fileSize)) break;
    if (readU32(fileData.data() + cdPos) != ZIP_CENTRAL_DIR_SIG) break;

    uint16_t compression = readU16(fileData.data() + cdPos + 10);
    uint32_t uncompressedSize = readU32(fileData.data() + cdPos + 24);
    uint16_t nameLen = readU16(fileData.data() + cdPos + 28);
    uint16_t extraLen = readU16(fileData.data() + cdPos + 30);
    uint16_t commentLen = readU16(fileData.data() + cdPos + 32);
    uint32_t localHeaderOffset = readU32(fileData.data() + cdPos + 42);

    std::string name(reinterpret_cast<const char*>(fileData.data() + cdPos + 46), nameLen);

    // Only support STORED
    if (compression != 0) {
      DSP_DIAG(JIT, "TritonCacheBundle: unsupported compression %d for entry '%s'",
               compression, name.c_str());
      cdPos += 46 + nameLen + extraLen + commentLen;
      continue;
    }

    // Read data from local file header
    if (localHeaderOffset + 30 > static_cast<uint32_t>(fileSize)) {
      cdPos += 46 + nameLen + extraLen + commentLen;
      continue;
    }

    uint16_t localNameLen = readU16(fileData.data() + localHeaderOffset + 26);
    uint16_t localExtraLen = readU16(fileData.data() + localHeaderOffset + 28);
    size_t dataOffset = localHeaderOffset + 30 + localNameLen + localExtraLen;

    if (dataOffset + uncompressedSize > static_cast<size_t>(fileSize)) {
      cdPos += 46 + nameLen + extraLen + commentLen;
      continue;
    }

    ReadZipEntry entry;
    entry.name = name;
    entry.data = std::string(reinterpret_cast<const char*>(fileData.data() + dataOffset), uncompressedSize);
    entries.push_back(std::move(entry));

    cdPos += 46 + nameLen + extraLen + commentLen;
  }

  return !entries.empty();
}

// ─── Import ─────────────────────────────────────────────────────────────────

int importTritonCacheBundleImpl(const char* bundlePath, bool validateArch) {
  if (bundlePath == nullptr || bundlePath[0] == '\0') {
    DSP_DIAG(JIT, "TritonCacheBundle: import failed — null bundle path");
    return -1;
  }

  std::vector<ReadZipEntry> entries;
  if (!readZipEntries(bundlePath, entries)) return -1;

  // Find and parse manifest
  std::string manifestData;
  for (const auto& e : entries) {
    if (e.name == "manifest.json") {
      manifestData = e.data;
      break;
    }
  }

  if (manifestData.empty()) {
    DSP_DIAG(JIT, "TritonCacheBundle: no manifest.json in bundle: %s", bundlePath);
    return -1;
  }

  // Parse targetArch from manifest (simple string search, no JSON library)
  std::string bundleArch;
  {
    size_t pos = manifestData.find("\"targetArch\"");
    if (pos != std::string::npos) {
      size_t colonPos = manifestData.find(':', pos);
      if (colonPos != std::string::npos) {
        size_t firstQuote = manifestData.find('"', colonPos + 1);
        size_t secondQuote = manifestData.find('"', firstQuote + 1);
        if (firstQuote != std::string::npos && secondQuote != std::string::npos) {
          bundleArch = manifestData.substr(firstQuote + 1, secondQuote - firstQuote - 1);
        }
      }
    }
  }

  // Architecture validation
  if (validateArch && !bundleArch.empty()) {
    std::string hostArch = TritonTargetDispatch::getTargetArch();
    const std::string archOverride = sd::Environment::getInstance().tritonOverrideArch();
    if (!archOverride.empty()) hostArch = archOverride;

    if (!isArchCompatible(bundleArch, hostArch)) {
      DSP_DIAG(JIT, "TritonCacheBundle: architecture mismatch — bundle=%s host=%s",
               bundleArch.c_str(), hostArch.c_str());
      return -2;
    }
  }

  // Determine override directory as import target
  const auto& env = sd::Environment::getInstance();
  const std::string overrideDir = configuredOrDefaultTritonDir(
      env.tritonOverrideDir(), env.homeDirectory(), "triton_override");

  if (!ensureDir(overrideDir)) {
    DSP_DIAG(JIT, "TritonCacheBundle: cannot create override dir: %s", overrideDir.c_str());
    return -1;
  }

  // Extract kernel files to override directory
  int imported = 0;
  for (const auto& e : entries) {
    // Skip manifest and non-kernel entries
    if (e.name.find("kernels/") != 0) continue;

    // Extract filename from path
    size_t slashPos = e.name.rfind('/');
    if (slashPos == std::string::npos) continue;
    std::string filename = e.name.substr(slashPos + 1);
    if (filename.empty()) continue;

    std::string outPath = overrideDir + "/" + filename;
    std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      DSP_DIAG(JIT, "TritonCacheBundle: failed to write: %s", outPath.c_str());
      continue;
    }
    out.write(e.data.data(), static_cast<std::streamsize>(e.data.size()));
    out.flush();
    if (!out.good()) {
      DSP_DIAG(JIT, "TritonCacheBundle: write failed for: %s", outPath.c_str());
      continue;
    }

    // Count .ptx files as "imported kernels"
    if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".ptx") {
      imported++;
    }
  }

  // Enable kernel override if it was disabled
  if (imported > 0) {
    sd::Environment::getInstance().setTritonKernelOverride(true);
  }

  DSP_DIAG(JIT, "TritonCacheBundle: imported %d kernels from %s to %s",
           imported, bundlePath, overrideDir.c_str());

  return imported;
}

// ─── Inspect ────────────────────────────────────────────────────────────────

static thread_local std::string inspectResultBuffer;

const char* inspectTritonCacheBundleImpl(const char* bundlePath) {
  inspectResultBuffer.clear();

  if (bundlePath == nullptr || bundlePath[0] == '\0') {
    inspectResultBuffer = "{\"error\": \"null bundle path\"}";
    return inspectResultBuffer.c_str();
  }

  std::vector<ReadZipEntry> entries;
  if (!readZipEntries(bundlePath, entries)) {
    inspectResultBuffer = "{\"error\": \"cannot read bundle\"}";
    return inspectResultBuffer.c_str();
  }

  // Return manifest contents directly (it's already JSON)
  for (const auto& e : entries) {
    if (e.name == "manifest.json") {
      inspectResultBuffer = e.data;
      return inspectResultBuffer.c_str();
    }
  }

  inspectResultBuffer = "{\"error\": \"no manifest.json in bundle\"}";
  return inspectResultBuffer.c_str();
}

}  // namespace sd::graph

#endif  // HAVE_TRITON
