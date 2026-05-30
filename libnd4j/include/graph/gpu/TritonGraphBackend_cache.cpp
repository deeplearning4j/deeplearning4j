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

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonGraphBackend.h>
#include <graph/gpu/TritonGraphBackend_internal.h>
#include <graph/gpu/TritonTargetDispatch.h>
#include <graph/DspDiagnostics.h>
#include <system/Environment.h>
#include <helpers/logger.h>

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <process.h>
#include <direct.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

namespace sd::graph {

using namespace triton_internal;

std::string TritonGraphBackend::getDiskCacheDir() const {
  const auto& env = sd::Environment::getInstance();
  return configuredOrDefaultTritonDir(env.tritonCacheDir(), env.homeDirectory(), "triton_cache");
}

bool TritonGraphBackend::ensureDiskCacheDir(const std::string& cacheDir) const {
  if (cacheDir.empty()) return false;

  std::string currentPath;
  size_t start = 0;
  if (cacheDir[0] == '/') {
    currentPath = "/";
    start = 1;
  }

  while (start <= cacheDir.size()) {
    size_t slashPos = cacheDir.find('/', start);
    std::string part = (slashPos == std::string::npos)
                           ? cacheDir.substr(start)
                           : cacheDir.substr(start, slashPos - start);
    start = (slashPos == std::string::npos) ? (cacheDir.size() + 1) : (slashPos + 1);
    if (part.empty()) continue;

    if (!currentPath.empty() && currentPath.back() != '/') currentPath += "/";
    currentPath += part;

    struct stat st;
    if (stat(currentPath.c_str(), &st) == 0) {
#ifdef _WIN32
      if (!(st.st_mode & _S_IFDIR)) {
#else
      if (!S_ISDIR(st.st_mode)) {
#endif
        DSP_DIAG(JIT, "TritonGraphBackend: cache path exists but is not a directory: %s",
                  currentPath.c_str());
        return false;
      }
      continue;
    }

    if (errno != ENOENT) {
      DSP_DIAG(JIT, "TritonGraphBackend: stat failed for cache path %s (errno=%d)",
                currentPath.c_str(), errno);
      return false;
    }

#ifdef _WIN32
    if (_mkdir(currentPath.c_str()) != 0 && errno != EEXIST) {
#else
    if (mkdir(currentPath.c_str(), 0755) != 0 && errno != EEXIST) {
#endif
      DSP_DIAG(JIT, "TritonGraphBackend: mkdir failed for cache path %s (errno=%d)",
                currentPath.c_str(), errno);
      return false;
    }
  }

  return true;
}

std::string TritonGraphBackend::computeDiskCacheHash(const std::string& ttirText,
                                                     int numWarps, int numStages) const {
  const auto& env = sd::Environment::getInstance();
  uint64_t hash = FNV1A64_OFFSET_BASIS;
  // NOTE: startSlot/endSlot and segmentShapeKey are intentionally EXCLUDED from
  // the disk cache hash.  Slot numbers are plan-lifetime-specific; the same
  // kernel ops get different slot assignments when a plan is destroyed and
  // recreated (e.g. between pages in a VLM pipeline).  segmentShapeKey is
  // derived from the shapes of slots in the current plan and also changes
  // between plan lifetimes even when the kernel is identical.  Including either
  // caused 100% disk cache misses across plan lifetimes, forcing full
  // recompilation of every kernel on every page.
  //
  // The ttirText (which contains the kernel name derived from op names, all
  // tensor shapes, and the full MLIR IR) plus compile params and target arch
  // already uniquely identify the compiled binary.  Two kernels with different
  // shapes produce different TTIR text, so shapes are implicitly captured.
  mixFNV1a(hash, &numWarps, sizeof(numWarps));
  mixFNV1a(hash, &numStages, sizeof(numStages));
  int numCTAs = std::max(1, env.tritonNumCTAs());
  int maxNreg = std::max(0, env.tritonMaxNreg());
  int fpFusion = env.tritonEnableFpFusion() ? 1 : 0;
  int disableLineInfo = env.tritonDisableLineInfo() ? 1 : 0;
  mixFNV1a(hash, &numCTAs, sizeof(numCTAs));
  mixFNV1a(hash, &maxNreg, sizeof(maxNreg));
  mixFNV1a(hash, &fpFusion, sizeof(fpFusion));
  mixFNV1a(hash, &disableLineInfo, sizeof(disableLineInfo));
  mixFNV1a(hash, ttirText.data(), ttirText.size());

  std::string arch = TritonTargetDispatch::getTargetArch();
  const std::string archOverride = env.tritonOverrideArch();
  if (!archOverride.empty()) {
    arch = archOverride;
  }
  if (!arch.empty()) {
    mixFNV1a(hash, arch.data(), arch.size());
  }

  std::ostringstream oss;
  oss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return oss.str();
}

bool TritonGraphBackend::loadBinaryFromDiskCache(int startSlot, int endSlot,
                                                 const std::string& cacheHash,
                                                 const TritonIRModule& irModule,
                                                 TritonCompiledBinary& binary) const {
  if (!sd::Environment::getInstance().tritonCacheEnabled()) return false;
  if (cacheHash.empty()) return false;

  const std::string cacheDir = getDiskCacheDir();
  std::ostringstream name;
  name << "ttir_" << cacheHash;
  const std::string basePath = cacheDir + "/" + name.str();
  const std::string ptxPath = basePath + ".ptx";
  const std::string metaPath = basePath + ".meta";

  std::ifstream ptxFile(ptxPath, std::ios::binary);
  if (!ptxFile.good()) return false;

  std::ifstream metaFile(metaPath);
  if (!metaFile.good()) return false;

  std::string ptxText((std::istreambuf_iterator<char>(ptxFile)),
                      std::istreambuf_iterator<char>());
  if (ptxText.empty()) return false;
  if (ptxText.back() != '\0') ptxText.push_back('\0');

  int metaNumWarps = irModule.numWarps;
  int metaSharedMem = 0;
  bool metaSharedMemPresent = false;
  int metaGlobalScratchBytes = 0;
  int metaGlobalScratchAlignment = 128;
  std::string metaKernelName;
  std::string line;
  while (std::getline(metaFile, line)) {
    size_t eqPos = line.find('=');
    if (eqPos == std::string::npos) continue;

    const std::string key = line.substr(0, eqPos);
    const std::string value = line.substr(eqPos + 1);
    if (key == "numWarps") {
      parseIntValue(value, metaNumWarps);
    } else if (key == "sharedMemBytes") {
      parseIntValue(value, metaSharedMem);
      metaSharedMemPresent = true;
    } else if (key == "globalScratchBytes") {
      parseIntValue(value, metaGlobalScratchBytes);
    } else if (key == "globalScratchAlignment") {
      parseIntValue(value, metaGlobalScratchAlignment);
    } else if (key == "kernelName") {
      metaKernelName = value;
    }
  }

  if (!metaKernelName.empty() && metaKernelName != irModule.kernelName) {
    return false;
  }

  // Older cache entries were missing sharedMemBytes metadata entirely.
  // Recompile those if PTX requires extern shared memory; otherwise launches
  // would pass sharedMem=0 and corrupt memory.
  // Note: sharedMemBytes=0 is VALID for element-wise Triton kernels that
  // declare extern .shared (Triton convention) but don't actually use it.
  // Only reject entries where the field was missing entirely (pre-metadata era).
  if (!metaSharedMemPresent && metaSharedMem == 0 && ptxUsesExternSharedMemory(ptxText)) {
    DSP_DIAG(JIT, "TritonGraphBackend: disk cache entry for [%d-%d] is stale "
             "(extern shared PTX with no sharedMemBytes metadata); forcing recompile",
             startSlot, endSlot);
    return false;
  }

  binary.data = new char[ptxText.size()];
  std::memcpy(binary.data, ptxText.data(), ptxText.size());
  binary.size = ptxText.size() - 1;  // Excludes null terminator
  binary.target = TritonTargetDispatch::detectTarget();
  binary.targetArch = TritonTargetDispatch::getTargetArch();
  const std::string archOverride = sd::Environment::getInstance().tritonOverrideArch();
  if (!archOverride.empty()) {
    binary.targetArch = archOverride;
  }
  binary.numWarps = metaNumWarps;
  binary.sharedMemBytes = metaSharedMem;
  binary.globalScratchBytes = metaGlobalScratchBytes;
  binary.globalScratchAlignment = metaGlobalScratchAlignment;

  DSP_DIAG(JIT, "TritonGraphBackend: disk cache HIT for sub-segment [%d-%d] (%zu bytes)",
           startSlot, endSlot, binary.size);
  return true;
}

bool TritonGraphBackend::loadBinaryFromDiskCacheByHash(
    const std::string& cacheHash,
    const std::string& kernelName,
    TritonCompiledBinary& binary) const {
  // Reload variant for module residency cache eviction recovery.
  // Mirrors loadBinaryFromDiskCache but does not require a TritonIRModule
  // (which would require an MLIR context the launch path does not have).
  if (!sd::Environment::getInstance().tritonCacheEnabled()) return false;
  if (cacheHash.empty()) return false;

  const std::string cacheDir = getDiskCacheDir();
  std::ostringstream name;
  name << "ttir_" << cacheHash;
  const std::string basePath = cacheDir + "/" + name.str();
  const std::string ptxPath = basePath + ".ptx";
  const std::string metaPath = basePath + ".meta";

  std::ifstream ptxFile(ptxPath, std::ios::binary);
  if (!ptxFile.good()) return false;

  std::ifstream metaFile(metaPath);
  if (!metaFile.good()) return false;

  std::string ptxText((std::istreambuf_iterator<char>(ptxFile)),
                      std::istreambuf_iterator<char>());
  if (ptxText.empty()) return false;
  if (ptxText.back() != '\0') ptxText.push_back('\0');

  int metaNumWarps = 0;
  int metaSharedMem = 0;
  bool metaSharedMemPresent = false;
  int metaGlobalScratchBytes = 0;
  int metaGlobalScratchAlignment = 128;
  std::string metaKernelName;
  std::string line;
  while (std::getline(metaFile, line)) {
    size_t eqPos = line.find('=');
    if (eqPos == std::string::npos) continue;
    const std::string key = line.substr(0, eqPos);
    const std::string value = line.substr(eqPos + 1);
    if (key == "numWarps") {
      parseIntValue(value, metaNumWarps);
    } else if (key == "sharedMemBytes") {
      parseIntValue(value, metaSharedMem);
      metaSharedMemPresent = true;
    } else if (key == "globalScratchBytes") {
      parseIntValue(value, metaGlobalScratchBytes);
    } else if (key == "globalScratchAlignment") {
      parseIntValue(value, metaGlobalScratchAlignment);
    } else if (key == "kernelName") {
      metaKernelName = value;
    }
  }

  if (!kernelName.empty() && !metaKernelName.empty() && metaKernelName != kernelName) {
    DSP_DIAG(JIT, "TritonGraphBackend: reload disk cache hash %s metadata kernel name '%s' "
             "does not match expected '%s'",
             cacheHash.c_str(), metaKernelName.c_str(), kernelName.c_str());
    return false;
  }

  if (!metaSharedMemPresent && metaSharedMem == 0 && ptxUsesExternSharedMemory(ptxText)) {
    DSP_DIAG(JIT, "TritonGraphBackend: reload disk cache hash %s is stale "
             "(extern shared PTX with no sharedMemBytes metadata)", cacheHash.c_str());
    return false;
  }

  binary.data = new char[ptxText.size()];
  std::memcpy(binary.data, ptxText.data(), ptxText.size());
  binary.size = ptxText.size() - 1;
  binary.target = TritonTargetDispatch::detectTarget();
  binary.targetArch = TritonTargetDispatch::getTargetArch();
  const std::string archOverride = sd::Environment::getInstance().tritonOverrideArch();
  if (!archOverride.empty()) {
    binary.targetArch = archOverride;
  }
  binary.numWarps = metaNumWarps;
  binary.sharedMemBytes = metaSharedMem;
  binary.globalScratchBytes = metaGlobalScratchBytes;
  binary.globalScratchAlignment = metaGlobalScratchAlignment;

  DSP_DIAG(JIT, "TritonGraphBackend: reload disk cache HIT for hash %s (%zu bytes)",
           cacheHash.c_str(), binary.size);
  return true;
}

void TritonGraphBackend::writeBinaryToDiskCache(int startSlot, int endSlot,
                                                const std::string& cacheHash,
                                                const TritonIRModule& irModule,
                                                const TritonCompiledBinary& binary) const {
  if (!sd::Environment::getInstance().tritonCacheEnabled()) return;
  if (cacheHash.empty() || binary.data == nullptr || binary.size == 0) return;

  const std::string cacheDir = getDiskCacheDir();
  if (!ensureDiskCacheDir(cacheDir)) return;

  std::ostringstream name;
  name << "ttir_" << cacheHash;
  const std::string basePath = cacheDir + "/" + name.str();
  const std::string ptxPath = basePath + ".ptx";
  const std::string metaPath = basePath + ".meta";

  const auto tidHash = std::hash<std::thread::id>()(std::this_thread::get_id());
  std::ostringstream suffix;
  suffix << ".tmp." << static_cast<long long>(::getpid()) << "." << tidHash;
  const std::string ptxTmp = ptxPath + suffix.str();
  const std::string metaTmp = metaPath + suffix.str();

  {
    std::ofstream out(ptxTmp, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      DSP_DIAG(JIT, "TritonGraphBackend: failed to open PTX cache temp file %s", ptxTmp.c_str());
      return;
    }
    out.write(static_cast<const char*>(binary.data), static_cast<std::streamsize>(binary.size));
    out.flush();
    if (!out.good()) {
      DSP_DIAG(JIT, "TritonGraphBackend: failed to write PTX cache temp file %s", ptxTmp.c_str());
      out.close();
      std::remove(ptxTmp.c_str());
      return;
    }
  }

  if (std::rename(ptxTmp.c_str(), ptxPath.c_str()) != 0) {
    DSP_DIAG(JIT, "TritonGraphBackend: failed to finalize PTX cache file %s (errno=%d)",
              ptxPath.c_str(), errno);
    std::remove(ptxTmp.c_str());
    return;
  }

  std::ostringstream meta;
  meta << "numWarps=" << binary.numWarps << "\n";
  meta << "sharedMemBytes=" << binary.sharedMemBytes << "\n";
  meta << "globalScratchBytes=" << binary.globalScratchBytes << "\n";
  meta << "globalScratchAlignment=" << binary.globalScratchAlignment << "\n";
  meta << "kernelName=" << irModule.kernelName << "\n";
  meta << "gridX=" << irModule.gridX << "\n";
  meta << "gridY=" << irModule.gridY << "\n";
  meta << "gridZ=" << irModule.gridZ << "\n";
  meta << "blockX=" << irModule.blockX << "\n";
  meta << "blockY=" << irModule.blockY << "\n";
  meta << "blockZ=" << irModule.blockZ << "\n";
  meta << "useIndirectArgs=" << (irModule.useIndirectArgs ? 1 : 0) << "\n";
  meta << "argSlotMapping=";
  for (size_t i = 0; i < irModule.args.size(); i++) {
    if (i > 0) meta << ";";
    const auto& arg = irModule.args[i];
    meta << arg.slotIndex << "," << arg.outputIndex << ","
         << (arg.isOutput ? 1 : 0) << "," << static_cast<int>(arg.dtype);
  }
  meta << "\n";

  {
    std::ofstream out(metaTmp, std::ios::trunc);
    if (!out.good()) {
      DSP_DIAG(JIT, "TritonGraphBackend: failed to open metadata cache temp file %s", metaTmp.c_str());
      std::remove(metaTmp.c_str());
      return;
    }
    out << meta.str();
    out.flush();
    if (!out.good()) {
      DSP_DIAG(JIT, "TritonGraphBackend: failed to write metadata cache temp file %s", metaTmp.c_str());
      out.close();
      std::remove(metaTmp.c_str());
      return;
    }
  }

  if (std::rename(metaTmp.c_str(), metaPath.c_str()) != 0) {
    DSP_DIAG(JIT, "TritonGraphBackend: failed to finalize metadata cache file %s (errno=%d)",
              metaPath.c_str(), errno);
    std::remove(metaTmp.c_str());
    return;
  }

  DSP_DIAG(JIT, "TritonGraphBackend: disk cache STORED for sub-segment [%d-%d] (%zu bytes)",
           startSlot, endSlot, binary.size);
}

}  // namespace sd::graph

#endif // HAVE_TRITON
