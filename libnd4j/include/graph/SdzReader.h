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

#ifndef LIBND4J_SDZ_READER_H
#define LIBND4J_SDZ_READER_H

#include <graph/SdnbReader.h>

#include <string>
#include <memory>
#include <utility>
#include <vector>

namespace sd {
namespace graph {

/**
 * C++ reader for SDZ (SameDiff ZIP) files.
 *
 * SDZ files are ZIP archives containing one or more SDNB files.
 * This reader extracts the SDNB entries and delegates to SdnbReader
 * for the actual model loading.
 *
 * Supports STORED entries and, when built with zlib, DEFLATE entries used by
 * SameDiff's SDZ serializer.
 */
class SD_LIB_EXPORT SdzReader {
 public:
  /**
   * Open an SDZ (ZIP) file.
   */
  static SdzReader* openFile(const char* zipPath);

  /**
   * Extract ALL entries of a ZIP archive (any entry name; STORED and, with
   * zlib, DEFLATE) into destDir, recreating subdirectories. Used for packed
   * .dspb bundle archives. Rejects absolute and '..'-traversing entry names.
   * Requires std::filesystem; returns false with an error message otherwise.
   */
  static bool extractArchive(const char* zipPath, const char* destDir, std::string* errorOut);

  ~SdzReader();

  // No copy
  SdzReader(const SdzReader&) = delete;
  SdzReader& operator=(const SdzReader&) = delete;

  /**
   * Load the model from the ZIP-packed SDNB files.
   * Returns the loaded model with all variables.
   */
  SdnbReader::LoadedModel load(bool inferenceOnly = false,
                               bool requireFileBacked = false) const;

  /**
   * Get the number of SDNB entries in the ZIP.
   */
  int numEntries() const { return static_cast<int>(sdnbEntries_.size()); }

 private:
  SdzReader() = default;

  struct SdnbEntry {
    std::string filename;
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t uncompressedSize = 0;
    uint16_t compression = 0;
    bool fileBacked = false;
    std::shared_ptr<void> backingOwner;
    std::vector<uint8_t> ownedData;
  };

  std::vector<SdnbEntry> sdnbEntries_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_SDZ_READER_H
