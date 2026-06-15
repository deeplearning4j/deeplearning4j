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

#ifndef LIBND4J_SDNB_READER_H
#define LIBND4J_SDNB_READER_H

#include <array/NDArray.h>
#include <graph/generated/graph_generated.h>
#include <system/common.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

/**
 * C++ reader for SDNB (SameDiff Native Binary) files.
 *
 * SDNB file format:
 *   [Java ObjectOutputStream manifest]  -- optional, for Java backward compat
 *   [FlatBuffer data (FlatGraph)]       -- at flatBufferOffset from header
 *   [Appended array data]               -- raw bytes, referenced by FlatArray fields
 *
 * The FlatGraph contains FlatVariables, each with a FlatArray.
 * FlatArray data can be:
 *   1. Inline: stored in the FlatBuffer's `buffer` field
 *   2. Appended: stored at `appendedDataOffset`/`appendedDataLength` in the file
 *   3. External: stored in separate files (externalDataFilename)
 */
class SD_LIB_EXPORT SdnbReader {
 public:
  /**
   * Result of loading a complete model.
   */
  struct LoadedModel {
    std::unordered_map<std::string, NDArray*> variables;  // Constants + variables (owned)
    std::vector<std::string> placeholderNames;
    const ::graph::FlatGraph* graph;                                // Points into reader's data

    ~LoadedModel() {
      for (auto& pair : variables) {
        delete pair.second;
      }
    }

    // No copy
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;

    // Move OK
    LoadedModel(LoadedModel&& other) noexcept
        : variables(std::move(other.variables)),
          placeholderNames(std::move(other.placeholderNames)),
          graph(other.graph) {
      other.graph = nullptr;
    }

    LoadedModel& operator=(LoadedModel&& other) noexcept {
      if (this != &other) {
        // Clean up existing variables
        for (auto& pair : variables) {
          delete pair.second;
        }
        variables = std::move(other.variables);
        placeholderNames = std::move(other.placeholderNames);
        graph = other.graph;
        other.graph = nullptr;
      }
      return *this;
    }

    LoadedModel() : graph(nullptr) {}
  };

  /**
   * Open from raw bytes (does not take ownership).
   */
  static SdnbReader* open(const void* data, size_t size);

  /**
   * Open from file path (reads entire file into memory).
   */
  static SdnbReader* openFile(const char* path);

  ~SdnbReader();

  // No copy
  SdnbReader(const SdnbReader&) = delete;
  SdnbReader& operator=(const SdnbReader&) = delete;

  /**
   * Access the parsed FlatGraph.
   */
  const ::graph::FlatGraph* getFlatGraph() const { return flatGraph_; }

  /**
   * Number of variables in the graph.
   */
  int numVariables() const;

  /**
   * Number of nodes (ops) in the graph.
   */
  int numNodes() const;

  /**
   * Load a single variable's array data by name.
   * Caller owns the returned NDArray.
   */
  NDArray* loadVariable(const char* name) const;

  /**
   * Load a single variable's array data by index.
   * Caller owns the returned NDArray.
   */
  NDArray* loadVariable(int index) const;

  /**
   * Load all variables and build the complete model.
   */
  LoadedModel loadAll() const;

 private:
  SdnbReader();

  const uint8_t* data_;
  size_t size_;
  bool ownsData_;

  // Parsed FlatGraph (points into data_)
  const ::graph::FlatGraph* flatGraph_;

  // File offsets (from SDNB header)
  long flatBufferOffset_;

  /**
   * Load NDArray from a FlatArray, handling inline, chunked, and appended data.
   */
  NDArray* loadFlatArray(const ::graph::FlatArray* fa) const;

  /**
   * Convert FlatBuffer DType to native DataType.
   */
  static DataType convertDType(::graph::DType fbDtype);
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_SDNB_READER_H
