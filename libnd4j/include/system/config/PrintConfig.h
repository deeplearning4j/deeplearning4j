/* ******************************************************************************
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

#ifndef LIBND4J_PRINT_CONFIG_H
#define LIBND4J_PRINT_CONFIG_H

#include <system/common.h>

#include <atomic>

namespace sd {
namespace config {

/**
 * NDArray print options (NumPy-style printoptions): edge items,
 * threshold, line width, and floating-point precision.
 */
class SD_LIB_EXPORT PrintConfig {
 private:
  std::atomic<int> _edgeItems{3};
  std::atomic<int> _threshold{1000};
  std::atomic<int> _lineWidth{75};
  std::atomic<int> _precision{8};

 public:
  PrintConfig() = default;

  int edgeItems() { return _edgeItems.load(); }
  void setEdgeItems(int v);

  int threshold() { return _threshold.load(); }
  void setThreshold(int v);

  int lineWidth() { return _lineWidth.load(); }
  void setLineWidth(int v);

  int precision() { return _precision.load(); }
  void setPrecision(int v);

  /**
   * Initialize from environment variables (currently none, but
   * available for future use).
   */
  void initFromEnvironment();
};

}  // namespace config
}  // namespace sd

#endif  // LIBND4J_PRINT_CONFIG_H
