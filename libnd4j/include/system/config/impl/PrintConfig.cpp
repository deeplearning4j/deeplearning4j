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

#include <system/config/PrintConfig.h>

namespace sd {
namespace config {

void PrintConfig::setEdgeItems(int v) {
  if (v > 0) _edgeItems.store(v);
}

void PrintConfig::setThreshold(int v) {
  if (v > 0) _threshold.store(v);
}

void PrintConfig::setLineWidth(int v) {
  if (v > 0) _lineWidth.store(v);
}

void PrintConfig::setPrecision(int v) {
  if (v >= 0 && v <= 20) _precision.store(v);
}

void PrintConfig::initFromEnvironment() {
  // No env vars for print config currently.
  // Future: ND4J_PRINT_EDGE_ITEMS, etc.
}

}  // namespace config
}  // namespace sd
