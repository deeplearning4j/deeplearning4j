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

#include <system/config/MemoryConfig.h>
#include <system/config/EnvHelper.h>

namespace sd {
namespace config {

MemoryConfig::MemoryConfig() {
  // Defaults set via member initializers.
}

int MemoryConfig::poolReleaseThresholdPercent() {
  return _poolReleaseThresholdPercent.load();
}

int MemoryConfig::nonPeerHeadroomPercent() {
  return _nonPeerHeadroomPercent.load();
}

void MemoryConfig::setPoolReleaseThresholdPercent(int percent) {
  if (percent < 1) percent = 1;
  if (percent > 100) percent = 100;
  _poolReleaseThresholdPercent.store(percent);
}

void MemoryConfig::setNonPeerHeadroomPercent(int percent) {
  if (percent < 1) percent = 1;
  if (percent > 100) percent = 100;
  _nonPeerHeadroomPercent.store(percent);
}

void MemoryConfig::initFromEnvironment() {
  {
    int v = readIntEnv("SD_POOL_RELEASE_THRESHOLD_PERCENT", -1);
    if (v >= 1 && v <= 100) _poolReleaseThresholdPercent.store(v);
  }
  {
    int v = readIntEnv("SD_NON_PEER_HEADROOM_PERCENT", -1);
    if (v >= 1 && v <= 100) _nonPeerHeadroomPercent.store(v);
  }
}

}  // namespace config
}  // namespace sd
