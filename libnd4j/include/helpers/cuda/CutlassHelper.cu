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

#include <helpers/CutlassHelper.h>
#include <system/Environment.h>
#include <config.h>

namespace sd {

int CutlassHelper::getSmVersion(int deviceId) {
  auto& caps = Environment::getInstance().capabilities();
  if (deviceId < 0 || deviceId >= static_cast<int>(caps.size()))
    return 0;
  // capabilities() returns vector<Pair> where first() = major, second() = minor
  int major = caps[deviceId].first();
  int minor = caps[deviceId].second();
  return major * 10 + minor;
}

bool CutlassHelper::hasFp8NativeSupport(int deviceId) {
  return getSmVersion(deviceId) >= 89;
}

bool CutlassHelper::hasHopperFeatures(int deviceId) {
  return getSmVersion(deviceId) >= 90;
}

bool CutlassHelper::isCutlassAvailable() {
#if HAVE_CUTLASS
  return true;
#else
  return false;
#endif
}

}  // namespace sd
