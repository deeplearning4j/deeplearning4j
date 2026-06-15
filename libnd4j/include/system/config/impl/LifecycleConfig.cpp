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

#include <system/config/LifecycleConfig.h>
#include <system/config/EnvHelper.h>

namespace sd {
namespace config {

LifecycleConfig::LifecycleConfig() {
  // Defaults set via member initializers.
}

void LifecycleConfig::setStackDepth(int v) {
  if (v > 0) _stackDepth.store(v);
}

void LifecycleConfig::setReportInterval(int v) {
  if (v > 0) _reportInterval.store(v);
}

void LifecycleConfig::initFromEnvironment() {
#if defined(SD_GCC_FUNCTRACE)
  {
    int v = readBoolEnvTriState("SD_LIFECYCLE_TRACKING");
    if (v >= 0) _lifecycleTracking.store(v == 1);
  }
  {
    int v = readBoolEnvTriState("SD_TRACK_VIEWS");
    if (v == 0) _trackViews.store(false);
  }
  {
    int v = readBoolEnvTriState("SD_TRACK_DELETIONS");
    if (v == 0) _trackDeletions.store(false);
  }
  {
    int v = readIntEnv("SD_STACK_DEPTH", -1);
    if (v > 0) _stackDepth.store(v);
  }
  {
    int v = readIntEnv("SD_REPORT_INTERVAL", -1);
    if (v > 0) _reportInterval.store(v);
  }
  {
    int64_t v = readInt64Env("SD_MAX_DELETION_HISTORY", -1);
    if (v >= 0) _maxDeletionHistory.store(static_cast<size_t>(v));
  }
  _snapshotFiles.store(readBoolEnv("SD_LIFECYCLE_SNAPSHOT_FILES", false));
  _trackOperations.store(readBoolEnv("SD_LIFECYCLE_TRACK_OPERATIONS", false));
#endif
}

}  // namespace config
}  // namespace sd
