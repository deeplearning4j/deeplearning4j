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

//
// @author raver119@gmail.com
//

#include <execution/AffinityManager.h>

namespace sd {
    int AffinityManager::currentDeviceId() {
        return 0;
    }

    int AffinityManager::currentNativeDeviceId() {
        return 0;
    }

    int AffinityManager::numberOfDevices() {
        return 1;
    }

    void AffinityManager::setCurrentDevice(int deviceId) {
        // no-op
    }

    void AffinityManager::setCurrentNativeDevice(int deviceId) {
        // no-op
    }
}