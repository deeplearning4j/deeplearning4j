/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.nativeblas;

import java.util.ArrayList;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.systeminfo.GPUInfo;
import org.nd4j.systeminfo.GPUInfoProvider;

@Slf4j
public class NativeOpsGPUInfoProvider implements GPUInfoProvider {

    @Override
    public List<GPUInfo> getGPUs() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        List<GPUInfo> gpus = new ArrayList<>();


        int nDevices = nativeOps.getAvailableDevices();
        if (nDevices > 0) {
            for (int i = 0; i < nDevices; i++) {
                try {
                    String name = nativeOps.getDeviceName(i);
                    long total = nativeOps.getDeviceTotalMemory(i);
                    long free = nativeOps.getDeviceFreeMemory(i);
                    int major = nativeOps.getDeviceMajor(i);
                    int minor = nativeOps.getDeviceMinor(i);

                    gpus.add(new GPUInfo(name, total, free, major, minor));
                } catch (Exception e) {
                    log.warn("Can't add GPU", e);
                }
            }
        }

        return gpus;
    }

}
