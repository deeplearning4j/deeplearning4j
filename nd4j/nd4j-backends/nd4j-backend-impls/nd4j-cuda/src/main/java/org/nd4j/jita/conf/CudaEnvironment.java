/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.jita.conf;

import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 * The cuda environment contains information
 * for a given {@link Configuration}
 * singleton.
 *
 * @author raver119@gmail.com
 */
public class CudaEnvironment {
    private static final CudaEnvironment INSTANCE = new CudaEnvironment();
    private static volatile Configuration configuration;
    private static Map<Integer, Integer> arch = new ConcurrentHashMap<>();

    private CudaEnvironment() {
        configuration = new Configuration();

    }

    public static CudaEnvironment getInstance() {
        return INSTANCE;
    }

    /**
     * Get the {@link Configuration}
     * for the environment
     * @return
     */
    public Configuration getConfiguration() {
        return configuration;
    }

    /**
     * Get the current device architecture
     * @return the major/minor version of
     * the current device
     */
    public int getCurrentDeviceArchitecture() {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        if (!arch.containsKey(deviceId)) {
            int major = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceMajor(new CudaPointer(deviceId));
            int minor = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceMinor(new CudaPointer(deviceId));
            Integer cc = Integer.parseInt(new String("" + major + minor));
            arch.put(deviceId, cc);
            return cc;
        }

        return arch.get(deviceId);
    }

    public void notifyConfigurationApplied() {
        configuration.updateDevice();
    }
}
