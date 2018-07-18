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

package org.nd4j.linalg.util;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * DeviceLocal implementation for INDArray, with special broadcast method
 * @author raver119@gmail.com
 */
@Slf4j
public class DeviceLocalNDArray extends DeviceLocal<INDArray> {

    public DeviceLocalNDArray() {
        super();
    }

    public DeviceLocalNDArray(INDArray array) {
        super();

        broadcast(array);
    }

    /**
     * This method duplicates array, and stores it to all devices
     *
     * @param array
     */
    public void broadcast(INDArray array) {
        if (array == null)
            return;

        Nd4j.getExecutioner().commit();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int i = 0; i < numDevices; i++) {
            // if current thread equal to this device - we just save it, without duplication
            if (Nd4j.getAffinityManager().getDeviceForCurrentThread() == i) {
                set(i, array);
            } else {
                set(i, Nd4j.getAffinityManager().replicateToDevice(i, array));
            }

        }
    }
}
