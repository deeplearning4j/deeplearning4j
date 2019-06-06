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

import lombok.Data;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Data
public class DeviceInformation {
    private int deviceId;

    private int ccMajor = 0;
    private int ccMinor = 0;

    /**
     * Total amount of memory available on current specific device
     */
    private long totalMemory = 0;

    /**
     * Available RAM
     */
    private long availableMemory = 0;

    /**
     * This is amount of RAM allocated within current JVM process
     */
    private AtomicLong allocatedMemory = new AtomicLong(0);

    /*
        Key features we care about: hostMapped, overlapped exec, number of cores/sm
     */
    private boolean canMapHostMemory = false;

    private boolean overlappedKernels = false;

    private boolean concurrentKernels = false;

    private long sharedMemPerBlock = 0;

    private long sharedMemPerMP = 0;

    private int warpSize = 0;
}
