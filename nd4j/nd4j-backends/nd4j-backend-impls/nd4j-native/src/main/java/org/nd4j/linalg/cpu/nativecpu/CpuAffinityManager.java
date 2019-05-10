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

package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.BasicAffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public class CpuAffinityManager extends BasicAffinityManager {

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     * Has no effect on CPU backend.
     *
     * @param array
     */
    @Override
    public void touch(INDArray array) {
        // no-op
    }

    /**
     * Utility method, to associate INDArray with specific device (backend-specific)
     * Has no effect on CPU backend.
     *
     * @param buffer
     */
    @Override
    public void touch(DataBuffer buffer) {
        // no-op
    }
}
