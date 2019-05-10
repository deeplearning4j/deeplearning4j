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

package org.nd4j.autodiff.samediff.flow;

import lombok.Data;
import lombok.NonNull;

/**
 * This class is a holder for state of loops imported from TensorFlow, via frame_name
 *
 * @author raver119@gmail.com
 */
@Data
public class FrameState {
    private String name;
    private long iterations = 0;
    private boolean rewindPlanned = false;
    private int rewindPosition = -1;

    private int numberOfEntries = 0;
    private int numberOfExits = 0;
    private boolean active = false;
    private int numberOfCycles;


    public FrameState(@NonNull String frame_name) {
        this.name = frame_name;
    }

    /**
     * This method returns number of cycles for this frame
     * @return
     */
    public int getNumberOfCycles() {
        return numberOfCycles;
    }

    /**
     * This method increments number of cycles by 1
     */
    public void incrementNumberOfCycles() {
        numberOfCycles++;
    }
}
