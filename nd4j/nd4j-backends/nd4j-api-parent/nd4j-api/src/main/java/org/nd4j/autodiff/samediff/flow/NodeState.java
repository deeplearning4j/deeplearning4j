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
import org.nd4j.linalg.primitives.Pair;

/**
 * This class describe Node state during execution time.
 *
 * @author raver119@gmail.com
 */
@Data
public class NodeState {
    private String nodeName;
    private boolean active = true;
    private int activeBranch = 0;
    private boolean executed = false;
    private long numCycles = 0;

    private int rewindPosition = -1;
    private String rewindNode;

    public NodeState(@NonNull String nodeName) {
        this.nodeName = nodeName;
    }

    public void incrementNumberOfCycles() {
        numCycles++;
    }

    public long getNumberOfCycles() {
        return numCycles;
    }
}
