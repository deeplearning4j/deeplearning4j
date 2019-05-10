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

package org.nd4j.linalg.api.memory.conf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.memory.enums.*;

import java.io.Serializable;

/**
 * This class is configuration bean for MemoryWorkspace.
 * It allows you to specify workspace parameters, and will define workspace behaviour.
 *
 * @author raver119@gmail.com
 */
@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
// TODO: add json mapping here
public class WorkspaceConfiguration implements Serializable {
    @Builder.Default protected AllocationPolicy policyAllocation = AllocationPolicy.OVERALLOCATE;
    @Builder.Default protected SpillPolicy policySpill = SpillPolicy.EXTERNAL;
    @Builder.Default protected MirroringPolicy policyMirroring = MirroringPolicy.FULL;
    @Builder.Default protected LearningPolicy policyLearning = LearningPolicy.FIRST_LOOP;
    @Builder.Default protected ResetPolicy policyReset = ResetPolicy.BLOCK_LEFT;
    @Builder.Default protected LocationPolicy policyLocation = LocationPolicy.RAM;

    /**
     * Path to file to be memory-mapped
     */
    @Builder.Default protected String tempFilePath = null;

    /**
     * This variable specifies amount of memory allocated for this workspace during initialization
     */
    @Builder.Default protected long initialSize = 0;

    /**
     * This variable specifies minimal workspace size
     */
    @Builder.Default protected long minSize = 0;

    /**
     * This variable specifies maximal workspace size
     */
    @Builder.Default protected long maxSize = 0;

    /**
     * For workspaces with learnable size, this variable defines how many cycles will be spent during learning phase
     */
    @Builder.Default protected int cyclesBeforeInitialization = 0;

    /**
     * If OVERALLOCATION policy is set, memory will be overallocated in addition to initialSize of learned size
     */
    @Builder.Default protected double overallocationLimit = 0.3;

    /**
     * This value is used only for circular workspaces
     */
    @Builder.Default protected int stepsNumber = 2;
}
