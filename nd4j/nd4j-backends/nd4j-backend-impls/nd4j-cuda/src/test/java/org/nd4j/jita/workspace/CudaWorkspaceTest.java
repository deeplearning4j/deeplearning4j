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

package org.nd4j.jita.workspace;

import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class CudaWorkspaceTest {

    @Test
    public void testCircularWorkspaceAsymmetry_1() {
        // circular workspace mode
        val configuration = WorkspaceConfiguration.builder().initialSize(10 * 1024 * 1024)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED).policyAllocation(AllocationPolicy.STRICT)
                .policySpill(SpillPolicy.FAIL).policyLearning(LearningPolicy.NONE).build();


        try (val ws = (CudaWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "circular_ws")) {
            val array = Nd4j.create(DataType.FLOAT, 10, 10);

            assertEquals(0, ws.getHostOffset());
            assertNotEquals(0, ws.getDeviceOffset());

            // we expect that this array has no data/buffer on HOST side
            assertEquals(AffinityManager.Location.DEVICE, Nd4j.getAffinityManager().getActiveLocation(array));

            // since this array doesn't have HOST buffer - it will allocate one now
            array.getDouble(3L);

            assertEquals(ws.getHostOffset(), ws.getDeviceOffset());
        }

        try (val ws = (CudaWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "circular_ws")) {
            assertEquals(ws.getHostOffset(), ws.getDeviceOffset());
        }

        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @Test
    public void testCircularWorkspaceAsymmetry_2() {
        // circular workspace mode
        val configuration = WorkspaceConfiguration.builder().initialSize(10 * 1024 * 1024)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED).policyAllocation(AllocationPolicy.STRICT)
                .policySpill(SpillPolicy.FAIL).policyLearning(LearningPolicy.NONE).build();

        val root = Nd4j.create(DataType.FLOAT, 1000000).assign(119);

        for (int e = 0; e < 100; e++) {
            try (val ws = (CudaWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "circular_ws")) {
                val array = Nd4j.create(DataType.FLOAT, root.shape());
                array.assign(root);

                array.data().getInt(3);

                assertEquals(ws.getHostOffset(), ws.getDeviceOffset());
            }
        }
    }

    @Test
    public void testCircularWorkspaceAsymmetry_3() {
        // circular workspace mode
        val configuration = WorkspaceConfiguration.builder().initialSize(10 * 1024 * 1024)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED).policyAllocation(AllocationPolicy.STRICT)
                .policySpill(SpillPolicy.FAIL).policyLearning(LearningPolicy.NONE).build();

        val root = Nd4j.create(DataType.FLOAT, 1000000).assign(119);

        for (int e = 0; e < 100; e++) {
            try (val ws = (CudaWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "circular_ws")) {
                val array = Nd4j.create(DataType.FLOAT, root.shape());
                array.assign(root);

                val second = Nd4j.create(DataType.FLOAT, root.shape());

                array.data().getInt(3);
            }
        }
    }
}