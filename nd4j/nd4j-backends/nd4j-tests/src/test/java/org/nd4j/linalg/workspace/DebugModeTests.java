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

package org.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import static org.junit.Assert.assertEquals;

@Slf4j
@RunWith(Parameterized.class)
public class DebugModeTests extends BaseNd4jTest {
    DataBuffer.Type initialType;

    public DebugModeTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void turnMeUp() {
        Nd4j.getWorkspaceManager().setDebugMode(DebugMode.DISABLED);
    }

    @After
    public void turnMeDown() {
        Nd4j.getWorkspaceManager().setDebugMode(DebugMode.DISABLED);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testDebugMode_1() {
        assertEquals(DebugMode.DISABLED, Nd4j.getWorkspaceManager().getDebugMode());

        Nd4j.getWorkspaceManager().setDebugMode(DebugMode.SPILL_EVERYTHING);

        assertEquals(DebugMode.SPILL_EVERYTHING, Nd4j.getWorkspaceManager().getDebugMode());
    }

    @Test
    public void testSpillMode_1() {
        Nd4j.getWorkspaceManager().setDebugMode(DebugMode.SPILL_EVERYTHING);

        val basicConfig = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024 * 1024).maxSize(10 * 1024 * 1024).overallocationLimit(0.1)
                .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build();

        try (val ws = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "R_119_1992")) {
            assertEquals(10 * 1024 * 1024L, ws.getCurrentSize());
            assertEquals(0, ws.getDeviceOffset());
            assertEquals(0, ws.getHostOffset());

            val array = Nd4j.create(10, 10).assign(1.0f);

            // nothing should get into workspace
            assertEquals(0, ws.getHostOffset());
            assertEquals(0, ws.getDeviceOffset());

            // array buffer should be spilled now
            assertEquals(10 * 10 * Nd4j.sizeOfDataType(), ws.getSpilledSize());
        }
    }

    @Test
    public void testSpillMode_2() {
        Nd4j.getWorkspaceManager().setDebugMode(DebugMode.SPILL_EVERYTHING);

        val basicConfig = WorkspaceConfiguration.builder()
                .initialSize(0).maxSize(10 * 1024 * 1024).overallocationLimit(0.1)
                .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build();

        try (val ws = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "R_119_1992")) {
            assertEquals(0L, ws.getCurrentSize());
            assertEquals(0, ws.getDeviceOffset());
            assertEquals(0, ws.getHostOffset());

            val array = Nd4j.create(10, 10).assign(1.0f);

            // nothing should get into workspace
            assertEquals(0, ws.getHostOffset());
            assertEquals(0, ws.getDeviceOffset());

            // array buffer should be spilled now
            assertEquals(10 * 10 * Nd4j.sizeOfDataType(), ws.getSpilledSize());
        }

        try (val ws = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "R_119_1992")) {
            assertEquals(0L, ws.getCurrentSize());
            assertEquals(0, ws.getDeviceOffset());
            assertEquals(0, ws.getHostOffset());
            assertEquals(0, ws.getSpilledSize());
        }
    }
}
