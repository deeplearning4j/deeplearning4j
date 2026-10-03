/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.dl4jcore.nn.misc;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.WorkspaceUseMetaData;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * A workspace manager recorded a WorkspaceUseMetaData, stack trace included, for every enter, borrow and close of its
 * workspaces and the event log kept all of them, whatever Environment#isTrackWorkspaceOpenClose said. A network opens
 * and closes workspaces several times per layer per iteration, so the heap filled with stack frames (CNNGradientCheckTest
 * ran out of its 8 GB heap). The events are recorded only while tracking is on.
 */
@NativeTag
@Tag(TagNames.WORKSPACES)
public class WorkspaceEventTrackingTest extends BaseDL4JTest {

    private static final int CYCLES = 1000;

    private static LayerWorkspaceMgr manager(String workspaceName) {
        WorkspaceConfiguration conf = WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.02)
                .policyLearning(LearningPolicy.OVER_TIME)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();
        return LayerWorkspaceMgr.builder()
                .with(ArrayType.ACTIVATIONS, workspaceName, conf)
                .defaultNoWorkspace()
                .build();
    }

    /** Opens and closes the manager's ACTIVATIONS workspace, returning the workspace's unique id. */
    private static long cycle(LayerWorkspaceMgr mgr, int times) {
        long id = -1;
        for (int i = 0; i < times; i++) {
            try (MemoryWorkspace ws = mgr.notifyScopeEntered(ArrayType.ACTIVATIONS)) {
                id = ws.getUniqueId();
                Nd4j.create(4);
            }
        }
        return id;
    }

    /**
     * A workspace's id was a fresh counter value on every call, so the deallocator service filed it under one id and
     * removed it by another (the entry stayed, and a flush of dead entries deallocated the workspace again), and every
     * logged event had a key of its own.
     */
    @Test
    public void workspaceUniqueIdIsStable() {
        LayerWorkspaceMgr mgr = manager("WS_EVENT_TRACKING_ID");
        try (MemoryWorkspace ws = mgr.notifyScopeEntered(ArrayType.ACTIVATIONS)) {
            assertEquals(ws.getUniqueId(), ws.getUniqueId());
        }
    }

    private static long loggedWorkspaceEvents() {
        return Nd4j.getExecutioner().getNd4jEventLog().workspaceEvents().values().stream().mapToLong(List::size).sum();
    }

    @Test
    public void workspaceCyclesRecordNothingWhileTrackingIsOff() {
        assertFalse(Nd4j.getEnvironment().isTrackWorkspaceOpenClose(), "workspace tracking is off by default");
        long before = loggedWorkspaceEvents();
        cycle(manager("WS_EVENT_TRACKING_OFF"), CYCLES);
        assertEquals(before, loggedWorkspaceEvents(), "workspace events recorded with tracking off");
    }

    @Test
    public void workspaceCyclesAreRecordedWhileTrackingIsOn() {
        Nd4j.getEnvironment().setTrackWorkspaceOpenClose(true);
        try {
            long id = cycle(manager("WS_EVENT_TRACKING_ON"), 10);
            List<WorkspaceUseMetaData> events = Nd4j.getExecutioner().getNd4jEventLog().eventsForWorkspaceUniqueId(id);
            assertEquals(10, events.stream().filter(e -> e.getEventType() == WorkspaceUseMetaData.EventTypes.ENTER).count(),
                    "one enter event per cycle");
            assertEquals(10, events.stream().filter(e -> e.getEventType() == WorkspaceUseMetaData.EventTypes.CLOSE).count(),
                    "one close event per cycle");
        } finally {
            Nd4j.getEnvironment().setTrackWorkspaceOpenClose(false);
        }
    }
}
