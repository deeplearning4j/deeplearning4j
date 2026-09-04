/* ******************************************************************************
 *
 *  Copyright (c) 2026 Konduit K.K.
 *
 *  This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
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
 *******************************************************************************/

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.DspLifecycleSnapshot;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DspLifecycleSnapshotTest {

    @BeforeEach
    void enableDynamicShapePlan() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    void commitPendingWork() {
        Nd4j.getExecutioner().commit();
    }

    @Test
    void snapshotIsNativeAuthoritativeAndImmutable() {
        SameDiff sameDiff = SameDiff.create();
        try {
            sameDiff.placeHolder("input", DataType.FLOAT, 2, 2)
                    .mul("out", 2.0);
            sameDiff.setGraphExecutionMode(GraphExecutionMode.EMULATED_REPLAY);

            sameDiff.output(Collections.singletonMap("input", Nd4j.ones(DataType.FLOAT, 2, 2)), "out");
            DynamicShapePlanExecutor executor =
                    sameDiff.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            DspHandle handle = sameDiff.dsp();

            DspLifecycleSnapshot before = handle.lifecycleSnapshot();
            assertTrue(before.isValid(), "native lifecycle snapshot must be valid after execution");
            assertTrue(before.getExecutionCount() >= 1,
                    "execution count must come from the native plan");
            assertTrue(before.getSegmentCount() > 0,
                    "snapshot must include the native segment topology");
            assertEquals(before.getPlanPhase(), executor.getPlanPhase(),
                    "convenience phase query must use the same native snapshot contract");

            sameDiff.output(Collections.singletonMap("input", Nd4j.ones(DataType.FLOAT, 2, 2)), "out");
            DspLifecycleSnapshot after = handle.lifecycleSnapshot();
            assertTrue(after.getExecutionCount() > before.getExecutionCount(),
                    "native execution count did not advance between executions");
            assertTrue(after.getSegmentCount() == before.getSegmentCount(),
                    "lifecycle snapshot changed immutable segment topology");
            assertThrows(UnsupportedOperationException.class,
                    () -> after.asMap().put("planPhase", "mutated"));
        } finally {
            sameDiff.close();
        }
    }
}
