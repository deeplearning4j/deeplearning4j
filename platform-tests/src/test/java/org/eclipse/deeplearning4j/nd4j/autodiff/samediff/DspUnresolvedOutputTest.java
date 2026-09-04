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

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DspUnresolvedOutputTest {

    @Test
    void serializeRejectsRequestedOutputWithoutProducerSlot() {
        DynamicShapeSlot slot = DynamicShapeSlot.builder()
                .opName("synthetic")
                .customOp(true)
                .inputSourceIndices(new int[0])
                .inputSourceTypes(new byte[0])
                .inputVarNames(new String[0])
                .outputSlotIndices(new int[]{0})
                .outputVarNames(new String[]{"produced"})
                .iArgs(new long[0])
                .tArgs(new double[0])
                .bArgs(new boolean[0])
                .dArgs(new DataType[0])
                .sArgs(new String[0])
                .stepIndex(0)
                .opNameHash(1L)
                .needsIntLongSync(false)
                .requiresDynamicShapeInference(false)
                .allIntLongInputsExternal(true)
                .outputShapeDependsOnInputValues(false)
                .build();

        DynamicShapePlan plan = new DynamicShapePlan(
                new DynamicShapeSlot[]{slot},
                1,
                new int[][]{new int[0]},
                new org.nd4j.linalg.api.ops.OpContext[0],
                new String[0],
                new byte[0],
                Collections.singleton("missing"),
                Collections.emptyMap(),
                false,
                null,
                null,
                null,
                null,
                null,
                null);

        IllegalStateException error = assertThrows(
                IllegalStateException.class, plan::serialize);
        assertTrue(error.getMessage().contains("missing"));
        assertTrue(error.getMessage().contains("valid producer/output slot"));
    }
}
