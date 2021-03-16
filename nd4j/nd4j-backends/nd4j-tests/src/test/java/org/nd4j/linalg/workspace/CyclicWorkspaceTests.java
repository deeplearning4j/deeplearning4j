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

package org.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

@Slf4j

public class CyclicWorkspaceTests extends BaseNd4jTestWithBackends {

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testBasicMechanics_1(Nd4jBackend backend) {
        val fShape = new long[]{128, 784};
        val lShape = new long[] {128, 10};
        val prefetchSize = 24;
        val configuration = WorkspaceConfiguration.builder().minSize(10 * 1024L * 1024L)
                .overallocationLimit(prefetchSize + 1).policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                .policyLearning(LearningPolicy.FIRST_LOOP).policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policySpill(SpillPolicy.REALLOCATE).build();

        for (int e = 0; e < 100; e++) {
            try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "randomNameHere" + 119)) {
                val fArray = Nd4j.create(fShape).assign(e);
                val lArray = Nd4j.create(lShape).assign(e);

//                log.info("Current offset: {}; Current size: {};", ws.getCurrentOffset(), ws.getCurrentSize());
            }
        }
    }

    @Test
    @Disabled
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testGc(Nd4jBackend backend) {
        val indArray = Nd4j.create(4, 4);
        indArray.putRow(0, Nd4j.create(new float[]{0, 2, -2, 0}));
        indArray.putRow(1, Nd4j.create(new float[]{0, 1, -1, 0}));
        indArray.putRow(2, Nd4j.create(new float[]{0, -1, 1, 0}));
        indArray.putRow(3, Nd4j.create(new float[]{0, -2, 2, 0}));

        for (int i = 0; i < 100000000; i++) {
            indArray.getRow(i % 3);
            //Thread.sleep(1);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
