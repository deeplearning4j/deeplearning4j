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

package org.nd4j.linalg.memory;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class AccountingTests extends BaseNd4jTest {
    public AccountingTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testDetached_1() {
        val array = Nd4j.createFromArray(1, 2, 3, 4, 5);
        assertEquals(DataType.INT, array.dataType());

        assertTrue(Nd4j.getMemoryManager().allocatedMemory(0) > 0L);
    }

    @Test
    public void testWorkspaceAccounting_1() {
        val wsConf = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024 * 1024)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .build();

        val before = Nd4j.getMemoryManager().allocatedMemory(0);

        val workspace = Nd4j.getWorkspaceManager().createNewWorkspace(wsConf, "random_name_here");

        val middle = Nd4j.getMemoryManager().allocatedMemory(0);

        Nd4j.getWorkspaceManager().destroyWorkspace(workspace);

        val after = Nd4j.getMemoryManager().allocatedMemory(0);

        assertTrue(middle > before);
        assertTrue(after < middle);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
