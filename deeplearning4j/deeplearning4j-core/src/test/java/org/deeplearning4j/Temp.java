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

package org.deeplearning4j;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

public class Temp {

    @Test
    public void testWorkspaceBool(){
        WorkspaceConfiguration conf = WorkspaceConfiguration.builder().minSize(10 * 1024 * 1024)
                .overallocationLimit(1.0).policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL)
                .policySpill(SpillPolicy.EXTERNAL).build();

        MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(conf, "WS");

        for( int i=0; i<10; i++ ) {
            try (Nd4jWorkspace workspace = (Nd4jWorkspace)ws.notifyScopeEntered() ) {
                INDArray bool = Nd4j.create(DataType.BOOL, 1, 10);
                INDArray dbl = Nd4j.create(DataType.DOUBLE, 1, 10);

                boolean boolAttached = bool.isAttached();
                boolean doubleAttached = dbl.isAttached();

                System.out.println(i + "\tboolAttached=" + boolAttached + ", doubleAttached=" + doubleAttached );
                System.out.println("bool: " + bool);        //java.lang.IllegalStateException: Indexer must never be null
                System.out.println("double: " + dbl);
            }
        }
    }

    @Test
    public void testWorkspace2(){
        WorkspaceConfiguration conf = WorkspaceConfiguration.builder().minSize(10 * 1024 * 1024)
                .overallocationLimit(1.0).policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL)
                .policySpill(SpillPolicy.EXTERNAL).build();

        MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(conf, "WS");

        for( int i=0; i<10; i++ ) {
            try (Nd4jWorkspace workspace = (Nd4jWorkspace)ws.notifyScopeEntered() ) {
                INDArray bool = Nd4j.create(DataType.BOOL, 1, 10);
                INDArray dbl = Nd4j.create(DataType.DOUBLE, 1, 10);

                boolean boolAttached = bool.isAttached();
                boolean doubleAttached = dbl.isAttached();

                System.out.println(i + "\tboolAttached=" + boolAttached + ", doubleAttached=" + doubleAttached );
                System.out.println("bool: " + bool);        //java.lang.IllegalStateException: Indexer must never be null
                System.out.println("double: " + dbl);
            }
        }
    }

    @Test
    public void reproduceWorkspaceCrash(){
        WorkspaceConfiguration conf = WorkspaceConfiguration.builder().build();

        MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(conf, "WS");

        INDArray arr = Nd4j.create(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 0, 0}, new long[]{1, 10});

        for( int i=0; i<100; i++ ) {
            try(MemoryWorkspace ws2 = ws.notifyScopeEntered()) {
                System.out.println("Iteration: " + i);
                INDArray ok = arr.eq(0.0);
                ok.dup();

                INDArray crash = arr.eq(0.0).castTo(Nd4j.defaultFloatintPointType());
                crash.dup();        //Crashes here on i=1 iteration
            }
        }
    }

}
