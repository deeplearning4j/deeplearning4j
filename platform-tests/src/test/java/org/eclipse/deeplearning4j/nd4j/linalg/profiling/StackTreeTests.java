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
package org.eclipse.deeplearning4j.nd4j.linalg.profiling;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.primitives.StackDescriptor;
import org.nd4j.linalg.profiler.data.primitives.StackTree;
import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;

import java.util.List;

public class StackTreeTests {

    @Test
    public void testBasicTraversal() {
        Nd4j.getEnvironment().setLogNDArrayEvents(true);
        INDArray arr = Nd4j.create(10);
        StackTree stackTree = new StackTree();
        stackTree.consumeStackTrace(new StackDescriptor(Thread.currentThread().getStackTrace()),1);
        Nd4jEventLog nd4jEventLog = Nd4j.getExecutioner().getNd4jEventLog();
        List<NDArrayEvent> testBasicTraversal = nd4jEventLog.arrayEventsForStackTracePoint(StackTreeTests.class.getName(), "testBasicTraversal", 39);
        System.out.println(stackTree.renderTree(true));
    }
}
