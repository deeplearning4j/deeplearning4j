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

package org.deeplearning4j.clustering.sptree;

import org.apache.commons.lang3.time.StopWatch;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Adam Gibson
 */
public class SPTreeTest {

    @Before
    public void setUp() {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
    }

    @Test
    public void testStructure() {
        INDArray data = Nd4j.create(new float[][] {{1, 2, 3}, {4, 5, 6}});
        SpTree tree = new SpTree(data);
        try (MemoryWorkspace ws = tree.workspace().notifyScopeEntered()) {
            assertEquals(Nd4j.create(new float[]{2.5f, 3.5f, 4.5f}), tree.getCenterOfMass());
            assertEquals(2, tree.getCumSize());
            assertEquals(8, tree.getNumChildren());
            assertTrue(tree.isCorrect());
        }
    }

    @Test
    //@Ignore
    public void testLargeTree() {
        int num = 100000;
        StopWatch watch = new StopWatch();
        watch.start();
        INDArray arr = Nd4j.linspace(1, num, num, Nd4j.dataType()).reshape(num, 1);
        SpTree tree = new SpTree(arr);
        watch.stop();
        System.out.println("Tree created in " + watch);
    }

}
