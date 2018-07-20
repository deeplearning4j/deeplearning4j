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

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
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
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testStructure() {
        INDArray data = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        SpTree tree = new SpTree(data);
        try (MemoryWorkspace ws = tree.workspace().notifyScopeEntered()) {
            assertEquals(Nd4j.create(new double[]{2.5, 3.5, 4.5}), tree.getCenterOfMass());
            assertEquals(2, tree.getCumSize());
            assertEquals(8, tree.getNumChildren());
            assertTrue(tree.isCorrect());
        }
    }

    @Test
    @Ignore
    public void testLargeTree() {
        int num = 100000;
        INDArray arr = Nd4j.linspace(1, num, num).reshape(num, 1);
        SpTree tree = new SpTree(arr);
    }

}
