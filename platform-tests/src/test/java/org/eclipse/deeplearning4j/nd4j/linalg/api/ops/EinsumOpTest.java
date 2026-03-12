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

package org.eclipse.deeplearning4j.nd4j.linalg.api.ops;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Einsum;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

public class EinsumOpTest extends BaseNd4jTestWithBackends {

    @Test
    public void testMatMul() {
        // "ij,jk->ik" : standard matrix multiply
        INDArray a = Nd4j.createFromArray(new float[][]{
                {1, 2},
                {3, 4}
        });
        INDArray b = Nd4j.createFromArray(new float[][]{
                {5, 6},
                {7, 8}
        });
        INDArray result = Nd4j.exec(new Einsum(new INDArray[]{a, b}, "ij,jk->ik"))[0];

        INDArray expected = a.mmul(b);
        assertEquals(expected, result);
    }

    @Test
    public void testBatchMatMul() {
        // "bij,bjk->bik" : batched matrix multiply
        INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        INDArray b = Nd4j.rand(DataType.FLOAT, 2, 4, 5);
        INDArray result = Nd4j.exec(new Einsum(new INDArray[]{a, b}, "bij,bjk->bik"))[0];

        // Verify shape
        assertArrayEquals(new long[]{2, 3, 5}, result.shape());

        // Verify values batch-by-batch using slice along first dim
        for (int batch = 0; batch < 2; batch++) {
            INDArray aBatch = a.slice(batch, 0);
            INDArray bBatch = b.slice(batch, 0);
            INDArray expectedBatch = aBatch.mmul(bBatch);
            INDArray resultBatch = result.slice(batch, 0);
            assertEquals(expectedBatch, resultBatch, "Batch " + batch + " mismatch");
        }
    }

    @Test
    public void testTranspose() {
        // "ij->ji" : transpose
        INDArray a = Nd4j.createFromArray(new float[][]{
                {1, 2, 3},
                {4, 5, 6}
        });
        INDArray result = Nd4j.exec(new Einsum(new INDArray[]{a}, "ij->ji"))[0];

        INDArray expected = a.transpose();
        assertEquals(expected, result);
    }

    @Test
    public void testTrace() {
        // "ii->" : trace (sum of diagonal)
        INDArray a = Nd4j.createFromArray(new float[][]{
                {1, 2},
                {3, 4}
        });
        INDArray result = Nd4j.exec(new Einsum(new INDArray[]{a}, "ii->"))[0];

        // trace = 1 + 4 = 5
        assertEquals(5.0f, result.getFloat(0), 1e-5);
    }

    @Test
    public void testDotProduct() {
        // "i,i->" : dot product
        INDArray a = Nd4j.createFromArray(new float[]{1, 2, 3});
        INDArray b = Nd4j.createFromArray(new float[]{4, 5, 6});
        INDArray result = Nd4j.exec(new Einsum(new INDArray[]{a, b}, "i,i->"))[0];

        // dot = 1*4 + 2*5 + 3*6 = 32
        assertEquals(32.0f, result.getFloat(0), 1e-5);
    }

    @Test
    public void testOuterProduct() {
        // "i,j->ij" : outer product
        INDArray a = Nd4j.createFromArray(new float[]{1, 2, 3});
        INDArray b = Nd4j.createFromArray(new float[]{4, 5});
        INDArray result = Nd4j.exec(new Einsum(new INDArray[]{a, b}, "i,j->ij"))[0];

        assertArrayEquals(new long[]{3, 2}, result.shape());
        // Expected: [[4,5],[8,10],[12,15]]
        assertEquals(4.0f, result.getFloat(0, 0), 1e-5);
        assertEquals(5.0f, result.getFloat(0, 1), 1e-5);
        assertEquals(8.0f, result.getFloat(1, 0), 1e-5);
        assertEquals(10.0f, result.getFloat(1, 1), 1e-5);
        assertEquals(12.0f, result.getFloat(2, 0), 1e-5);
        assertEquals(15.0f, result.getFloat(2, 1), 1e-5);
    }

    @Test
    public void testThreeInputChain() {
        // "ij,jk,kl->il" : three matrix chain multiply
        INDArray a = Nd4j.createFromArray(new float[][]{
                {1, 2},
                {3, 4}
        });
        INDArray b = Nd4j.createFromArray(new float[][]{
                {5, 6},
                {7, 8}
        });
        INDArray c = Nd4j.createFromArray(new float[][]{
                {9, 10},
                {11, 12}
        });
        INDArray result = Nd4j.exec(new Einsum(new INDArray[]{a, b, c}, "ij,jk,kl->il"))[0];

        INDArray expected = a.mmul(b).mmul(c);
        assertEquals(expected, result);
    }

    @Test
    public void testImplicitOutput() {
        // "ij,jk" without -> should produce "ik" (labels appearing exactly once, sorted)
        INDArray a = Nd4j.createFromArray(new float[][]{
                {1, 2},
                {3, 4}
        });
        INDArray b = Nd4j.createFromArray(new float[][]{
                {5, 6},
                {7, 8}
        });
        INDArray result = Nd4j.exec(new Einsum(new INDArray[]{a, b}, "ij,jk"))[0];

        INDArray expected = a.mmul(b);
        assertEquals(expected, result);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
