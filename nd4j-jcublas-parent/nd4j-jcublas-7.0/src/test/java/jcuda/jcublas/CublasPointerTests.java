/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package jcuda.jcublas;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;

public class CublasPointerTests {

    @Test
    public void testCopyMatrix() {
        INDArray twoByThree = Nd4j.linspace(1, 784, 784).reshape(28, 28);
        INDArray copy = Nd4j.create(28, 28);
        Nd4j.getBlasWrapper().copy(twoByThree.linearView(), copy.linearView());
    }


    @Test
    public void testVectorAlongDimension() throws Exception {
        INDArray arr = Nd4j.create(new double[]{1,2,3,4},new int[]{2,2},'c');
        INDArray column = arr.getColumn(1);
        INDArray otherColumnAssertion = column.dup();
        CublasPointer p = new CublasPointer(column);
        p.copyToHost();
        assertEquals(otherColumnAssertion,column);
        p.close();
    }


    @Test
    public void testColumnStd() throws Exception {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        CublasPointer p = new CublasPointer(twoByThree);
        p.close();
        INDArray columnStd = twoByThree.std(0);
        INDArray assertion = Nd4j.create(new float[]{173.78147196982766f, 173.78147196982766f, 173.78147196982766f, 173.78147196982766f});
        assertEquals(assertion, columnStd);

    }


    @Test
    public void testAllocateAndCopyBackToHost() throws Exception {

        INDArray test = Nd4j.rand(5, 5);

        CublasPointer p = new CublasPointer(test);
        CublasPointer p1 = new CublasPointer((JCudaBuffer)test.data());

        p.copyToHost();
        p1.copyToHost();

        assertEquals(p.getBuffer(), p1.getBuffer());
        assertArrayEquals(p.getBuffer().asBytes(), p1.getBuffer().asBytes());

        p.close();
        p1.close();
    }


    @Test
    public void testColumnCopy() throws Exception {
        INDArray mat = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray column = mat.getColumn(1);
        INDArray columnDup = column.dup();
        CublasPointer copy = new CublasPointer(column);
        copy.getDevicePointer();
        copy.copyToHost();
        copy.close();
        assertEquals(columnDup, column);


    }

    @Test
    public void testColumnCopyCOrdering() throws Exception {
        Nd4j.factory().setOrder('c');
        INDArray mat = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray column = mat.getColumn(1);
        INDArray columnDup = column.dup();
        CublasPointer copy = new CublasPointer(column);
        copy.getDevicePointer();
        copy.copyToHost();
        copy.close();
        assertEquals(columnDup, column);
        Nd4j.factory().setOrder('f');

    }
    /**
     * Test that when using offsets, the data is not corrupted
     * @throws Exception
     */
    @Test
    public void testRowOffsettingCopyBackToHost() throws Exception {
        for(int i = 1; i < 100; i++) {
            INDArray test = Nd4j.rand(i,i);

            INDArray testDupe = test.dup();

            // Create an offsetted set of pointers and copy to and from device, this should copy back to the same offset it started at.
            for(int x = 0; x < i; x++) {
                INDArray test2 = test.getRow(x);
                CublasPointer p1 = new CublasPointer(test2);
                p1.copyToHost();
                p1.close();
            }


            assertArrayEquals(testDupe.data().asBytes(), test.data().asBytes());
        }
    }

}
