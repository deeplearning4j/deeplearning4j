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

import jcuda.Pointer;
import jcuda.Sizeof;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.util.ComplexUtil;
import org.nd4j.linalg.api.shape.Shape;


public class CublasPointerTests {
    @Test
    public void testAllocateArrays() {
        INDArray arr1OffsetFor = Nd4j.linspace(1,12,12).reshape(4, 3);
        //2,6,10,3,7,11
        INDArray arr1Offset = arr1OffsetFor.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        arr1Offset = Shape.toOffsetZero(arr1Offset);
        CublasPointer p = new CublasPointer(arr1Offset);
        String s = p.toString();
        float[] data = new float[6];
        float[] assertion = {4,5,6,7,8,9};
        JCublas2.cublasGetVector(
                6
                , Sizeof.FLOAT
                ,p.getDevicePointer().withByteOffset(arr1Offset.offset() * arr1Offset.data().getElementSize())
                ,arr1Offset.majorStride()
                , Pointer.to(data),1);
        for(int i = 0; i < assertion.length;i++)
            assertEquals(data[i],assertion[i],1e-1f);
    }

    @Test
    public void testVectorAlongDimension() throws Exception {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2}, 'c');
        INDArray column = arr.getColumn(1);
        INDArray otherColumnAssertion = column.dup();
        CublasPointer p = new CublasPointer(column);
        p.copyToHost();
        assertEquals(otherColumnAssertion,column);
        p.close();
    }

    @Test
    public void testTwoByTwoBuffer() throws Exception {
        IComplexNDArray arr = Nd4j.createComplex(ComplexUtil.complexNumbersFor(new double[]{2, 6}), new int[]{2, 1});
        IComplexNDArray dup = arr.dup();
        CublasPointer pointer = new CublasPointer(arr);
        pointer.copyToHost();
        assertEquals(dup,arr);
        pointer.close();

    }


    @Test
    public void testBufferPointer() throws Exception {
        DataBuffer buff = Nd4j.createBuffer(new double[]{1});
        DataBuffer clone = buff.dup();
        CublasPointer pointer  = new CublasPointer((JCudaBuffer) buff);
        pointer.copyToHost();
        pointer.close();
        assertEquals(clone,buff);

        INDArray arr = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});
        System.err.println("arr=" + arr);
        INDArray brr = Nd4j.create(new float[]{5,6},new int[]{1,2});
        System.err.println("brr = " + brr);
        INDArray row = arr.getRow(0);
        CublasPointer pointer2 = new CublasPointer(row);
        pointer2.copyToHost();

    }

    @Test
    public void testAllocateAndCopyBackToHostC() throws Exception {
        Nd4j.factory().setOrder('c');
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

    @Test
    public void testSlicePointers() throws Exception {
        INDArray arr = Nd4j.create(5,5);
        JCudaBuffer buffer = (JCudaBuffer) arr.data();
        for(int i = 0; i < arr.slices(); i++) {
            CublasPointer pointer = new CublasPointer(arr.slice(i));
            pointer.copyToHost();
            pointer.close();
        }
    }


}
