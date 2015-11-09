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
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.util.ComplexUtil;
import org.nd4j.linalg.api.shape.Shape;


public class CublasPointerTests {
    @Test
    public void testAllocateArrays() {
        INDArray arr1OffsetFor = Nd4j.linspace(1,12,12).reshape(4, 3);
        //2,6,10,3,7,11
        INDArray arr1Offset = arr1OffsetFor.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        arr1Offset = Shape.toOffsetZero(arr1Offset);
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        CublasPointer p = new CublasPointer(arr1Offset,ctx);
        float[] data = new float[6];
        float[] assertion = {4,5,6,7,8,9};
        JCublas2.cublasGetVectorAsync(
                6
                , Sizeof.FLOAT
                , p.getDevicePointer().withByteOffset(arr1Offset.offset() * arr1Offset.data().getElementSize())
                , arr1Offset.elementWiseStride()
                , Pointer.to(data), 1, ctx.getOldStream());
        ctx.syncOldStream();
        for(int i = 0; i < assertion.length;i++) {
            if(assertion[i] != data[i])
                System.out.println("Failed with pointer " + p);
            assertEquals(assertion[i], data[i], 1e-1f);
        }
        ctx.destroy();
    }


    @Test
    public void testAllocInt() {
        JCudaBuffer intBuffer = KernelFunctions.alloc(new int[]{1,2,3,4,5,6});
        assertEquals(6,intBuffer.length());
        for(int i = 0; i  < intBuffer.length(); i++) {
            assertEquals(i + 1,intBuffer.getInt(i));
        }

        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        CublasPointer pointer = new CublasPointer(intBuffer,ctx);
        pointer.copyToHost();
        for(int i = 0; i  < intBuffer.length(); i++) {
            assertEquals(i + 1,intBuffer.getInt(i));
        }


    }

    @Test
    public void testVectorAlongDimension() throws Exception {
        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2}, 'c');
        INDArray column = arr.getColumn(1);
        INDArray otherColumnAssertion = column.dup();
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        CublasPointer p = new CublasPointer(column,ctx);
        p.copyToHost();
        assertEquals(otherColumnAssertion, column);
        p.close();
        ctx.destroy();
    }

    @Test
    public void testTwoByTwoBuffer() throws Exception {
        IComplexNDArray arr = Nd4j.createComplex(ComplexUtil.complexNumbersFor(new double[]{2, 6}), new int[]{2, 1});
        IComplexNDArray dup = arr.dup();
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        CublasPointer pointer = new CublasPointer(arr,ctx);
        pointer.copyToHost();
        assertEquals(dup,arr);
        pointer.close();
        ctx.destroy();
    }


    @Test
    public void testBufferPointer() throws Exception {
        DataBuffer buff = Nd4j.createBuffer(new double[]{1});
        DataBuffer clone = buff.dup();
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        CublasPointer pointer  = new CublasPointer((JCudaBuffer) buff,ctx);
        pointer.copyToHost();
        pointer.close();
        assertEquals(clone,buff);

        INDArray arr = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});
        System.err.println("arr=" + arr);
        INDArray brr = Nd4j.create(new float[]{5,6},new int[]{1,2});
        System.err.println("brr = " + brr);
        INDArray row = arr.getRow(0);
        CublasPointer pointer2 = new CublasPointer(row,ctx);
        pointer2.copyToHost();
        ctx.destroy();

    }

    @Test
    public void testAllocateAndCopyBackToHostC() throws Exception {
        Nd4j.factory().setOrder('c');
        INDArray test = Nd4j.rand(5, 5);
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        CublasPointer p = new CublasPointer(test,ctx);

        CublasPointer p1 = new CublasPointer((JCudaBuffer)test.data(),ctx);

        p.copyToHost();
        p1.copyToHost();

        assertEquals(p.getBuffer(), p1.getBuffer());
        assertArrayEquals(p.getBuffer().asBytes(), p1.getBuffer().asBytes());

        p.close();
        p1.close();
        ctx.destroy();
    }

    @Test
    public void testAllocateAndCopyBackToHost() throws Exception {

        INDArray test = Nd4j.rand(5, 5);
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        CublasPointer p = new CublasPointer(test,ctx);
        CublasPointer p1 = new CublasPointer((JCudaBuffer)test.data(),ctx);

        p.copyToHost();
        p1.copyToHost();

        assertEquals(p.getBuffer(), p1.getBuffer());
        assertArrayEquals(p.getBuffer().asBytes(), p1.getBuffer().asBytes());

        p.close();
        p1.close();
        ctx.destroy();
    }


    @Test
    public void testColumnCopy() throws Exception {
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        INDArray mat = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray column = mat.getColumn(1);
        INDArray columnDup = column.dup();
        CublasPointer copy = new CublasPointer(column,ctx);
        copy.getDevicePointer();
        copy.copyToHost();
        copy.close();
        assertEquals(columnDup, column);
        ctx.destroy();


    }



    @Test
    public void testColumnCopyCOrdering() throws Exception {
        Nd4j.factory().setOrder('c');
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        INDArray mat = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray column = mat.getColumn(1);
        INDArray columnDup = column.dup();
        CublasPointer copy = new CublasPointer(column,ctx);
        copy.getDevicePointer();
        copy.copyToHost();
        copy.close();
        assertEquals(columnDup, column);
        Nd4j.factory().setOrder('f');
        ctx.destroy();

    }

    @Test
    public void testSlicePointers() throws Exception {
        INDArray arr = Nd4j.create(5,5);
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        JCudaBuffer buffer = (JCudaBuffer) arr.data();
        for(int i = 0; i < arr.slices(); i++) {
            CublasPointer pointer = new CublasPointer(arr.slice(i),ctx);
            pointer.copyToHost();
            pointer.close();
        }
        ctx.destroy();
    }


}
