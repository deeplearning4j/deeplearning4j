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

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.JCuda.cudaHostGetDevicePointer;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.buffer.allocation.PinnedMemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.ComplexUtil;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.NioUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

@net.jcip.annotations.NotThreadSafe
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
        if(!(ContextHolder.getInstance().getMemoryStrategy() instanceof PinnedMemoryStrategy)) {
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
    @Ignore
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
    public void testHostMemory() {
        Pointer hostPointer = Pointer.to(new double[]{1});
        JCuda.cudaHostAlloc(hostPointer,Sizeof.DOUBLE , JCuda.cudaHostAllocMapped);
        Pointer devicePointer = new Pointer();
        JCuda.cudaHostGetDevicePointer(devicePointer,hostPointer, 0);
        JCuda.cudaFreeHost(hostPointer);
    }

    @Test
    public void testPortableHostMemory() {
        Pointer hostPointer = Pointer.to(new double[]{1});
        JCuda.cudaHostAlloc(hostPointer,Sizeof.DOUBLE ,JCuda.cudaHostAllocMapped | JCuda.cudaHostAllocPortable);
        Pointer devicePointer = new Pointer();
        JCuda.cudaHostGetDevicePointer(devicePointer,hostPointer, 0);
        JCuda.cudaFreeHost(hostPointer);
    }

    @Test
    @Ignore
    public void testHostPrinting() {
        // Set the flag indicating that mapped memory will be used
        cudaSetDeviceFlags(cudaDeviceMapHost);

        // Allocate mappable host memory
        int n = 5;
        Pointer host = new Pointer();
        cudaHostAlloc(host, n * Sizeof.INT, cudaHostAllocMapped);

        // Create a device pointer mapping the host memory
        Pointer device = new Pointer();
        cudaHostGetDevicePointer(device, host, 0);

        // Obtain a ByteBuffer for accessing the data in the host
        // pointer. Modifications in this ByteBuffer will be
        // visible in the device memory.
        ByteBuffer byteBuffer = host.getByteBuffer(0, n * Sizeof.INT);

        // Set the byte order of the ByteBuffer
        byteBuffer.order(ByteOrder.nativeOrder());
        FloatBuffer buf = byteBuffer.asFloatBuffer();
        for(int i = 0; i < n; i++) {
            buf.put(i,(float) i + 1);
        }


        DataBuffer buffer = Nd4j.createBuffer(new float[]{1, 2, 3, 4, 5});
        ByteBuffer nio = buffer.asNio();
        nio.order(ByteOrder.nativeOrder());
        FloatBuffer buf2 = nio.asFloatBuffer();
        for(int i = 0; i < n; i++) {
            System.out.println("Buffer input " + buf2.get(i));
        }
        NioUtil.copyAtStride(5, NioUtil.BufferType.FLOAT,nio,0,1,byteBuffer,0,1);
        JCublas.printVector(n,device);
        System.out.println();

    }

    @Test
    public void testPrinting() throws Exception {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        int[] buff = PointerUtil.toShapeInfoBuffer(arr);
        DataBuffer shapeBuffer = Nd4j.createBuffer(buff);
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        KernelFunctionLoader.printBuffer((JCudaBuffer) shapeBuffer,ctx);
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
        INDArray assertionRow = row.dup();
        CublasPointer pointer2 = new CublasPointer(row,ctx);
        pointer2.copyToHost();
        ctx.destroy();
        assertEquals(assertionRow,row);
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
