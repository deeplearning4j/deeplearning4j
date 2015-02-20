/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.jcublas.kernel;



import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;
/**
 * Created by agibsonccc on 2/17/15.
 */
public class KernelTests {
    @Test
    public void testStridedAdditionInvocationBuffer() {
        float[] ones = new float[]{1,1};
        float[] two = new float[]{2,2};
        float[] result = new float[]{3,3};

        JCudaBuffer buff1 = (JCudaBuffer) Nd4j.createBuffer(ones);
        JCudaBuffer buff2 = (JCudaBuffer) Nd4j.createBuffer(two);



        Pointer onesP = Pointer.to(buff1.pointer());
        Pointer twoP = Pointer.to(buff2.pointer());

        Pointer kernelParameters = KernelFunctions.constructKernelParameters(
                //number of elements
                Pointer.to(new int[]{ones.length})
                ,onesP
                , twoP
                ,Pointer.to(new int[]{1})
                ,Pointer.to(new int[]{1}));
        KernelFunctions.invoke2d(2,KernelFunctions.getFunction("add_strided","float"),kernelParameters);
        assertArrayEquals(result, buff2.asFloat(), 1e-1f);
        buff1.destroy();
        buff2.destroy();

    }

    @Test
    public void testStridedAdditionInvocationBufferSkip() {
        float[] ones = new float[]{1,1};
        float[] two = new float[]{2,2};
        float[] result = new float[]{3,2};

        JCudaBuffer buff1 = (JCudaBuffer) Nd4j.createBuffer(ones);
        JCudaBuffer buff2 = (JCudaBuffer) Nd4j.createBuffer(two);



        Pointer onesP = Pointer.to(buff1.pointer());
        Pointer twoP = Pointer.to(buff2.pointer());

        Pointer kernelParameters = KernelFunctions.constructKernelParameters(
                //number of elements
                Pointer.to(new int[]{ones.length})
                ,onesP
                , twoP
                ,Pointer.to(new int[]{2})
                ,Pointer.to(new int[]{2}));
        KernelFunctions.invoke2d(2,KernelFunctions.getFunction("add_strided","float"),kernelParameters);
        assertArrayEquals(result, buff2.asFloat(), 1e-1f);

    }


    @Test
    public void testStridedAdditionInvocation() {
        float[] ones = new float[]{1,1};
        float[] two = new float[]{2,2};
        Pointer onesP = KernelFunctions.alloc(ones);
        Pointer twoP = KernelFunctions.alloc(two);
        Pointer kernelParameters = KernelFunctions.constructKernelParameters(
                //number of elements
                Pointer.to(new int[]{ones.length})
                ,Pointer.to(onesP)
                , Pointer.to(twoP),Pointer.to(new int[]{1}),Pointer.to(new int[]{1}));
        KernelFunctions.invoke(2, KernelFunctions.getFunction("add_strided", "float"), kernelParameters);

    }

    @Test
    public void addInPlace() throws IOException {

        float[] ones = new float[]{1,1};
        float[] two = new float[]{2,2};
        float[] result = new float[]{3,3};

        JCudaBuffer buff1 = (JCudaBuffer) Nd4j.createBuffer(ones);
        JCudaBuffer buff2 = (JCudaBuffer) Nd4j.createBuffer(two);



        Pointer p1 = Pointer.to(buff1.pointer());
        Pointer p2 = Pointer.to(buff2.pointer());
        Pointer pointer = KernelFunctions.constructKernelParameters(
                //number of elements
                Pointer.to(new int[]{2})
                , p1
                , p2, p2);

        KernelFunctions.invoke(2,KernelFunctions.getFunction("add","float"),pointer);
        assertArrayEquals(result,buff2.asFloat(),1e-1f);

    }


}
