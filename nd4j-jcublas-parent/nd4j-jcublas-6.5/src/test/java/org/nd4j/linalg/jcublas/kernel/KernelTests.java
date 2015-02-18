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

import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 2/17/15.
 */
public class KernelTests {


    @Test
    public void testKernelLoading() throws IOException {
        String loaded = KernelFunctions.load("addi.cu", DataBuffer.FLOAT);
        CUfunction function = KernelFunctions.loadFunction(0, loaded, "add");
        assertEquals(function, KernelFunctions.loadFunction(0, loaded, "add"));


        float[] ones = new float[]{1,1};
        float[] two = new float[]{2,2};
        float[] result = new float[]{3,3};
        CUdeviceptr result2 = KernelFunctions.constructAndAlloc(2,DataBuffer.FLOAT);
        Pointer pointer = KernelFunctions.constructKernelParameters(
                Pointer.to(new int[]{2})
                ,Pointer.to(KernelFunctions.alloc(ones))
                , Pointer.to(KernelFunctions.alloc(two))
                ,Pointer.to(result2));
        KernelFunctions.invoke(2,function,pointer,result2,DataBuffer.FLOAT);

    }


}
