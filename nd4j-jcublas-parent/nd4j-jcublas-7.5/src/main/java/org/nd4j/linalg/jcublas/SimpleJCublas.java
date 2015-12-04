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

package org.nd4j.linalg.jcublas;



import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;


/**
 * Simple abstraction for jcublas operations
 *
 * @author mjk
 * @author Adam Gibson
 */
public class SimpleJCublas {

    private static boolean init = false;


    static {
        try {
            init();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Assert that the data buffer for each ndarray
     * is a cuda buffer
     * @param buffer the arrays to tests
     */
    public static void assertCudaBuffer(INDArray... buffer) throws Exception {
        for (INDArray b1 : buffer)
            if (!(b1.data() instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }

    /**
     * Assert that the data buffer for each ndarray
     * is a cuda buffer
     * @param buffer the arrays to tests
     */
    public static void assertCudaBuffer(DataBuffer... buffer) throws Exception {
        for (DataBuffer b1 : buffer)
            if (!(b1 instanceof JCudaBuffer))
                throw new IllegalArgumentException("Unable to allocate pointer for buffer of type " + buffer.getClass().toString());
    }






    /**
     * Initialize JCublas2. Only called once
     */
    public static synchronized void init() throws Exception {
        if (init)
            return;





        init = true;
    }





}
