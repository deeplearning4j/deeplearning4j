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

package org.nd4j.linalg.jcublas.kernel;


import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.utils.KernelLauncher;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentSkipListSet;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Kernel functions.
 * <p/>
 * Derived from:
 * http://www.jcuda.org/samples/JCudaVectorAdd.java
 *
 * @author Adam Gibson
 */
public class KernelFunctions {

    public final static String NAME_SPACE = "org.nd4j.linalg.jcuda.jcublas";
    public final static String DOUBLE = NAME_SPACE + ".double.functions";
    public final static String FLOAT = NAME_SPACE + ".float.functions";
    public final static String REDUCE = NAME_SPACE + ".reducefunctions";
    public final static String SHARED_MEM_KEY = NAME_SPACE + ".sharedmem";
    public final static String THREADS_KEY = NAME_SPACE + ".threads";
    public final static String BLOCKS_KEY = NAME_SPACE + ".blocks";
    public static int SHARED_MEM = 512;
    public static int THREADS = 128;
    public static int BLOCKS = 512;
    private static Set<String> reduceFunctions = new ConcurrentSkipListSet<>();


    private KernelFunctions() {}


    static {
        try {
            register();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Called at initialization in the static context.
     * Registers cuda functions based on
     * the cudafunctions.properties in the classpath
     *
     * @throws IOException
     */
    public static void register() throws Exception {
        ClassPathResource res = new ClassPathResource("/cudafunctions.properties", KernelFunctions.class.getClassLoader());
        if (!res.exists())
            throw new IllegalStateException("Please put a cudafunctions.properties in your class path");
        Properties props = new Properties();
        props.load(res.getInputStream());
        KernelFunctionLoader.getInstance().load();

        String reduceFunctionsList = props.getProperty(REDUCE);
        for (String function : reduceFunctionsList.split(","))
            reduceFunctions.add(function);

        SHARED_MEM = Integer.parseInt(props.getProperty(SHARED_MEM_KEY, "512"));
        THREADS = Integer.parseInt(props.getProperty(THREADS_KEY, "128"));
        BLOCKS = Integer.parseInt(props.getProperty(BLOCKS_KEY, "64"));

    }



    /**
     * Invoke a function with the given number of parameters
     *
     * @param blocks           the number of blocks to launch the kernel
     * @param threadsPerBlock  the number of threads per block
     * @param kernelParameters the parameters
     * @param dataType         the data type to use
     */
    public static  void invoke(int blocks, int threadsPerBlock, String functionName,String dataType,Object...kernelParameters) {

        // Call the kernel function.
        //dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,dev_partial_c );
        CUstream stream = ContextHolder.getInstance().getStream();
        int sharedMemSize = threadsPerBlock * (dataType.equals("float") ? Sizeof.FLOAT : Sizeof.DOUBLE) * 2;
        KernelLauncher launcher = KernelFunctionLoader.launcher(functionName, dataType);
        if(launcher == null)
            throw new IllegalArgumentException("Launcher for function " + functionName + " and data type " + dataType + " does not exist!");

        launcher.forFunction(functionName + "_" + dataType)
                .setBlockSize(threadsPerBlock,1,1)
                .setGridSize(blocks,1,1).setStream(stream)
                .setSharedMemSize(sharedMemSize)
                .call(kernelParameters);

        ContextHolder.syncStream();
    }


    /**
     * Allocate a pointer of a given data type
     *
     * @param data the data for the pointer
     * @return the pointer
     */
    public static JCudaBuffer alloc(double[] data) {
        // Allocate the device input data, and copy the
        // host input data to the device
        JCudaBuffer doubleBuffer = new CudaDoubleDataBuffer(data);
        return doubleBuffer;
    }

    /**
     * Allocate a pointer of a given data type
     *
     * @param data the data for the pointer
     * @return the pointer
     */
    public static JCudaBuffer alloc(float[] data) {
        // Allocate the device input data, and copy the
        // host input data to the device
        JCudaBuffer floatBuffer = new CudaFloatDataBuffer(data);
        return floatBuffer;
    }

}
