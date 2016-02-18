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



import jcuda.runtime.JCuda;
import jcuda.utils.KernelLauncher;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaIntDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentSkipListSet;

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


    private KernelFunctions() {}


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
        SHARED_MEM = Integer.parseInt(props.getProperty(SHARED_MEM_KEY, "512"));
        THREADS = Integer.parseInt(props.getProperty(THREADS_KEY, "128"));
        BLOCKS = Integer.parseInt(props.getProperty(BLOCKS_KEY, "64"));

    }

    /**
     * Invoke a function
     * @param metrics
     * @param functionName
     * @param dataType
     * @param cudaContext
     * @param kernelParameters
     */
    public static  void invoke(GpuMetrics metrics, boolean sync, String moduleName, String functionName, DataBuffer.Type dataType, CudaContext cudaContext, Object...kernelParameters) {
        // Call the kernel function.
        int sharedMemSize = metrics.getSharedMemory();
        KernelLauncher launcher = KernelFunctionLoader.launcher(moduleName, dataType);
        if(launcher == null)
            throw new IllegalArgumentException("Launcher for function " + functionName + " and data type " + dataType + " does not exist!");

        launcher.forFunction(functionName)
                .setBlockSize(metrics.getBlockSize(),1,1)
                .setGridSize(metrics.getGridSize(),1,1).setStream(cudaContext.getStream())
                .setSharedMemSize(sharedMemSize)
                .call(kernelParameters);
        cudaContext.startNewEvent();
     //   if(sync)
        // TODO: we always sync for now, later sync will be removed from this place
        cudaContext.syncStream();


    }

    /**
     * Invoke a function
     * @param metrics
     * @param functionName the name of the module to load
     * @param dataType
     * @param cudaContext
     * @param kernelParameters
     */
    public static  void invoke(GpuMetrics metrics, boolean sync,String functionName,DataBuffer.Type dataType,CudaContext cudaContext,Object...kernelParameters) {
        // FIXME: this is bad AND ugly, remove this crappy shit at some point
        String functionName2 = functionName + StringUtils.capitalize(dataType.toString().toLowerCase()); // KernelLauncher.FUNCTION_NAME + "_" + dataType;
        invoke(metrics, sync, functionName, functionName2, dataType, cudaContext, kernelParameters);
    }


    /**
     * Allocate a pointer of a given data type
     *
     * @param data the data for the pointer
     * @return the pointer
     */
    public static JCudaBuffer alloc(int[] data) {
        // Allocate the device input data, and copy the
        // host input data to the device
        JCudaBuffer doubleBuffer = new CudaIntDataBuffer(data);
        return doubleBuffer;
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
