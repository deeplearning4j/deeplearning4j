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
import jcuda.Sizeof;
import jcuda.driver.*;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.*;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;

import static jcuda.driver.JCudaDriver.*;

/**
 * Kernel functions.
 * <p/>
 * Derived from:
 * http://www.jcuda.org/samples/JCudaVectorAdd.java
 *
 * @author Adam Gibson
 */
public class KernelFunctions {


    private static Logger log = LoggerFactory.getLogger(KernelFunctions.class);
    private static Set<String> reduceFunctions = new ConcurrentSkipListSet<>();
    public final static String NAME_SPACE = "org.nd4j.linalg.jcublas";
    public final static String DOUBLE = NAME_SPACE + ".double.functions";
    public final static String FLOAT = NAME_SPACE + ".float.functions";
    public final static String REDUCE = NAME_SPACE + ".reducefunctions";


    private KernelFunctions() {
    }


    static {
        try {
            register();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Called at initialization in the static context.
     * Registers cuda functions based on the cudafunctions.properties in the classpath
     * @throws IOException
     */
    public static void register() throws Exception {
        ClassPathResource res = new ClassPathResource("/cudafunctions.properties");
        if (!res.exists())
            throw new IllegalStateException("Please put a cudafunctions.properties in your class path");
        Properties props = new Properties();
        props.load(res.getInputStream());
        KernelFunctionLoader.getInstance().load();

        String reduceFunctionsList = props.getProperty(REDUCE);
        for(String function : reduceFunctionsList.split(","))
            reduceFunctions.add(function);

    }




    private static int sizeFor(int dataType) {
        return dataType == DataBuffer.DOUBLE ? Sizeof.DOUBLE : Sizeof.FLOAT;
    }

    /**
     * Construct and allocate a device pointer
     * @param length the length of the pointer
     * @param dType the data type to choose
     * @return the new pointer
     */
    public static CUdeviceptr constructAndAlloc(int length,int dType) {
        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, length * dType == DataBuffer.FLOAT ? Sizeof.FLOAT : Sizeof.DOUBLE);
        return deviceOutput;
    }

    /**
     * Construct kernel parameters from the given pointers.
     * Think of it as follows. If I have a standard linear operator
     * such as 2 vectors with 1 output vector, this would be 3 pointers
     * such that the first 2 are the inputs and the third one is the outputs
     * @param pointers the pointers to create parameters from
     * @return the pointer to the pointers
     */
    public static Pointer constructKernelParameters(Pointer...pointers) {
        return Pointer.to(pointers);
    }

    /**
     * Invoke a reduce
     * function with the given number of parameters
     * Invoke reduce makes a different set of assumptions
     * about how to allocate memory.
     * From Nvidia's dot product example:
     *      #define imin(a,b) (a<b?a:b)
     *
     *        const int N = 33 * 1024;
     *        const int threadsPerBlock = 256;
     *       const int blocksPerGrid =
     *        imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );
     *        dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,
     *        dev_partial_c );
     *
     *  The major thing is that the number of threads per block have to be even
     * @param threadsPerBlock the number of threads to use per block
     * @param blocks  the number of blocks to use
     * @param function   the function to invoke
     * @param kernelParameters the parameters
     */
    public static void invokeReduce(int threadsPerBlock,int blocks, CUfunction function, Pointer kernelParameters) {
        // Call the kernel function.
        //dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,dev_partial_c );
        int sharedMemSize = threadsPerBlock * (Nd4j.dataType() == DataBuffer.DOUBLE ? Sizeof.DOUBLE : Sizeof.FLOAT);
        if (threadsPerBlock <= 32)
            sharedMemSize *= 2;

        cuLaunchKernel(function,
                blocks, 1, 1,      // Grid dimension
                threadsPerBlock, 1, 1,      // Block dimension
                sharedMemSize, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );

        cuCtxSynchronize();


    }



    /**
     * Perform a reduction of the specified number of elements in the given
     * device input memory, using the given number of threads and blocks,
     * and write the results into the given output memory.
     *
     * @param size The size (number of elements)
     * @param threads The number of threads
     * @param blocks The number of blocks
     * @param deviceInput The device input memory
     * @param deviceOutput The device output memory. Its size must at least
     * be numBlocks*Sizeof.FLOAT
     */
    private static void reduce(CUfunction function,int size, int threads, int blocks,
                               Pointer deviceInput, Pointer deviceOutput) {
        //System.out.println("Reduce "+size+" elements with "+
        //    threads+" threads in "+blocks+" blocks");

        // Compute the shared memory size (as done in
        // the NIVIDA sample)
        int sharedMemSize = threads * Sizeof.FLOAT;
        if (threads <= 32)

            sharedMemSize *= 2;


        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceInput),
                Pointer.to(deviceOutput),
                Pointer.to(new int[]{size})
        );

        // Call the kernel function.
        cuLaunchKernel(function,
                blocks,  1, 1,         // Grid dimension
                threads, 1, 1,         // Block dimension
                sharedMemSize, null,   // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
    }


    /**
     * Invoke a function with the given number of parameters
     * The block size is 256 and the grid size is set with respect
     * to the number of elements now
     * @param numElements the number of
     * @param function   the function to invoke
     * @param kernelParameters the parameters
     */
    public static void invoke(int numElements, CUfunction function, Pointer kernelParameters) {
        // Call the kernel function.
        int blockSizeX = 256;
        int threads = (int) Math.ceil((double) numElements / blockSizeX);
        int sharedMemSize = threads * Nd4j.dataType() == DataBuffer.DOUBLE ? Sizeof.DOUBLE : Sizeof.FLOAT;
        if (threads <= 32)
            sharedMemSize *= 2;

        cuLaunchKernel(function,
                threads, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();


    }


    /**
     * Invoke a function with the given number of parameters
     *
     * @param numElements
     * @param function         the function to invoke
     * @param kernelParameters the parameters
     * @param deviceOutput     the output pointer
     * @return the data (either float or double array)
     */
    public static Object invoke(int numElements, CUfunction function, Pointer kernelParameters, CUdeviceptr deviceOutput, int dType) {
        // Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) numElements / blockSizeX);
        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        if (dType == DataBuffer.FLOAT) {
            float hostOutput[] = new float[numElements];
            cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
                    numElements * Sizeof.FLOAT);
            return hostOutput;

        } else {
            double hostOutput[] = new double[numElements];
            cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
                    numElements * Sizeof.DOUBLE);
            return hostOutput;

        }

    }

    /**
     * Allocate a pointer of a given data type
     *
     * @param data the data for the pointer
     * @return the pointer
     */
    public static Pointer alloc(double[] data) {
        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, data.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputA, Pointer.to(data),
                data.length * Sizeof.DOUBLE);
        return deviceInputA;
    }

    /**
     * Allocate a pointer of a given data type
     *
     * @param data the data for the pointer
     * @return the pointer
     */
    public static Pointer alloc(float[] data) {
        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, data.length * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputA, Pointer.to(data),
                data.length * Sizeof.FLOAT);
        return deviceInputA;
    }


    /**
     * Allocate a pointer of a given data type
     *
     * @param dataType the data type
     * @param length   the length of the pointer
     * @return the pointer
     */
    public static Pointer alloc(int dataType, int length) {
        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, length * sizeFor(dataType));

        return deviceInputA;
    }

    /**
     * Execute a cuda function
     *
     * @param function         the function to execute
     * @param blockSize        the block size to execute on
     * @param gridSize         the grid size to execute on
     * @param kernelParameters the kernel parameters
     */
    public static void exec(CUfunction function, int blockSize, int gridSize, Pointer kernelParameters) {
        cuLaunchKernel(function,
                blockSize, 1, 1,      // Grid dimension
                gridSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();


    }




    private static String dataFolder(int type) {
        return "/kernels/" + (type == DataBuffer.FLOAT ? "float" : "double");
    }

    //extract the source file
    private static void extract(String file, int dataType) throws IOException {

        String path = dataFolder(dataType);
        String tmpDir = System.getProperty("java.io.tmpdir");
        File dataDir = new File(tmpDir, path);
        if (!dataDir.exists())
            dataDir.mkdirs();
        ClassPathResource resource = new ClassPathResource(file);
        if (!resource.exists())
            throw new IllegalStateException("Unable to find file " + resource);
        File out = new File(tmpDir,file);
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(out));
        IOUtils.copy(resource.getInputStream(), bos);
        bos.flush();
        bos.close();

        out.deleteOnExit();

    }






    /**
     * Returns whether the given function is a reduce function
     * @param functionName the function name to check for
     * @return true if the function is a reduce function false otherwise
     */
    public static boolean isReduce(String functionName) {
        return reduceFunctions.contains(functionName);
    }




}
