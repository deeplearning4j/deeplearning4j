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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

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
    private static Map<String, CUfunction> functions = new ConcurrentHashMap<>();
    private static Map<Integer,CUcontext> devices = new ConcurrentHashMap<>();

    public final static String NAME_SPACE = "org.nd4j.linalg.jcublas";
    public final static String DOUBLE = NAME_SPACE + ".double.functions";
    public final static String FLOAT = NAME_SPACE + ".float.functions";

    private KernelFunctions() {
    }


    static {
        try {
            register();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Called at initialization in the static context.
     * Registers cuda functions based on the cudafunctions.properties in the classpath
     * @throws IOException
     */
    public static void register() throws IOException {
        ClassPathResource res = new ClassPathResource("/cudafunctions.properties");
        if (!res.exists())
            throw new IllegalStateException("Please put a cudafunctions.properties in your class path");
        Properties props = new Properties();
        props.load(res.getInputStream());
        log.info("Registering cuda functions...");
        String d = props.getProperty(DOUBLE);
        if (d != null) {
            String[] split = d.split(",");
            log.info("Found functions for double" + d);
            for (String s : split) {
                String loaded = KernelFunctions.load("/kernels/double/" + s + ".cu", DataBuffer.DOUBLE);
                KernelFunctions.loadFunction(loaded,s,"double");
            }


        }

        String f = props.getProperty(FLOAT);
        if (f != null) {
            String[] split = f.split(",");
            log.info("Found functions for float" + d);

            for (String s : split) {
                String loaded = KernelFunctions.load("/kernels/float/" + s + ".cu", DataBuffer.FLOAT);
                KernelFunctions.loadFunction(loaded,s,"float");
            }
        }
    }

    /**
     * Get the cuda function of the given name and data type
     * @param name the name of the function
     * @param dType the data type (float or double)
     * @return the given function or null
     */
    public static CUfunction getFunction(String name,String dType) {
        return functions.get(name + "_" + dType);
    }


    public static void initDevices() {
        if(devices.containsKey(0))
            return;
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        devices.put(0,context);
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
     * Invoke a function with the given number of parameters
     *
     * @param numElements the number of
     * @param function   the function to invoke
     * @param kernelParameters the parameters
     */
    public static void invoke(int numElements, CUfunction function, Pointer kernelParameters) {
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

    /**
     * Load the given file
     *
     * @param fileName the file name
     * @param dataType the data type
     * @throws IOException
     */
    public static String load(String fileName, int dataType) throws IOException {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        initDevices();
        // Create the PTX file by calling the NVCC
        String ptxFileName = preparePtxFile(fileName, dataType);


        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        return ptxFileName;


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
     * Load the function
     * @param ptxFileName  the ptx file name
     * @param functionName the function name to use as a handle
     * @param dataType the data type(float or double) to operate on
     */
    public static CUfunction loadFunction(String ptxFileName, String functionName,String dataType) {
        if (functions.containsKey(functionName))
            return functions.get(functionName);
        // Initialize the driver and create a context for the first device.
        initDevices();
        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "add" function.
        CUfunction function = new CUfunction();
        String name = functionName + "_" + dataType;
        try {
            cuModuleGetFunction(function, module, name);
        }catch(Exception e) {
            throw new RuntimeException("Function " + name + " not found!");
        }
        functions.put(name, function);

        return function;

    }


    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     * <p/>
     * <p/>
     * Note that you may run in to an error akin to:
     * Unsupported GCC version
     * <p/>
     * At your own risk, comment the lines under:
     * /usr/local/cuda-$VERSION/include/host_config.h
     * <p/>
     * #if defined(__GNUC__)
     * <p/>
     * if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8)
     * #error -- unsupported GNU version! gcc 4.9 and up are not supported!
     * <p/>
     * #endif /* __GNUC__> 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 8)
     * <p/>
     * #endif  __GNUC__
     * <p/>
     * This will allow you to bypass the compiler restrictions. Again, do so at your own risk.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static String preparePtxFile(String cuFileName, int dataType) throws IOException {


        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) {
            endIndex = cuFileName.length() - 1;
        }

        String path = dataFolder(dataType);
        String tmpDir = System.getProperty("java.io.tmpdir");
        File dataDir = new File(tmpDir, path);


        String ptxFileName = tmpDir + cuFileName.substring(0, endIndex + 1) + "ptx";
        File ptxFile = new File(dataDir, ptxFileName);
        if (ptxFile.exists()) {
            return ptxFileName;
        } else
            extract(cuFileName, dataType);


        File cuFile = new File(tmpDir,cuFileName);
        if (!cuFile.exists())
            throw new IOException("Input file not found: " + cuFileName);


        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        String command = "nvcc " + modelString + " -ptx " + cuFile.getPath() + " -o " + ptxFileName;

        log.info("Executing " + command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
                new String(toByteArray(process.getErrorStream()));
        String outputMessage =
                new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(
                    "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0) {
            log.info("nvcc process exitValue " + exitValue);
            log.info("errorMessage:\n" + errorMessage);
            log.info("outputMessage:\n" + outputMessage);
            throw new IOException(
                    "Could not create .ptx file: " + errorMessage);
        }

        log.info("Finished creating PTX file");
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
            throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true) {
            int read = inputStream.read(buffer);
            if (read == -1) {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }


}
