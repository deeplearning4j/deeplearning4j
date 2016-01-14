/*
 * JCudaUtils - Utilities for JCuda
 * http://www.jcuda.org
 *
 * Copyright (c) 2010 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

package jcuda.utils;

import com.google.common.io.ByteStreams;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.dim3;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.naming.Context;
import java.io.*;

import static jcuda.driver.JCudaDriver.*;

/**
 * This is a utility class that simplifies the setup and launching
 * of CUDA kernels using the JCuda Driver API. <br />
 * <br />
 * Instances of this class may be created using one of the following
 * methods: <br />
 * <ul>
 *   <li>
 *     {@link KernelLauncher#compile(String, String, String...)} will
 *     compile a kernel from a String containing the CUDA source code
 *   </li>
 *   <li>
 *     {@link KernelLauncher#create(String, String, String...)} will
 *     create a kernel for a function that is contained in a CUDA
 *     source file
 *   </li>
 *   <li>
 *     {@link KernelLauncher#load(String, String)} will load a kernel from
 *     a PTX or CUBIN (CUDA binary) file.
 *   </li>
 *   <li>
 *     {@link KernelLauncher#load(InputStream, String)} will load a kernel
 *     from PTX- or CUBIN data which is provided via an InputStream
 *     (useful for packaging PTX- or CUBIN files into JAR archives)<br />
 *   </li>
 * </ul>
 *
 * <br />
 * These instances may then be used to call a kernel function with
 * the {@link KernelLauncher#call(Object...)} method. The actual
 * kernel function arguments which are passed to this method
 * will be set up automatically, and aligned appropriately for
 * their respective size.<br />
 * <br />
 * The setup of the execution may be performed similarly as the invocation
 * of a kernel when using the Runtime API in C. Such a call has the
 * form<br />
 * <code>
 * &nbsp;&nbsp;&nbsp;&nbsp;kernel&lt;&lt;&lt;gridDim, blockDim,
 * sharedMemorySize, stream&gt;&gt;&gt;(...);
 * </code>
 * <br />
 * where
 * <ul>
 *   <li>
 *     <b>gridDim</b> is a dim3 which specifies the number of blocks per
 *     grid
 *   </li>
 *   <li>
 *     <b>blockDim</b> is a dim3 that specifies the number of threads
 *     per block
 *   </li>
 *   <li>
 *     <b>sharedMemorySize</b> is the size of the shared memory for
 *     the kernel
 *   </li>
 *   <li>
 *     <b>stream</b> is a stream for asynchronous kernel execution
 *   </li>
 * </ul>
 * Similarly, the KernelLauncher allows specifying these parameters
 * in the {@link KernelLauncher#setup(dim3, dim3, int, CUstream)}
 * method: <br />
 * <br />
 * <code>
 * &nbsp;&nbsp;&nbsp;&nbsp;kernelLauncher.setup(gridDim,
 * blockDim, sharedMemorySize, stream).call(...);
 * </code>
 * <br />
 * <br />
 * When default values for some of the parameters should be used,
 * one of the overloaded versions of the setup method may be called:
 * <br />
 * <br />
 * <code>
 * &nbsp;&nbsp;&nbsp;&nbsp;kernelLauncher.setup(gridDim,
 * blockDim).call(kernel);
 * </code>
 * <br />
 * <br />
 * The parameters may also be set individually:<br />
 * <br />
 * <code>
 * &nbsp;&nbsp;&nbsp;&nbsp;kernelLauncher.setGridSize(gridSize);<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;kernelLauncher.setBlockSize(blockSize);<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;kernelLauncher.call(...);
 * </code>
 * <br />
 */
public class KernelLauncher {



    public final static String FUNCTION_NAME = "transform";


    /**
     * The logger used in this class
     */
    private static final Logger logger = LoggerFactory.getLogger(KernelLauncher.class.getName());

    /**
     * The path prefix, containing the path to the NVCC compiler.
     * Not required if the path to the NVCC is present in an
     * environment variable.
     */
    private static String compilerPath = "";

    /**
     * The number of the device which should be used by the
     * KernelLauncher
     */
    private  int deviceNumber = 0;

    /**
     * Set the path to the NVCC compiler. For example: <br />
     * <code>setCompilerPath("C:/CUDA/bin");</code>
     * <br />
     * By default, this path is empty, assuming that the compiler
     * is in a path that is visible via an environment variable.
     *
     * @param path The path to the NVCC compiler.
     */
    public static void setCompilerPath(String path)
    {
        if (path == null)
        {
            compilerPath = "";
        }
        compilerPath = path;
        if (!compilerPath.endsWith(File.separator))
        {
            compilerPath += File.separator;
        }
    }

    /**
     * Set the number (index) of the device which should be used
     * by the KernelLauncher
     *
     * @param number The number of the device to use
     * @throws CudaException If number < 0 or number >= deviceCount
     */
    public  void setDeviceNumber(int number)
    {
        int count[] = new int[1];
        cuDeviceGetCount(count);
        if (number < 0)
        {
            throw new CudaException(
                    "Invalid device number: " + number + ". "+
                            "There are only " + count[0] + " devices available");
        }
        deviceNumber = number;
    }

    /**
     * Create a new KernelLauncher for the function with the given
     * name, that is defined in the given source code. <br />
     * <br />
     * The source code is stored in a temporary .CU CUDA source file,
     * and a PTX file is compiled from this source file using the
     * NVCC (NVIDIA CUDA C Compiler) in a separate process.
     * The optional nvccArguments are passed to the NVCC.<br />
     * <br />
     * The NVCC has to be in a visible directory. E.g. for Windows, the
     * NVCC.EXE has to be in a directory that is contained in the PATH
     * environment variable. Alternatively, the path to the NVCC may
     * be specified by calling {@link KernelLauncher#setCompilerPath(String)}
     * with the respective path. <br />
     * <br />
     * <u><b>Note</b></u>: In order to make the function accessible
     * by the name it has in the source code, the function has to
     * be declared as an <code><u>extern "C"</u></code> function: <br />
     *  <br />
     * <code>
     * <b>extern "C"</b><br />
     * __global__ void functionName(...)<br />
     * {<br />
     *     ... <br />
     * }<br />
     * </code>
     *
     *
     * @see KernelLauncher#create(String, String, String...)
     * @see KernelLauncher#create(String, String, boolean, String...)
     *
     * @param sourceCode The source code containing the function
     * @param functionName The name of the function.
     * @param nvccArguments Optional arguments for the NVCC
     * @return The KernelLauncher for the specified function
     * @throws CudaException If the creation of the CU- or PTX file
     * fails, or the PTX may not be loaded, or the specified
     * function can not be obtained.
     */
    public static KernelLauncher compile(
            String sourceCode, String functionName, String ... nvccArguments)
    {
        File cuFile = null;
        try
        {
            cuFile = File.createTempFile("temp_JCuda_", ".cu");
        }
        catch (IOException e)
        {
            throw new CudaException("Could not create temporary .cu file", e);
        }
        String cuFileName = cuFile.getPath();
        FileOutputStream fos = null;
        try
        {
            fos = new FileOutputStream(cuFile);
            fos.write(sourceCode.getBytes());
        }
        catch (IOException e)
        {
            throw new CudaException("Could not write temporary .cu file", e);
        }
        finally
        {
            if (fos != null)
            {
                try
                {
                    fos.close();
                }
                catch (IOException e)
                {
                    throw new CudaException(
                            "Could not close temporary .cu file", e);
                }
            }
        }
        return create(cuFileName, functionName, nvccArguments);
    }


    public void setModule(CUmodule module) {
        this.module = module;
    }

    /**
     * Create a new KernelLauncher for the function with the given
     * name, that is contained in the .CU CUDA source file with the
     * given name. <br />
     * <br />
     * <u><b>Note</b></u>: In order to make the function accessible
     * by the name it has in the source code, the function has to
     * be declared as an <code><u>extern "C"</u></code> function: <br />
     *  <br />
     * <code>
     * <b>extern "C"</b><br />
     * __global__ void functionName(...)<br />
     * {<br />
     *     ... <br />
     * }<br />
     * </code>
     * <br />
     * The extension of the given file name is replaced with "ptx".
     * If the PTX file with the resulting name does not exist,
     * or is older than the .CU file, it is compiled from
     * the specified source file using the NVCC (NVIDIA CUDA C
     * Compiler) in a separate process. The optional nvccArguments
     * are passed to the NVCC.<br />
     * <br />
     * The NVCC has to be in a visible directory. E.g. for Windows, the
     * NVCC.EXE has to be in a directory that is contained in the PATH
     * environment variable. Alternatively, the path to the NVCC may
     * be specified by calling {@link KernelLauncher#setCompilerPath(String)}
     * with the respective path. <br />
     *
     * @see KernelLauncher#compile(String, String, String...)
     * @see KernelLauncher#create(String, String, boolean, String...)
     * @see KernelLauncher#load(InputStream, String)
     *
     * @param cuFileName The name of the source file.
     * @param functionName The name of the function.
     * @param nvccArguments Optional arguments for the NVCC
     * @return The KernelLauncher for the specified function
     * @throws CudaException If the creation of the PTX file fails,
     * or the PTX may not be loaded, or the specified function can
     * not be obtained.
     */
    public static KernelLauncher create(
            String cuFileName, String functionName, String ... nvccArguments)
    {
        return create(cuFileName, functionName, false, nvccArguments);
    }

    /**
     * Create a new KernelLauncher for the function with the given
     * name, that is contained in the .CU CUDA source file with the
     * given name. <br />
     * <br />
     * <u><b>Note</b></u>: In order to make the function accessible
     * by the name it has in the source code, the function has to
     * be declared as an <code><u>extern "C"</u></code> function: <br />
     *  <br />
     * <code>
     * <b>extern "C"</b><br />
     * __global__ void functionName(...)<br />
     * {<br />
     *     ... <br />
     * }<br />
     * </code>
     * <br />
     * The extension of the given file name is replaced with "ptx".
     * If the PTX file with the resulting name does not exist,
     * or is older than the .CU file, it is compiled from
     * the specified source file using the NVCC (NVIDIA CUDA C
     * Compiler) in a separate process. The optional nvccArguments
     * are passed to the NVCC.<br />
     * <br />
     * If the <code>forceRebuild</code> flag is 'true', then the
     * PTX file will be recompiled from the given source file,
     * even if it already existed or was newer than the source
     * file, and the already existing PTX file will be
     * overwritten.<br />
     * <br />
     * The NVCC has to be in a visible directory. E.g. for Windows, the
     * NVCC.EXE has to be in a directory that is contained in the PATH
     * environment variable. Alternatively, the path to the NVCC may
     * be specified by calling {@link KernelLauncher#setCompilerPath(String)}
     * with the respective path. <br />
     *
     * @see KernelLauncher#compile(String, String, String...)
     * @see KernelLauncher#create(String, String, String...)
     * @see KernelLauncher#load(InputStream, String)
     *
     * @param cuFileName The name of the source file.
     * @param functionName The name of the function.
     * @param forceRebuild Whether the PTX file should be recompiled
     * and overwritten if it already exists.
     * @param nvccArguments Optional arguments for the NVCC
     * @return The KernelLauncher for the specified function
     * @throws CudaException If the creation of the PTX file fails,
     * or the PTX may not be loaded, or the specified function can
     * not be obtained.
     */
    public static KernelLauncher create(
            String cuFileName, String functionName,
            boolean forceRebuild, String ... nvccArguments)
    {

        // Prepare the PTX file for the CU source file
        String ptxFileName = null;
        try
        {
            ptxFileName =
                    preparePtxFile(cuFileName, forceRebuild, nvccArguments);
        }
        catch (IOException e)
        {
            throw new CudaException(
                    "Could not prepare PTX for source file '" + cuFileName + "'", e);
        }

        KernelLauncher kernelLauncher = new KernelLauncher();
        byte ptxData[] = loadData(ptxFileName);
        kernelLauncher.initModule(ptxData);
        kernelLauncher.initFunction(functionName);
        return kernelLauncher;
    }

    /**
     * Create a new KernelLauncher which may be used to execute the
     * specified function which is loaded from the PTX- or CUBIN
     * (CUDA binary) file with the given name.
     *
     * @see KernelLauncher#compile(String, String, String...)
     * @see KernelLauncher#create(String, String, boolean, String...)
     * @see KernelLauncher#load(InputStream, String)
     *
     * @param functionName The name of the function
     * @return The KernelLauncher for the specified function
     * @throws CudaException If the PTX- or CUBIN may not be loaded,
     * or the specified function can not be obtained.
     */
    public static KernelLauncher load(String functionName,CUmodule module) {
        KernelLauncher kernelLauncher = new KernelLauncher();
        kernelLauncher.setModule(module);
        kernelLauncher.initFunction(functionName);
        return kernelLauncher;
    }
    /**
     * Create a new KernelLauncher which may be used to execute the
     * specified function which is loaded from the PTX- or CUBIN
     * (CUDA binary) file with the given name.
     *
     * @see KernelLauncher#compile(String, String, String...)
     * @see KernelLauncher#create(String, String, boolean, String...)
     * @see KernelLauncher#load(InputStream, String)
     *
     * @param functionName The name of the function
     * @return The KernelLauncher for the specified function
     * @throws CudaException If the PTX- or CUBIN may not be loaded,
     * or the specified function can not be obtained.
     */
    public static KernelLauncher load(String functionName,String type,CUmodule module) {
        KernelLauncher kernelLauncher = new KernelLauncher();
        kernelLauncher.setModule(module);
        kernelLauncher.initFunction(FUNCTION_NAME + "_" + type);
        return kernelLauncher;
    }

    /**
     * Create a new KernelLauncher which may be used to execute the
     * specified function which is loaded from the PTX- or CUBIN
     * (CUDA binary) file with the given name.
     *
     * @see KernelLauncher#compile(String, String, String...)
     * @see KernelLauncher#create(String, String, boolean, String...)
     * @see KernelLauncher#load(InputStream, String)
     *
     * @param moduleFileName The name of the PTX- or CUBIN file
     * @param functionName The name of the function
     * @return The KernelLauncher for the specified function
     * @throws CudaException If the PTX- or CUBIN may not be loaded,
     * or the specified function can not be obtained.
     */
    public static KernelLauncher load(
            String moduleFileName, String functionName,String type) {
        KernelLauncher kernelLauncher = new KernelLauncher();

        try {
            kernelLauncher.initModule(ByteStreams.toByteArray(new FileInputStream(moduleFileName)));
        } catch (IOException e) {
            e.printStackTrace();
        }

        kernelLauncher.initFunction(functionName);
        return kernelLauncher;
    }

    /**
     * Create a new KernelLauncher which may be used to execute the
     * specified function which is loaded from the PTX- or CUBIN
     * data that is read from the given input stream.
     *
     * @see KernelLauncher#compile(String, String, String...)
     * @see KernelLauncher#create(String, String, boolean, String...)
     * @see KernelLauncher#load(InputStream, String)
     *
     * @param moduleInputStream The stream for the PTX- or CUBIN data
     * @param functionName The name of the function
     * @return The KernelLauncher for the specified function
     * @throws CudaException If the PTX- or CUBIN may not be loaded,
     * or the specified function can not be obtained.
     */
    public static KernelLauncher load(
            InputStream moduleInputStream, String functionName)
    {
        KernelLauncher kernelLauncher = new KernelLauncher();
        byte moduleData[] = loadData(moduleInputStream);
        kernelLauncher.initModule(moduleData);
        kernelLauncher.initFunction(functionName);
        return kernelLauncher;
    }


    /**
     * Load the data from the file with the given name and returns
     * it as a 0-terminated byte array
     *
     * @param fileName The name of the file
     * @return The data from the file
     */
    private static byte[] loadData(String fileName)
    {
        InputStream inputStream = null;
        try
        {
            inputStream= new FileInputStream(new File(fileName));
            return loadData(inputStream);
        }
        catch (FileNotFoundException e)
        {
            throw new CudaException(
                    "Could not open '"+fileName+"'", e);
        }
        finally
        {
            if (inputStream != null)
            {
                try
                {
                    inputStream.close();
                }
                catch (IOException e)
                {
                    throw new CudaException(
                            "Could not close '"+fileName+"'", e);
                }
            }
        }
    }

    /**
     * Reads the data from the given inputStream and returns it as
     * a 0-terminated byte array.
     *
     * @param inputStream The inputStream to read
     * @return The data from the inputStream
     */
    private static byte[] loadData(InputStream inputStream)
    {
        ByteArrayOutputStream baos = null;
        try
        {
            baos = new ByteArrayOutputStream();
            byte buffer[] = new byte[8192];
            while (true)
            {
                int read = inputStream.read(buffer);
                if (read == -1)
                {
                    break;
                }
                baos.write(buffer, 0, read);
            }
            baos.write('\0');
            baos.flush();
            return baos.toByteArray();
        }
        catch (IOException e)
        {
            throw new CudaException(
                    "Could not load data", e);
        }
        finally
        {
            if (baos != null)
            {
                try
                {
                    baos.close();
                }
                catch (IOException e)
                {
                    throw new CudaException(
                            "Could not close output", e);
                }
            }
        }

    }



    /**
     * The context which was used to create this instance
     */
    private   CUcontext context;

    /**
     * The module which contains the function
     */
    private CUmodule module;

    /**
     * The function which is executed with this KernelLauncher
     */
    private CUfunction function;

    /**
     * The current block size (number of threads per block)
     * which will be used for the function call.
     */
    private dim3 blockSize = new dim3(1,1,1);

    /**
     * The current grid size (number of blocks per grid)
     * which will be used for the function call.
     */
    private dim3 gridSize = new dim3(1,1,1);

    /**
     * The currently specified size of the shared memory
     * for the function call.
     */
    private int sharedMemSize = 0;

    /**
     * The stream that should be associated with the function call.
     */
    private CUstream stream;


    /**
     * Private constructor. Instantiation only via the static
     * methods.
     */
    private KernelLauncher() {
        initialize();
    }

    /**
     * Initializes this KernelLauncher. This method will try to
     * initialize the JCuda driver API. Then it will try to
     * attach to the current CUDA context. If no active CUDA
     * context exists, then it will try to create one, for
     * the device which is specified by the current
     * deviceNumber.
     *
     * @throws CudaException If it is neither possible to
     * attach to an existing context, nor to create a new
     * context.
     */
    private void initialize() {
        context = ContextHolder.getInstance().getContext(deviceNumber,Thread.currentThread().getName());
    }



    /**
     * Create a new KernelLauncher which uses the same module as
     * this KernelLauncher, but may be used to execute a different
     * function. All parameters (grid size, block size, shared
     * memory size and stream) of the returned KernelLauncher
     * will be independent of 'this' one and initially contain
     * the default values.
     *
     * @param functionName The name of the function
     * @return The KernelLauncher for the specified function
     * @throws CudaException If the specified function can not
     * be obtained from the module of this KernelLauncher.
     */
    public KernelLauncher forFunction(String functionName) {
        KernelLauncher kernelLauncher = new KernelLauncher();
        kernelLauncher.module = this.module;
        kernelLauncher.initFunction(functionName);
        return kernelLauncher;
    }


    /**
     * Initialize the module for this KernelLauncher by loading
     * the PTX- or CUBIN file with the given name.
     *
     * @param path The data from the PTX- or CUBIN file
     */
    private void loadModuleData(String path)
    {
        module = new CUmodule();
        try {
            InputStream fis = new FileInputStream(path);
            JCudaDriver.cuModuleLoadData(module, ByteStreams.toByteArray(fis));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    /**
     * Initialize the module for this KernelLauncher by loading
     * the PTX- or CUBIN file with the given name.
     *
     * @param moduleData The data from the PTX- or CUBIN file
     */
    private void initModule(Pointer moduleData)
    {
        module = new CUmodule();
        cuModuleLoadDataEx(module, moduleData,
                0, new int[0], Pointer.to(new int[0]));
    }



    /**
     * Initialize the module for this KernelLauncher by loading
     * the PTX- or CUBIN file with the given name.
     *
     * @param fileName The data from the PTX- or CUBIN file
     */
    private void initModule(String fileName) {
        module = new CUmodule();
        cuModuleLoad(module, fileName);
    }


    /**
     * Initialize the module for this KernelLauncher by loading
     * the PTX- or CUBIN file with the given name.
     *
     * @param moduleData The data from the PTX- or CUBIN file
     */
    private void initModule(byte moduleData[]) {
        module = new CUmodule();
        cuModuleLoadData(module,moduleData);
    }

    /**
     * Initialize this KernelLauncher for calling the function with
     * the given name, which is contained in the module of this
     * KernelLauncher
     *
     * @param functionName The name of the function
     */
    private void initFunction(String functionName)
    {
        // Obtain the function from the module
        function = new CUfunction();
        String functionErrorString =
                "Could not get function '" + functionName + "' from module " + functionName  +  " "+"\n"+
                        "Name in module might be mangled. Try adding the line "+"\n"+
                        "extern \"C\""+"\n"+
                        "before the function you want to call, or open the " +
                        "PTX/CUBIN "+"\n"+"file with a text editor to find out " +
                        "the mangled function name";
        try
        {
            int result = cuModuleGetFunction(function, module, functionName);
            if (result != CUresult.CUDA_SUCCESS)
                throw new CudaException(functionErrorString);


        }
        catch (CudaException e)
        {
            throw new CudaException(functionErrorString, e);
        }
    }

    /**
     * Returns the module that was created from the PTX- or CUBIN file, and
     * which contains the function that should be executed. This
     * module may also be used to access symbols and texture
     * references. However, clients should not modify or unload
     * the module.
     *
     * @return The CUmodule
     */
    public CUmodule getModule()
    {
        return module;
    }

    /**
     * Set the grid size (number of blocks per grid) for the function
     * call.<br />
     * <br />
     * This corresponds to the first parameter in the runtime call:<br />
     * <br />
     * <code>
     * kernel&lt;&lt;&lt;<b><u>gridSize</u></b>, blockSize,
     * sharedMemSize, stream&gt;&gt;&gt;(...);
     * </code>
     * <br />
     * <br />
     * The default grid size is (1,1,1)
     *
     * @see KernelLauncher#call(Object...)
     * @see KernelLauncher#setup(dim3, dim3)
     * @see KernelLauncher#setup(dim3, dim3, int)
     * @see KernelLauncher#setup(dim3, dim3, int, CUstream)
     *
     * @param x The number of blocks per grid in x-direction
     * @param y The number of blocks per grid in y-direction
     * @return This instance
     */
    public KernelLauncher setGridSize(int x, int y)
    {
        gridSize.x = x;
        gridSize.y = y;
        return this;
    }

    /**
     * Set the grid size (number of blocks per grid) for the function
     * call.<br />
     * <br />
     * This corresponds to the first parameter in the runtime call:<br />
     * <br />
     * <code>
     * kernel&lt;&lt;&lt;<b><u>gridSize</u></b>, blockSize,
     * sharedMemSize, stream&gt;&gt;&gt;(...);
     * </code>
     * <br />
     * <br />
     * The default grid size is (1,1,1)
     *
     * @see KernelLauncher#call(Object...)
     * @see KernelLauncher#setup(dim3, dim3)
     * @see KernelLauncher#setup(dim3, dim3, int)
     * @see KernelLauncher#setup(dim3, dim3, int, CUstream)
     *
     * @param x The number of blocks per grid in x-direction
     * @param y The number of blocks per grid in y-direction
     * @param z The number of blocks per grid in z-direction
     * @return This instance
     */
    public KernelLauncher setGridSize(int x, int y, int z)
    {
        gridSize.x = x;
        gridSize.y = y;
        gridSize.z = z;
        return this;
    }

    /**
     * Set the block size (number of threads per block) for the function
     * call.<br />
     * <br />
     * This corresponds to the second parameter in the runtime call:<br />
     * <br />
     * <code>
     * kernel&lt;&lt;&lt;gridSize, <b><u>blockSize</u></b>,
     * sharedMemSize, stream&gt;&gt;&gt;(...);
     * </code>
     * <br />
     * <br />
     * The default block size is (1,1,1)
     *
     * @see KernelLauncher#call(Object...)
     * @see KernelLauncher#setup(dim3, dim3)
     * @see KernelLauncher#setup(dim3, dim3, int)
     * @see KernelLauncher#setup(dim3, dim3, int, CUstream)
     *
     * @param x The number of threads per block in x-direction
     * @param y The number of threads per block in y-direction
     * @param z The number of threads per block in z-direction
     * @return This instance
     */
    public KernelLauncher setBlockSize(int x, int y, int z)
    {
        blockSize.x = x;
        blockSize.y = y;
        blockSize.z = z;
        return this;
    }

    /**
     * Set the size of the shared memory for the function
     * call.<br />
     * <br />
     * This corresponds to the third parameter in the runtime call:<br />
     * <br />
     * <code>
     * kernel&lt;&lt;&lt;gridSize, blockSize,
     * <b><u>sharedMemSize</u></b>, stream&gt;&gt;&gt;(...);
     * </code>
     * <br />
     * <br />
     * The default shared memory size is 0.
     *
     * @see KernelLauncher#call(Object...)
     * @see KernelLauncher#setup(dim3, dim3)
     * @see KernelLauncher#setup(dim3, dim3, int)
     * @see KernelLauncher#setup(dim3, dim3, int, CUstream)
     *
     * @param sharedMemSize The size of the shared memory, in bytes
     * @return This instance
     */
    public KernelLauncher setSharedMemSize(int sharedMemSize)
    {
        this.sharedMemSize = sharedMemSize;
        return this;
    }

    /**
     * Set the stream for the function call.<br />
     * <br />
     * This corresponds to the fourth parameter in the runtime call:<br />
     * <br />
     * <code>
     * kernel&lt;&lt;&lt;gridSize, blockSize,
     * sharedMemSize, <b><u>stream</u></b>&gt;&gt;&gt;(...);
     * </code>
     * <br />
     * <br />
     * The default stream is null (0).
     *
     * @see KernelLauncher#call(Object...)
     * @see KernelLauncher#setup(dim3, dim3)
     * @see KernelLauncher#setup(dim3, dim3, int)
     * @see KernelLauncher#setup(dim3, dim3, int, CUstream)
     *
     * @param stream The stream for the function call
     * @return This instance
     */
    public KernelLauncher setStream(CUstream stream) {
        this.stream = stream;
        return this;
    }



    /**
     * Set the given grid size and block size for this KernelLauncher.
     *
     * @see KernelLauncher#call(Object...)
     * @see KernelLauncher#setup(dim3, dim3, int)
     * @see KernelLauncher#setup(dim3, dim3, int, CUstream)
     *
     * @param gridSize The grid size (number of blocks per grid)
     * @param blockSize The block size (number of threads per block)
     * @return This instance
     */
    public KernelLauncher setup(dim3 gridSize, dim3 blockSize)
    {
        return setup(gridSize, blockSize, sharedMemSize, stream);
    }

    /**
     * Set the given grid size and block size and shared memory size
     * for this KernelLauncher.
     *
     * @see KernelLauncher#call(Object...)
     * @see KernelLauncher#setup(dim3, dim3)
     * @see KernelLauncher#setup(dim3, dim3, int, CUstream)
     *
     * @param gridSize The grid size (number of blocks per grid)
     * @param blockSize The block size (number of threads per block)
     * @param sharedMemSize The size of the shared memory
     * @return This instance
     */
    public KernelLauncher setup(dim3 gridSize, dim3 blockSize,
                                int sharedMemSize)
    {
        return setup(gridSize, blockSize, sharedMemSize, stream);
    }


    public  CUcontext context() {
        return context;
    }

    /**
     * Set the given grid size and block size, shared memory size
     * and stream for this KernelLauncher.
     *
     * @see KernelLauncher#call(Object...)
     * @see KernelLauncher#setup(dim3, dim3)
     * @see KernelLauncher#setup(dim3, dim3, int)
     *
     * @param gridSize The grid size (number of blocks per grid)
     * @param blockSize The block size (number of threads per block)
     * @param sharedMemSize The size of the shared memory
     * @param stream The stream for the kernel invocation
     * @return This instance
     */
    public KernelLauncher setup(dim3 gridSize, dim3 blockSize,
                                int sharedMemSize, CUstream stream) {
        setGridSize(gridSize.x, gridSize.y);
        setBlockSize(blockSize.x, blockSize.y, blockSize.z);
        setSharedMemSize(sharedMemSize);
        setStream(stream);
        return this;
    }

    /**
     * Call the function of this KernelLauncher with the current
     * grid size, block size, shared memory size and stream, and
     * with the given arguments.<br />
     * <br />
     * The given arguments must all be either of the type
     * <code>Pointer</code>, or of a primitive type except boolean.
     * Otherwise, a CudaException will be thrown.
     *
     * @param args The arguments for the function call
     * @throws CudaException if an argument with an invalid type
     * was given, or one of the internal functions for setting
     * up and executing the kernel failed.
     */
    public  void call(Object ... args) {

        Pointer kernelParameters[] = new Pointer[args.length];

        for (int i = 0; i < args.length; i++) {
            Object arg = args[i];

            if (arg instanceof Pointer)
            {
                Pointer argPointer = (Pointer)arg;
                Pointer pointer = Pointer.to(argPointer);
                kernelParameters[i] = pointer;
                //logger.info("argument " + i + " type is Pointer");
            }
            else if (arg instanceof Byte)
            {
                Byte value = (Byte)arg;
                Pointer pointer = Pointer.to(new byte[]{value});
                kernelParameters[i] = pointer;
                //logger.info("argument " + i + " type is Byte");
            }
            else if (arg instanceof Short)
            {
                Short value = (Short)arg;
                Pointer pointer = Pointer.to(new short[]{value});
                kernelParameters[i] = pointer;
                // logger.info("argument " + i + " type is Short");
            }
            else if (arg instanceof Integer)
            {
                Integer value = (Integer)arg;
                Pointer pointer = Pointer.to(new int[]{value});
                kernelParameters[i] = pointer;
                //logger.info("argument " + i + " type is Integer");
            }
            else if (arg instanceof Long)
            {
                Long value = (Long)arg;
                Pointer pointer = Pointer.to(new long[]{value});
                kernelParameters[i] = pointer;
                //logger.info("argument " + i + " type is Long");
            }
            else if (arg instanceof Float)
            {
                Float value = (Float)arg;
                Pointer pointer = Pointer.to(new float[]{value});
                kernelParameters[i] = pointer;
                //logger.info("argument " + i + " type is Float");
            }
            else if (arg instanceof Double)
            {
                Double value = (Double)arg;
                Pointer pointer = Pointer.to(new double[]{value});
                kernelParameters[i] = pointer;
                //logger.info("argument " + i + " type is Double");
            }

            /**
             * Of note here. double[] of length 1 is
             * passed to the cuda kernel as a direct double.
             * Eg: double
             * Rather than double *
             * If it's actually, a buffer, we need to ensure
             * data is allocated on the gpu (hence why we throw the exception)
             *
             * This is applicable for any numerical primitive array (the below few)
             */
            else if (arg instanceof double[]) {
                double[] d = (double[]) arg;
                if(d.length == 1)
                    kernelParameters[i] = Pointer.to(d);
                else
                    throw new IllegalArgumentException("Please wrap double arrays in a buffer with kernelfunctions.alloc()");
            }



            else if (arg instanceof float[])
            {
                float[] f = (float[]) arg;
                if(f.length == 1)
                    kernelParameters[i] = Pointer.to(f);

                else
                    throw new IllegalArgumentException("Please wrap float arrays in a buffer with kernelfunctions.alloc()");

            }

            else if (arg instanceof int[])
            {
                int[] i2 = (int[]) arg;
                if(i2.length == 1)
                    kernelParameters[i] = Pointer.to(i2);
                else
                    throw new IllegalArgumentException("Please wrap int arrays in a buffer with kernelfunctions.alloc()");

            }
            else if(arg instanceof jcuda.jcurand.curandGenerator) {
                jcuda.jcurand.curandGenerator rng = (jcuda.jcurand.curandGenerator) arg;
                kernelParameters[i] = Pointer.to(rng);

            }
            else
            {
                throw new CudaException(
                        "Type " + arg.getClass() + " may not be passed to a function");
            }
        }

        cuLaunchKernel(function,
                gridSize.x, gridSize.y, gridSize.z,
                blockSize.x, blockSize.y, blockSize.z,
                sharedMemSize, stream,
                Pointer.to(kernelParameters), null
        );


    }

    public CUfunction getFunction() {
        return function;
    }

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist or is older
     * than the source file, it is compiled from the given file
     * using NVCC. If the forceRebuild flag is 'true', then the PTX
     * file is rebuilt even if it already exists or is newer than the
     * source file. The name of the PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @param forceRebuild Whether the PTX file should be re-created
     * even if it exists already.
     * @param nvccArguments Optional arguments for the NVCC
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     * @throws CudaException If the creation of the PTX file fails
     */
    private static String preparePtxFile(
            String cuFileName, boolean forceRebuild, String ... nvccArguments)
            throws IOException {
        logger.info("Preparing PTX for \n"+cuFileName);

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())

            throw new CudaException("Input file not found: " + cuFileName);


        // Replace the file extension with "ptx"
        String ptxFileName = null;
        int lastIndex = cuFileName.lastIndexOf('.');
        if (lastIndex == -1)
            ptxFileName = cuFileName + ".ptx";

        else
            ptxFileName = cuFileName.substring(0, lastIndex)+".ptx";


        // Return if the file already exists and should not be rebuilt
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists() && !forceRebuild) {
            long cuLastModified = cuFile.lastModified();
            long ptxLastModified = ptxFile.lastModified();
            if (cuLastModified < ptxLastModified)
                return ptxFileName;

        }

        // Build the command line
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String defaultArguments = "";
        String optionalArguments = createArgumentsString(nvccArguments);
        String command =
                compilerPath + "nvcc " + modelString + " " + defaultArguments +
                        " " + optionalArguments + " -ptx "+
                        cuFile.getPath()+" -o "+ptxFileName;


        // Execute the command line and wait for the output
        logger.info("Executing\n" + command);
        Process process = Runtime.getRuntime().exec(command);
        String errorMessage =
                new String(toByteArray(process.getErrorStream()));
        String outputMessage =
                new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new CudaException(
                    "Interrupted while waiting for nvcc output", e);
        }

        logger.info("nvcc process exitValue "+exitValue);
        if (exitValue != 0) {
            logger.error("errorMessage:\n"+errorMessage);
            logger.error("outputMessage:\n"+outputMessage);
            throw new CudaException(
                    "Could not create .ptx file: "+errorMessage);
        }

        return ptxFileName;
    }

    /**
     * Creates a single string from the given argument strings
     *
     * @param nvccArguments The argument strings
     * @return A single string containing the arguments
     */
    private static String createArgumentsString(String ... nvccArguments)
    {
        if (nvccArguments == null || nvccArguments.length == 0)
        {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (String s : nvccArguments)
        {
            sb.append(s);
            sb.append(" ");
        }
        return sb.toString();
    }


    /**
     * Fully reads the given InputStream and returns it as a byte array.
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(
            InputStream inputStream) throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    @Override
    public String toString() {
        return "KernelLauncher{" +
                "deviceNumber=" + deviceNumber +
                ", context=" + context +
                ", module=" + module +
                ", function=" + function +
                ", blockSize=" + blockSize +
                ", gridSize=" + gridSize +
                ", sharedMemSize=" + sharedMemSize +
                ", stream=" + stream +
                '}';
    }
}



