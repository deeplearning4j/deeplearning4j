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


import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;
import jcuda.utils.KernelLauncher;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.util.CudaArgs;
import org.nd4j.linalg.util.JarResource;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.*;
import java.util.regex.Pattern;


/**
 * Kernel function loader:
 *
 * @author Adam Gibson
 */
public class KernelFunctionLoader {
    public final static String NAME_SPACE = "org.nd4j.linalg.jcuda.jcublas";
    public final static String DOUBLE = NAME_SPACE + ".double.functions";
    public final static String FLOAT = NAME_SPACE + ".float.functions";
    public final static String CACHE_COMPILED = NAME_SPACE + ".cache_compiled";
    public final static String FUNCTION_KEY = "org.nd4j.linalg.jcuda.jcublas.functions";

    private static KernelFunctionLoader INSTANCE;
    private boolean alreadyCompiled = false;

    private boolean init = false;
    private static Logger log = LoggerFactory.getLogger(KernelFunctionLoader.class);
    private String kernelPath;
    private String[] modules;
    public final static String PRINT_KERNEL_NAME = "printShapeBuffer";
    private static KernelLauncher printFunction;

    private Table<String, DataBuffer.Type, String> paths = HashBasedTable.create();

    // Thread, <FunctionName, DataType>, KernelLauncher
    private static Table<String, Pair<String, DataBuffer.Type>,KernelLauncher> launchers = HashBasedTable.create();

    private KernelFunctionLoader() {}

    /**
     * Singleton pattern
     *
     * @return
     */
    public static synchronized KernelFunctionLoader getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new KernelFunctionLoader();
            Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                @Override
                public void run() {
                    INSTANCE.unload();
                }
            }));
            try {
                INSTANCE.load();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }


        return INSTANCE;
    }


    /**
     * Get the launcher for a function
     * @param functionName the function to get the launcher for
     * @param dataType the data type to launch with
     * @return the launcher for the given
     * function and data type
     */
    public  static KernelLauncher launcher(String functionName,DataBuffer.Type dataType) {
        KernelLauncher launcher =  KernelFunctionLoader.getInstance().get(functionName,dataType);
        return launcher;
    }


    /**
     * Returns whether the target Op has a kernel or not
     *
     * @param op Op to be checked for existance
     * @return true if the function has a kernel
     * false othr wise
     */
    public boolean exists(Op op) {
        /**
         * We should check for specific kernel
         */
        if (CudaArgs.getModuleNameFor(op) == null) return false;

        /**
         * And specific OpCode
         */
        if (CudaArgs.getOpCode(op) < 0) return false;

        return true;
    }


    /**
     * Gets a kernel launcher
     * for a given function name and data type
     * @param functionName the name of the function
     * @param dataType the data type to get
     * @return the kernel launcher for the
     * given function
     */
    public KernelLauncher get(String functionName,DataBuffer.Type dataType) {
        String name = functionName;// + "_" + dataType;
        if(!launchers.containsRow(Thread.currentThread().getName())) {

            try {
                log.info("Loading modules for " + Thread.currentThread().getName());
                loadModules();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        }

        KernelLauncher launcher = launchers.get(Thread.currentThread().getName(), Pair.of(name, dataType));
        if(launcher == null) {
            throw new RuntimeException("Can't get module for name: " + name);
            /*
            name = functionName + "_strided" + "_" + dataType;
            launcher = launchers.get(Thread.currentThread().getName(),Pair.of(name, dataType));
            if(launcher == null)
                return null;
            */
        }
        return launcher;
    }


    /**
     * Clean up all the modules
     */
    public void unload() {
        init = false;
    }



    /**
     * Load the appropriate functions from the class
     * path in to one module
     *
     * @return the module associated with this
     * @throws Exception
     */
    public void load() throws Exception {
        if (init)
            return;

        ClassPathResource res = new ClassPathResource("/cudafunctions.properties", KernelFunctionLoader.class.getClassLoader());
        if (!res.exists())
            throw new IllegalStateException("Please put a cudafunctions.properties in your class path");
        Properties props = new Properties();
        props.load(res.getInputStream());
        log.info("Registering cuda functions...");
        //ensure imports for each file before compiling
        compileAndLoad(props);

        init = true;
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
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private void compileAndLoad(Properties props) throws IOException {
        compileAndLoad(props,0);
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
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private void compileAndLoad(Properties props,int compiledAttempts) throws IOException {

        /*
            Since CUDA codebase was greatly changed, new kernel loader and mapper is required.
            So, here we go.

            We assume that:
                1. We have .ptx/.cubin file.
                2. We have predefined kernel names
                3. We have predefined and hardcoded codes for Ops

                So, we should load our distribution fille (since we have one now), and register all functions upon backend initialization.
                Ops->code translation will be handled via hardcoded structure on the call
         */

        // we have predefined list of kernels available, each of them has Float and Double suffix
        String f = props.getProperty(FUNCTION_KEY);
        log.info("Kernels to be loaded: " + f);


        /*
            We're loading PTX kernels from internal resource
         */

        File kernelsPtx = new JarResource("/all.ptx").getFile();


        /*

        // let's check if cubin/ptx was already extracted from jar.
        boolean shouldExtract = !(kernelsPtx.exists());
        if (shouldExtract) {
            // if not extracted - we'll do that now
            log.info("Unpacking kernels...");
            ClassPathResource ptxResource = new ClassPathResource("/all.ptx");
            ClassPathResource cubinResource = new ClassPathResource("/all.cubin");

            if (ptxResource.exists()) {
                log.info("Going for PTX distribution...");
                FileUtils.copyFile(ptxResource.getFile(), kernelsPtx);
                usingPtx = true;
            } else {
                throw new IllegalStateException("No CUDA kernels were found!");
            }
        }

        */

        String[] split = f.split(",");
        this.modules = split;


        /*
            We're not redistributing .cu files anymore, only .ptx (or .cubin), so we store all kernel names into paths map, that'll be reused on kernel invocations
         */
        String path = kernelsPtx.getAbsolutePath();

        for (String module : split) {
            // we have single .cubin file for all kernels
            // the only difference is concatenated to kernel data type of the function.
            // i.e. reduce3 = reduce3Float & reduce3Double

            String name = module;

            // so we're pushing both data typins pointing to the same reduce3. Concatenation will be applied on later stages
            paths.put(name,DataBuffer.Type.DOUBLE,path);
            paths.put(name,DataBuffer.Type.FLOAT, path);
        }

        /*
            now we map each kernel into cuda driver
        */
        try {
            loadModules();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        if (1>0) return;
        /*
        TODO: legacy code, to be removed later

        String f = props.getProperty(FUNCTION_KEY);
        String tmpDir = System.getProperty("java.io.tmpdir");
        StringBuffer dir = new StringBuffer();
        this.kernelPath = dir.append(tmpDir)
                .append(File.separator)
                .append("nd4j-kernels")
                .append(File.separator)
                .append("output")
                .append(File.separator)
                .toString();
        File tmpDir2 = new File(tmpDir + File.separator + "nd4j-kernels" + File.separatorChar + "output");

        log.info("Kernels to be loaded: " + f);

        boolean shouldCompile = !tmpDir2.exists() || tmpDir2.exists() && tmpDir2.listFiles().length <= 1 || alreadyCompiled;

        String[] split = f.split(",");
        this.modules = split;
        if(shouldCompile) {
            loadCudaKernels();
        }

        else {
            log.info("Modules appear to already be compiled..attempting to use cache");
            for (String module : split) {
                String path = kernelPath + module   + ".cubin";
                String nameDouble = module + "_double";
                String nameFloat = module + "_float";
                paths.put(nameDouble,path);
                paths.put(nameFloat,path);

            }
        }

        try {
            loadModules(split,kernelPath);
            alreadyCompiled = true;
        }

        catch (IOException e1) {
            if(!shouldCompile && compiledAttempts < 3) {
                log.warn("Error loading modules...attempting recompile");
                FileUtils.deleteDirectory(new File(kernelPath));
                props.setProperty(CACHE_COMPILED,String.valueOf(true));
                compileAndLoad(props,compiledAttempts + 1);
            }
            else
                throw new RuntimeException(e1);
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
        */
    }




    /**
     * Print the given buffer
     * @param buffer
     * @param ctx
     * @throws Exception
     */
    @Deprecated
    public static void printBuffer(JCudaBuffer buffer,CudaContext ctx) throws Exception {
        CublasPointer pointer = new CublasPointer(buffer,ctx);
        pointer.copyToHost();
        JCublas.printVector(buffer.length(),pointer.getDevicePointer());
        buffer.asNio().rewind();
        //JCublas.printVector(buffer.length(),pointer.getHostPointer());
        ByteBuffer pointer2 = pointer.getHostPointer().getByteBuffer(0, buffer.getElementSize() * buffer.length());
        FloatBuffer intBuffer = pointer2.asFloatBuffer();
        for(int i = 0; i < buffer.length(); i++) {
            System.out.println("Item " + i + " is " + intBuffer.get(i));
        }
        JCuda.cudaDeviceSynchronize();
    }

    /**
     * This method caches all kernels for current thread
     * @throws Exception
     */
    private void loadModules() throws Exception {
        for(String function: paths.rowKeySet()) {

            for (DataBuffer.Type dataType: DataBuffer.Type.values()) {

                // we don't have dataType INT kernels atm, so we'll skip it
                if (dataType.equals(DataBuffer.Type.INT)) continue;

                // we assume symmetric values for functions/datatypes. i.e.:path CAN'T be null
                String path = paths.get(function, dataType);
                String functionName = function + StringUtils.capitalize(dataType.toString().toLowerCase());
                log.info("Loading {}", functionName);


                KernelLauncher launch = KernelLauncher.load(path, functionName, dataType.toString());
                launchers.put(Thread.currentThread().getName(), Pair.of(function, dataType), launch);
            }
        }
    }

    @Deprecated
    private void loadModules(String[] split,String kernelPath) throws Exception {
        /*
        ContextHolder.getInstance().setContext();
        for (String module : split) {
            log.debug("Loading " + module);
            String path = kernelPath  +  module + ".cubin";
            if(!new File(path).exists())
                throw new IllegalStateException("Unable to find path " + path + ". Recompiling");
            String name = module;
            paths.put(name,path);
            KernelLauncher launch = KernelLauncher.load(path, name,"float");
            //reuse the module from the gpu but load the function instead
            KernelLauncher doubleLauncher = KernelLauncher.load(name,"double",launch.getModule());
            launchers.put(Thread.currentThread().getName(),name + "_double", doubleLauncher);
            launchers.put(Thread.currentThread().getName(),name + "_float",launch);
            if(printFunction == null) {
                printFunction =  KernelLauncher.load(PRINT_KERNEL_NAME,launch.getModule());
            }
        }
        */
    }


    private void loadCudaKernels() throws IOException {
        Set<String> resources = new Reflections("org.nd4j.nd4j-kernels", new ResourcesScanner()).getResources(Pattern.compile(".*"));
        for(String resource : resources) {
            extract(resource);
        }

        File outputDir = new File(System.getProperty("java.io.tmpdir") + File.separator + "nd4j-kernels","output");
        outputDir.mkdirs();
        log.info("Compiling cuda kernels into: " + outputDir.getAbsolutePath());
        String[] commands = {"bash","-c","make && /usr/bin/make install"};
        ProcessBuilder probuilder = new ProcessBuilder(commands);
        //You can set up your work directory
        probuilder.directory(new File(System.getProperty("java.io.tmpdir") + File.separator + "nd4j-kernels"));

        Process process = probuilder.start();
        //Read out dir output
        InputStream is = process.getInputStream();
        try {
            process.waitFor();
            BufferedInputStream bis = new BufferedInputStream(is);
            List<String> list = IOUtils.readLines(bis, "UTF-8");
            for(String item : list) {
                log.info(item);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

    }

    /**
     * Extract the resource ot the specified
     * absolute path
     *
     * @param file     the file

     * @throws IOException
     */
    public String extract(String file) throws IOException {
        String tmpDir = System.getProperty("java.io.tmpdir");
        String[] split = file.split("/");
        //minus the first 2 and the last entry for the parent directory path
        String[] newArray = new String[split.length - 2];
        for(int i = 0,j = 2; j < split.length; i++,j++) {
            newArray[i] = split[j];
        }

        String split2 = StringUtils.join(newArray,"/");
        File dataDir = new File(tmpDir,split2);
        if (!dataDir.getParentFile().exists())
            dataDir.mkdirs();


        return loadFile(file,dataDir);

    }



    private String loadFile(String file,File dataDir) throws IOException {
        ClassPathResource resource = new ClassPathResource(file, KernelFunctionLoader.class.getClassLoader());

        if (!resource.exists())
            throw new IllegalStateException("Unable to find file " + resource);
        File out = dataDir;
        if (!out.getParentFile().exists())
            out.getParentFile().mkdirs();
        if (out.exists())
            out.delete();
        out.createNewFile();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(out));
        IOUtils.copy(resource.getInputStream(), bos);
        bos.flush();
        bos.close();

        return out.getAbsolutePath();

    }


    /**
     * This method takes Op, and returns CUDA kernel name that contains implementation
     *
     * @param op Op to be discovered
     * @return
     */
    public static String getKernelName(Op op) {
        if (op instanceof Accumulation) {
            System.out.println("Accumulation");
        }
        return null;
    }

    /**
     * This method takes Op, and returns CUDA kernel op factory Id for specified op
     *
     * @param op
     * @return
     */
    public static int getOpCode(Op op) {
        return 0;
    }
}
