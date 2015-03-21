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

import jcuda.driver.*;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import static jcuda.driver.JCudaDriver.*;

/**
 * Kernel function loader:
 *
 * @author Adam Gibson
 */
public class KernelFunctionLoader {

    public final static String NAME_SPACE = "org.nd4j.linalg.jcublas";
    public final static String DOUBLE = NAME_SPACE + ".double.functions";
    public final static String FLOAT = NAME_SPACE + ".float.functions";
    public final static String IMPORTS_FLOAT = NAME_SPACE + ".float.imports";
    public final static String IMPORTS_DOUBLE = NAME_SPACE + ".double.imports";
    private static Logger log = LoggerFactory.getLogger(KernelFunctionLoader.class);
    private static Map<Integer, CUcontext> devices = new ConcurrentHashMap<>();
    private static KernelFunctionLoader INSTANCE;
    private Map<String, CUmodule> modules = new HashMap<>();
    private Map<String, CUfunction> functions = new HashMap<>();
    private boolean init = false;

    private KernelFunctionLoader() {
    }

    /**
     * Singleton pattern
     *
     * @return
     */
    public static KernelFunctionLoader getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new KernelFunctionLoader();
            Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                @Override
                public void run() {
                    INSTANCE.unload();
                }
            }));
        }
        return INSTANCE;
    }

    private static String dataFolder(int type) {
        return "/kernels/" + (type == DataBuffer.FLOAT ? "float" : "double");
    }

    /**
     * Get the given cuda module
     *
     * @param function the function name
     * @param dataType the data type for
     * @return the cuda module if it exists or null
     */
    public CUfunction getFunction(String function, String dataType) {
        return functions.get(function + "_" + dataType);
    }

    /**
     * Get the given cuda module
     *
     * @param function the function name
     * @param dataType the data type for
     * @return the cuda module if it exists or null
     */
    public CUmodule getModule(String function, String dataType) {
        return modules.get(function + "_" + dataType);
    }

    /**
     * Clean up all the modules
     */
    public void unload() {
        for (CUmodule module : modules.values())
            unload(module);
        init = false;
    }

    /**
     * Unload a module
     *
     * @param cUmodule the module to unload
     */
    public void unload(CUmodule cUmodule) {
        try {
            JCudaDriver.cuModuleUnload(cUmodule);
        } catch (Exception e) {

        }
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
        StringBuffer sb = new StringBuffer();
        sb.append("nvcc -g -G -ptx");

        ClassPathResource res = new ClassPathResource("/cudafunctions.properties");
        if (!res.exists())
            throw new IllegalStateException("Please put a cudafunctions.properties in your class path");
        Properties props = new Properties();
        props.load(res.getInputStream());
        log.info("Registering cuda functions...");
        initDevices();
        //ensure imports for each file before compiling
        ensureImports(props, "float");
        ensureImports(props, "double");

        compileAndLoad(props, FLOAT, "float");
        compileAndLoad(props, DOUBLE, "double");

        init = true;
    }

    public void initDevices() {
        if (devices.containsKey(0))
            return;
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        devices.put(0, context);
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
     * @throws java.io.IOException If an I/O error occurs
     */
    private void compileAndLoad(Properties props, String key, String dataType) throws IOException {
        String f = props.getProperty(key);
        StringBuffer sb = new StringBuffer();
        sb.append("nvcc -g -G ");
        sb.append(" --include-path ");
        String tmpDir = System.getProperty("java.io.tmpdir");
        StringBuffer dir = new StringBuffer();
        sb.append(tmpDir)
                .append(File.separator)
                .append("kernels")
                .append(File.separator).append(dataType).append(File.separator)
                .toString();
        String kernelPath = dir.append(tmpDir)
                .append(File.separator)
                .append("kernels")
                .append(File.separator).append(dataType).append(File.separator)
                .toString();

        sb.append(" ").append(" -ptx ");
        log.info("Loading " + dataType + " cuda functions");
        if (f != null) {
            String[] split = f.split(",");
            for (String s : split) {
                String loaded = extract("/kernels/" + dataType + "/" + s + ".cu", dataType.equals("float") ? DataBuffer.FLOAT : DataBuffer.DOUBLE);
                sb.append(" " + loaded);
            }

            sb.append(" --output-directory " + kernelPath);


            Process process = Runtime.getRuntime().exec(sb.toString());

            String errorMessage =
                    new String(IOUtils.toByteArray(process.getErrorStream()));
            String outputMessage =
                    new String(IOUtils.toByteArray(process.getInputStream()));
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

            for (String module : split) {
                CUmodule m = new CUmodule();
                log.info("Loading " + module);
                String path = kernelPath + module + ".ptx";
                cuModuleLoad(m, path);
                modules.put(key + "_" + dataType, m);
                // Obtain a function pointer to the "add" function.
                CUfunction function = new CUfunction();
                String name = module + "_" + dataType;
                try {
                    cuModuleGetFunction(function, m, name);
                } catch (Exception e) {
                    throw new RuntimeException("Function " + name + " not found!");
                }

                functions.put(name, function);
                //cleanup
                File ptxFile = new File(path);
                if (!ptxFile.exists())
                    throw new IllegalStateException("No ptx file " + ptxFile.getAbsolutePath() + " found!");
                ptxFile.delete();
            }

        }
    }

    //extract the source file

    /**
     * Extract the resource ot the specified
     * absolute path
     *
     * @param file     the file
     * @param dataType the data type to extract for
     * @return
     * @throws IOException
     */
    public String extract(String file, int dataType) throws IOException {

        String path = dataFolder(dataType);
        String tmpDir = System.getProperty("java.io.tmpdir");
        File dataDir = new File(tmpDir, path);
        if (!dataDir.exists())
            dataDir.mkdirs();


        return loadFile(file);

    }

    private void ensureImports(Properties props, String dataType) throws IOException {
        if (dataType.equals("float")) {
            String[] imports = props.getProperty(IMPORTS_FLOAT).split(",");
            for (String import1 : imports) {
                loadFile("/kernels/" + dataType + "/" + import1);
            }
        } else {
            String[] imports = props.getProperty(IMPORTS_DOUBLE).split(",");
            for (String import1 : imports) {
                loadFile("/kernels/" + dataType + "/" + import1);
            }
        }

    }


    private String loadFile(String file) throws IOException {
        ClassPathResource resource = new ClassPathResource(file);
        String tmpDir = System.getProperty("java.io.tmpdir");

        if (!resource.exists())
            throw new IllegalStateException("Unable to find file " + resource);
        File out = new File(tmpDir, file);
        if (!out.getParentFile().exists())
            out.getParentFile().mkdirs();
        if (out.exists())
            out.delete();
        out.createNewFile();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(out));
        IOUtils.copy(resource.getInputStream(), bos);
        bos.flush();
        bos.close();

        out.deleteOnExit();
        return out.getAbsolutePath();

    }


}
