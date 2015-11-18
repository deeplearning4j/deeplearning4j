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
import jcuda.utils.KernelLauncher;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.*;
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
    private Map<String,String> paths = new HashMap<>();
    private static KernelFunctionLoader INSTANCE;
    private static Table<String,String,KernelLauncher> launchers = HashBasedTable.create();
    private boolean init = false;
    private static Logger log = LoggerFactory.getLogger(KernelFunctionLoader.class);
    private String kernelPath;
    private String[] modules;
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
    public  static KernelLauncher launcher(String functionName,String dataType) {
        KernelLauncher launcher =  KernelFunctionLoader.getInstance().get(functionName,dataType);
        return launcher;
    }


    /**
     * Returns whether the function has a kernel or not
     * @param functionName the name of the function
     * @return true if the function has a kernel
     * false othr wise
     */
    public boolean exists(String functionName) {
        return get(functionName,"double") != null || get(functionName,"float") != null;
    }


    /**
     * Gets a kernel launcher
     * for a given function name and data type
     * @param functionName the name of the function
     * @param dataType the data type to get
     * @return the kernel launcher for the
     * given function
     */
    public KernelLauncher get(String functionName,String dataType) {
        String name = functionName + "_" + dataType;
        if(!launchers.containsRow(Thread.currentThread().getName())) {
            try {
                loadModules(modules,kernelPath);
                log.debug("Loading modules for " + Thread.currentThread().getName());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        KernelLauncher launcher = launchers.get(Thread.currentThread().getName(),name);
        if(launcher == null) {
            name = functionName + "_strided" + "_" + dataType;
            launcher = launchers.get(Thread.currentThread().getName(),name);
            if(launcher == null)
                return null;
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
        String f = props.getProperty(FUNCTION_KEY);
        String tmpDir = System.getProperty("java.io.tmpdir");
        StringBuffer dir = new StringBuffer();
        this.kernelPath = dir.append(tmpDir)
                .append(File.separator)
                .append("nd4j-kernels")
                .append(File.separator)
                .toString();
        File tmpDir2 = new File(tmpDir + File.separator + "nd4j-kernels");

        boolean shouldCompile = !tmpDir2.exists() || tmpDir2.exists() && tmpDir2.listFiles().length <= 1;
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
        }

        catch (Exception e) {
            if(!shouldCompile && compiledAttempts < 3) {
                log.warn("Error loading modules...attempting recompile");
                FileUtils.deleteDirectory(new File(kernelPath));
                props.setProperty(CACHE_COMPILED,String.valueOf(true));
                compileAndLoad(props,compiledAttempts + 1);
            }
            else
                throw new RuntimeException(e);
        }



    }


    private void loadModules(String[] split,String kernelPath) throws Exception {
        for (String module : split) {
            log.info("Loading " + module);
            String path = kernelPath + "output" + File.separator +  module + ".cubin";
            if(!new File(path).exists())
                throw new IllegalStateException("Unable to find path " + path + ". Recompiling");
            String name = module;
            paths.put(name,path);
            KernelLauncher launch = KernelLauncher.load(path, name,"float");
            //reuse the module from the gpu but load the function instead
            KernelLauncher doubleLauncher = KernelLauncher.load(name,"double",launch.getModule());
            launchers.put(Thread.currentThread().getName(),name + "_double", doubleLauncher);
            launchers.put(Thread.currentThread().getName(),name + "_float",launch);
        }

    }


    private void loadCudaKernels() throws IOException {
        Set<String> resources = new Reflections("org.nd4j.nd4j-kernels", new ResourcesScanner()).getResources(Pattern.compile(".*"));
        for(String resource : resources) {
            extract(resource);
        }

        File outputDir = new File(System.getProperty("java.io.tmpdir") + File.separator + "nd4j-kernels","output");
        outputDir.mkdirs();
        log.info("Compiling cuda kernels");
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


}
