/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.systeminfo;

import static org.nd4j.systeminfo.GPUInfo.fGpu;

import com.jakewharton.byteunits.BinaryByteUnit;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.ServiceLoader;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.SystemUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.versioncheck.VersionCheck;
import org.nd4j.versioncheck.VersionInfo;
import oshi.software.os.OperatingSystem;

/**
 * Utility class to get system info for debugging and error reporting
 */
public class SystemInfo {

    private static void appendField(StringBuilder sb, String name, Object value){
        sb.append(name).append(": ").append(value.toString()).append("\n");
    }

    private static void appendProperty(StringBuilder sb, String name, String property){
        appendField(sb, name, System.getProperty(property));
    }
    
    private static void appendHeader(StringBuilder sb, String name){
        sb.append("\n\n---------------").append(name).append("---------------\n\n");
    }

    private static final String FORMAT = "%-40s%s";

    public static String f(String s1, Object o){
        return String.format(FORMAT, s1, (o == null ? "null" : o.toString())) + "\n";
    }

    public static String fBytes(long bytes){
        String s = BinaryByteUnit.format(bytes, "#.00");
        String format = "%10s";
        s = String.format(format, s);
        if(bytes >= 1024){
            s += " (" + bytes + ")";
        }
        return s;
    }

    public static String fBytes(String s1, long bytes){
        String s = fBytes(bytes);
        return f(s1, s);
    }


    private static void appendCUDAInfo(StringBuilder sb, boolean isWindows){


        sb.append("Nvidia-smi:\n");

        try {
            ProcessBuilder pb = new ProcessBuilder("nvidia-smi");
            appendOutput(sb, pb);
        } catch (IOException e) {
            sb.append("nvidia-smi run failed.");

            if(isWindows) {
                sb.append("  Trying in C:\\Program Files\\NVIDIA Corporation\\NVSMI\n");

                try {
                    ProcessBuilder pb = new ProcessBuilder(
                            "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe");
                    appendOutput(sb, pb);
                } catch (IOException e1) {
                    sb.append("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi run failed\n");
                    sb.append(e1.getMessage());
                }
            } else {
                sb.append("\n");
            }

        }

        sb.append("\nnvcc --version:\n");
        try {

            ProcessBuilder pb = new ProcessBuilder("nvcc", "--version");
            appendOutput(sb, pb);
        } catch (IOException e) {
            sb.append("nvcc --version run failed.");
        }
    }

    private static void appendOutput(StringBuilder sb, ProcessBuilder pb) throws IOException {
        pb.redirectErrorStream(true);
        pb.redirectOutput();

        Process p = pb.start();
        try(InputStreamReader isr = new InputStreamReader(p.getInputStream())) {
            try(BufferedReader reader = new BufferedReader(isr)) {
                String line = null;
                while ((line = reader.readLine()) != null) {
                    sb.append(line);
                    sb.append(System.getProperty("line.separator"));
                }
            }
        }
        sb.append("\n");
    }


    /**
     * Gets system info in a string
     */
    public static String getSystemInfo(){
        StringBuilder sb = new StringBuilder();

        //nd4j info
        appendHeader(sb, "ND4J Info");

        Pair<String,String> pair = inferVersion();
        sb.append(f("Deeplearning4j Version", (pair.getFirst() == null ? "<could not determine>" : pair.getFirst())));
        sb.append(f("Deeplearning4j CUDA", (pair.getSecond() == null ? "<not present>" : pair.getSecond())));

        sb.append("\n");

        boolean isCUDA = false;

        try {
            appendField(sb, "Nd4j Backend", Nd4j.getBackend().getClass().getSimpleName());

            Properties props = Nd4j.getExecutioner().getEnvironmentInformation();

            double memory = ((Long) props.get("memory.available")) / (double) 1024 / 1024 / 1024;
            String fm = String.format("%.1f", memory);
            sb.append("Backend used: [").append(props.get("backend")).append("]; OS: [").append(props.get("os"))
                    .append("]\n");
            sb.append("Cores: [").append(props.get("cores")).append("]; Memory: [").append(fm).append("GB];\n");
            sb.append("Blas vendor: [").append(props.get("blas.vendor")).append("]\n");

            if (Nd4j.getExecutioner().getClass().getSimpleName().equals("CudaExecutioner")) {
                isCUDA = true;

                List<Map<String, Object>> devicesList = (List<Map<String, Object>>) props.get(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY);
                for (Map<String, Object> dev : devicesList) {
                    sb.append("Device Name: [").append(dev.get(Nd4jEnvironment.CUDA_DEVICE_NAME_KEY)).append("]; ")
                            .append("CC: [").append(dev.get(Nd4jEnvironment.CUDA_DEVICE_MAJOR_VERSION_KEY)).append(".")
                            .append(dev.get(Nd4jEnvironment.CUDA_DEVICE_MINOR_VERSION_KEY))
                            .append("]; Total/free memory: [").append(dev.get(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY))
                            .append("]").append("\n");
                }
            }

            sb.append("\nExecutor Properties:\n");

            for (Map.Entry<Object, Object> prop : props.entrySet()) {
                sb.append(prop.getKey().toString()).append("=").append(prop.getValue()).append("\n");
            }
        } catch (Exception e){
            sb.append("Could not get ND4J info\n");
            sb.append("Exception: ").append(e.getMessage()).append("\n");
            sb.append(ExceptionUtils.getStackTrace(e)).append("\n\n");
        }

        //hardware info
        appendHeader(sb, "Hardware Info");

        appendField(sb, "Available processors (cores)",
                Runtime.getRuntime().availableProcessors());

        oshi.SystemInfo sys = new oshi.SystemInfo();
        OperatingSystem os = sys.getOperatingSystem();
        String procName = sys.getHardware().getProcessor().getName();
        long totalMem = sys.getHardware().getMemory().getTotal();

        sb.append(f("Operating System", os.getManufacturer() + " " + os.getFamily() + " " + os.getVersion().getVersion()));
        sb.append(f("CPU", procName));
        sb.append(f("CPU Cores - Physical", sys.getHardware().getProcessor().getPhysicalProcessorCount()));
        sb.append(f("CPU Cores - Logical", sys.getHardware().getProcessor().getLogicalProcessorCount()));
        sb.append(fBytes("Total System Memory", totalMem));

        sb.append("\n");

        boolean hasGPUs = false;

        ServiceLoader<GPUInfoProvider> loader = ND4JClassLoading.loadService(GPUInfoProvider.class);
        Iterator<GPUInfoProvider> iter = loader.iterator();
        if (iter.hasNext()) {
            List<GPUInfo> gpus = iter.next().getGPUs();

            sb.append(f("Number of GPUs Detected", gpus.size()));

            if (!gpus.isEmpty()) {
                hasGPUs = true;
            }

            sb.append(String.format(fGpu, "Name", "CC", "Total Memory", "Used Memory", "Free Memory")).append("\n");

            for (GPUInfo gpuInfo : gpus) {
                sb.append(gpuInfo).append("\n");
            }
        } else {
            sb.append("GPU Provider not found (are you missing nd4j-native?)");
        }

        appendHeader(sb, "CUDA Info");

        if(!isCUDA){
            sb.append("NOT USING CUDA Nd4j\n");

            if(hasGPUs)
                sb.append("GPUs detected, trying to list CUDA info anyways\n");
        }

        if(isCUDA || hasGPUs)
            appendCUDAInfo(sb, SystemUtils.IS_OS_WINDOWS);

        //OS info
        appendHeader(sb, "OS Info");

        appendProperty(sb, "OS" , "os.name");
        appendProperty(sb, "Version","os.version");
        appendProperty(sb, "Arch","os.arch");

        //memory settings
        appendHeader(sb, "Memory Settings");

        appendField(sb, "Free memory (bytes)",
                Runtime.getRuntime().freeMemory());

        long maxMemory = Runtime.getRuntime().maxMemory();
        appendField(sb, "Maximum memory (bytes)",
                (maxMemory == Long.MAX_VALUE ? "No Limit" : maxMemory));

        appendField(sb, "Total memory available to JVM (bytes)",
                Runtime.getRuntime().totalMemory());

        sb.append("\n");

        long xmx = Runtime.getRuntime().maxMemory();
        long jvmTotal = Runtime.getRuntime().totalMemory();
        long javacppMaxPhys = Pointer.maxPhysicalBytes();
        long javacppMaxBytes = Pointer.maxBytes();
        long javacppCurrPhys = Pointer.physicalBytes();
        long javacppCurrBytes = Pointer.totalBytes();
        sb.append(fBytes("JVM Memory: XMX", xmx))
                .append(fBytes("JVM Memory: current", jvmTotal))
                .append(fBytes("JavaCPP Memory: Max Bytes", javacppMaxBytes))
                .append(fBytes("JavaCPP Memory: Max Physical", javacppMaxPhys))
                .append(fBytes("JavaCPP Memory: Current Bytes", javacppCurrBytes))
                .append(fBytes("JavaCPP Memory: Current Physical", javacppCurrPhys));
        boolean periodicGcEnabled = Nd4j.getMemoryManager().isPeriodicGcActive();
        long autoGcWindow = Nd4j.getMemoryManager().getAutoGcWindow();
        sb.append(f("Periodic GC Enabled", periodicGcEnabled));
        if(periodicGcEnabled){
            sb.append(f("Periodic GC Frequency", autoGcWindow + " ms"));
        }

        // Workspaces info

        appendHeader(sb, "Workspace Information");
        List<MemoryWorkspace> allWs = Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread();
        sb.append(f("Workspaces: # for current thread", (allWs == null ? 0 : allWs.size())));
        //sb.append(f("Workspaces: # for all threads", allWs.size()));      //TODO
        long totalWsSize = 0;
        if(allWs != null && allWs.size() > 0) {
            sb.append("Current thread workspaces:\n");
            //Name, open, size, currently allocated
            String wsFormat = "  %-26s%-12s%-30s%-20s";
            sb.append(String.format(wsFormat, "Name", "State", "Size", "# Cycles")).append("\n");
            for (MemoryWorkspace ws : allWs) {
                totalWsSize += ws.getCurrentSize();
                long numCycles = ws.getGenerationId();
                sb.append(String.format(wsFormat, ws.getId(),
                        (ws.isScopeActive() ? "OPEN" : "CLOSED"),
                        fBytes(ws.getCurrentSize()),
                        String.valueOf(numCycles))).append("\n");
            }
        }
        sb.append(fBytes("Workspaces total size", totalWsSize));

        //JVM info
        appendHeader(sb, "JVM Info");

        appendProperty(sb, "Runtime Name", "java.runtime.name");
        appendProperty(sb, "Java Version", "java.version");
        appendProperty(sb, "Runtime Version", "java.runtime.version");
        appendProperty(sb, "Vendor", "java.vm.vendor");
        appendProperty(sb, "Vendor Url", "java.vendor.url");

        sb.append("\n");

        appendProperty(sb, "VM Name", "java.vm.name");
        appendProperty(sb, "VM Version", "java.vm.version");
        appendProperty(sb, "VM Specification Name", "java.vm.specification.name");

        sb.append("\n");

        appendProperty(sb, "Library Path", "java.library.path");

        appendHeader(sb, "Classpath");

        URLClassLoader urlClassLoader = null;

        if (ND4JClassLoading.getNd4jClassloader() instanceof URLClassLoader) {
            urlClassLoader = (URLClassLoader) ND4JClassLoading.getNd4jClassloader();
        } else if (ClassLoader.getSystemClassLoader() instanceof URLClassLoader) {
            urlClassLoader = (URLClassLoader) ClassLoader.getSystemClassLoader();
        } else if (SystemInfo.class.getClassLoader() instanceof URLClassLoader) {
            urlClassLoader = (URLClassLoader) SystemInfo.class.getClassLoader();
        } else if (Thread.currentThread().getContextClassLoader() instanceof URLClassLoader) {
            urlClassLoader = (URLClassLoader) Thread.currentThread().getContextClassLoader();
        } else {
            sb.append("Can't cast class loader to URLClassLoader\n");
        }

        if (urlClassLoader != null) {
            for (URL url : urlClassLoader.getURLs()) {
                sb.append(url.getFile()).append("\n");
            }
        } else {
            sb.append("Using System property java.class.path\n");
            String[] cps = System.getProperty("java.class.path").split(";");
            for(String c : cps){
                sb.append(c).append("\n");
            }
        }

        //launch command
        appendHeader(sb, "Launch Command");

        try{
            // only works on Oracle JVMs
            appendProperty(sb, "Launch Command", "sun.java.command");
        } catch (Exception e){
            appendField(sb, "Launch Command", "Not available on this JVM");
        }

        List<String> inputArguments = ManagementFactory.getRuntimeMXBean().getInputArguments();
        appendField(sb, "JVM Arguments", inputArguments);


        //system properties
        appendHeader(sb, "System Properties");

        Properties props = System.getProperties();
        for(Map.Entry<Object, Object> prop : props.entrySet()){
            if(prop.getKey().toString().equals("line.separator")) {
                sb.append(prop.getKey().toString()).append("=")
                        .append(prop.getValue().toString().replace("\\", "\\\\")).append("\n");
            } else {
                sb.append(prop.getKey().toString()).append("=").append(prop.getValue()).append("\n");
            }
        }


        //enviroment variables
        appendHeader(sb, "Environment Variables");

        Map<String, String> env = System.getenv();

        for(String key : env.keySet()){
            sb.append(key).append("=").append(env.get(key)).append("\n");
        }

        return sb.toString();
    }

    /**
     * Writes system info to the given file
     */
    public static void writeSystemInfo(File file){
        try {
            file.createNewFile();
            FileUtils.writeStringToFile(file, getSystemInfo());
        } catch (IOException e) {
            throw new RuntimeException("IOException:" + e.getMessage(), e);
        }
    }

    /**
     * Prints system info
     */
    public static void printSystemInfo(){
        System.out.println(getSystemInfo());
    }

    public static Pair<String,String> inferVersion(){
        List<VersionInfo> vi = VersionCheck.getVersionInfos();

        String dl4jVersion = null;
        String dl4jCudaArtifact = null;
        for(VersionInfo v : vi){
            if("org.deeplearning4j".equals(v.getGroupId()) && "deeplearning4j-core".equals(v.getArtifactId())){
                String version = v.getBuildVersion();
                if(version.contains("SNAPSHOT")){
                    dl4jVersion = version + " (" + v.getCommitIdAbbrev() + ")";
                }
                dl4jVersion = version;
            } else if("org.deeplearning4j".equals(v.getGroupId()) && v.getArtifactId() != null && v.getArtifactId().contains("deeplearning4j-cuda")){
                dl4jCudaArtifact = v.getArtifactId();
            }

        }

        return new Pair<>(dl4jVersion, dl4jCudaArtifact);
    }
}
