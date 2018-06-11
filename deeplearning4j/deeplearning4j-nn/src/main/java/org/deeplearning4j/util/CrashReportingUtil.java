package org.deeplearning4j.util;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.util.GraphIndices;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.LayerVertex;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.BaseMultiLayerUpdater;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.updater.UpdaterBlock;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.solvers.BaseOptimizer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.util.StringUtils;
import org.nd4j.versioncheck.VersionCheck;
import org.nd4j.versioncheck.VersionInfo;
import oshi.SystemInfo;
import oshi.hardware.HardwareAbstractionLayer;
import oshi.software.os.OperatingSystem;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.text.SimpleDateFormat;
import java.util.*;

import static org.deeplearning4j.nn.conf.inputs.InputType.inferInputType;
import static org.deeplearning4j.nn.conf.inputs.InputType.inferInputTypes;

/**
 * A utility for generating crash reports when an out of memory error occurs.
 *
 * @author Alex Black
 */
@Slf4j
public class CrashReportingUtil {

    /**
     * System property that can be used to enable or disable memory crash reporting. Memory crash reporting is
     * enabled by default.
     */
    public static final String CRASH_DUMP_ENABLED_PROPERTY = "org.deeplearning4j.crash.reporting.enabled";

    /**
     * System property that can be use to customize the output directory for memory crash reporting. By default,
     * the current working directory will be used
     */
    public static final String CRASH_DUMP_OUTPUT_DIRECTORY_PROPERTY = "org.deeplearning4j.crash.reporting.directory";

    @Getter
    private static boolean crashDumpsEnabled = true;
    @Getter
    private static File crashDumpRootDirectory;

    static {
        String s = System.getProperty(CRASH_DUMP_ENABLED_PROPERTY);
        if(s != null && !s.isEmpty()){
            crashDumpsEnabled = Boolean.parseBoolean(s);
        }

        s = System.getProperty(CRASH_DUMP_OUTPUT_DIRECTORY_PROPERTY);
        boolean setDir = false;
        if(s != null && !s.isEmpty()){
            try{
                File f = new File(s);
                crashDumpOutputDirectory(f);
                setDir = true;
                log.debug("Crash dump output directory set to: {}", f.getAbsolutePath());
            } catch (Exception e){
                log.warn("Error setting crash dump output directory to value: {}", s, e);
            }
        }
        if(!setDir){
            crashDumpOutputDirectory(null);
        }
    }

    private CrashReportingUtil(){ }

    /**
     * Method that can be used to enable or disable memory crash reporting. Memory crash reporting is enabled by default.
     */
    public static void crashDumpsEnabled(boolean enabled){
        crashDumpsEnabled = enabled;
    }

    /**
     * Method that can be use to customize the output directory for memory crash reporting. By default,
     * the current working directory will be used.
     *
     * @param rootDir Root directory to use for crash reporting. If null is passed, the current working directory
     *                will be used
     */
    public static void crashDumpOutputDirectory(File rootDir){
        if(rootDir == null){
            String userDir = System.getProperty("user.dir");
            if(userDir == null){
                userDir = "";
            }
            crashDumpRootDirectory = new File(userDir);
            return;
        }
        crashDumpRootDirectory = rootDir;
    }

    /**
     * Generate and write the crash dump to the crash dump root directory (by default, the working directory).
     * Naming convention for crash dump files: "dl4j-memory-crash-dump-<timestamp>_<thread-id>.txt"
     *
     *
     * @param net   Net to generate the crash dump for. May not be null
     * @param e     Throwable/exception. Stack trace will be included in the network output
     */
    public static void writeMemoryCrashDump(@NonNull Model net, @NonNull Throwable e){
        if(!crashDumpsEnabled){
            return;
        }

        long now = System.currentTimeMillis();
        long tid = Thread.currentThread().getId();      //Also add thread ID to avoid name clashes (parallel wrapper, etc)
        String threadName = Thread.currentThread().getName();
        crashDumpRootDirectory.mkdirs();
        File f = new File(crashDumpRootDirectory, "dl4j-memory-crash-dump-" + now + "_" + tid + ".txt");
        StringBuilder sb = new StringBuilder();

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
        sb.append("Deeplearning4j OOM Exception Encountered for ").append(net.getClass().getSimpleName()).append("\n")
                .append(f("Timestamp: ", sdf.format(now)))
                .append(f("Thread ID", tid))
                .append(f("Thread Name", threadName))
                .append("\n\n");

        sb.append("Stack Trace:\n")
                .append(ExceptionUtils.getStackTrace(e));

        try{
            sb.append("\n\n");
            sb.append(generateMemoryStatus(net));
        } catch (Throwable t){
            sb.append("<Error generating network memory status information section>")
                    .append(ExceptionUtils.getStackTrace(t));
        }

        String toWrite = sb.toString();
        try{
            FileUtils.writeStringToFile(f, toWrite);
        } catch (IOException e2){
            log.error("Error writing memory crash dump information to disk: {}", f.getAbsolutePath(), e2);
        }

        log.error(">>> Out of Memory Exception Detected. Memory crash dump written to: {}", f.getAbsolutePath());
        log.warn("Memory crash dump reporting can be disabled with CrashUtil.crashDumpsEnabled(false) or using system " +
                "property -D" + CRASH_DUMP_ENABLED_PROPERTY + "=false");
        log.warn("Memory crash dump reporting output location can be set with CrashUtil.crashDumpOutputDirectory(File) or using system " +
                "property -D" + CRASH_DUMP_OUTPUT_DIRECTORY_PROPERTY + "=<path>");
    }

    private static final String FORMAT = "%-40s%s";

    /**
     * Generate memory/system report as a String, for the specified network.
     * Includes informatioun about the system, memory configuration, network, etc.
     *
     * @param net   Net to generate the report for
     * @return Report as a String
     */
    public static String generateMemoryStatus(Model net){
        MultiLayerNetwork mln = null;
        ComputationGraph cg = null;
        boolean isMLN;
        if(net instanceof MultiLayerNetwork){
            mln = (MultiLayerNetwork)net;
            isMLN = true;
        } else {
            cg = (ComputationGraph)net;
            isMLN = false;
        }

        StringBuilder sb = genericMemoryStatus();

        int bytesPerElement;
        switch (Nd4j.dataType()){
            case DOUBLE:
                bytesPerElement = 8;
                break;
            case FLOAT:
                bytesPerElement = 4;
                break;
            case HALF:
                bytesPerElement = 2;
                break;
            default:
                bytesPerElement = 0;    //TODO
        }

        sb.append("\n----- Workspace Information -----\n");
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
        Map<String,Pointer> helperWorkspaces;
        if(isMLN){
            helperWorkspaces = mln.getHelperWorkspaces();
        } else {
            helperWorkspaces = cg.getHelperWorkspaces();
        }
        if(helperWorkspaces != null && !helperWorkspaces.isEmpty()){
            boolean header = false;
            for(Map.Entry<String,Pointer> e : helperWorkspaces.entrySet()){
                Pointer p = e.getValue();
                if(p == null){
                    continue;
                }
                if(!header){
                    sb.append("Helper Workspaces\n");
                    header = true;
                }
                sb.append("  ").append(fBytes(e.getKey(), p.capacity()));
            }
        }

        long sumMem = 0;
        long nParams = net.params().length();
        sb.append("\n----- Network Information -----\n")
                .append(f("Network # Parameters", nParams))
                .append(fBytes("Parameter Memory", bytesPerElement * nParams));
        INDArray flattenedGradients;
        if(isMLN){
            flattenedGradients = mln.getFlattenedGradients();
        } else {
            flattenedGradients = cg.getFlattenedGradients();
        }
        if(flattenedGradients == null){
            sb.append(f("Parameter Gradients Memory", "<not allocated>"));
        } else {
            sumMem += (flattenedGradients.length() * bytesPerElement);
            sb.append(fBytes("Parameter Gradients Memory", bytesPerElement * flattenedGradients.length()));
        }
            //Updater info
        BaseMultiLayerUpdater u;
        if(isMLN){
            u = (BaseMultiLayerUpdater)mln.getUpdater(false);
        } else {
            u = cg.getUpdater(false);
        }
        Set<String> updaterClasses = new HashSet<>();
        if(u == null){
            sb.append(f("Updater","<not initialized>"));
        } else {
            INDArray stateArr = u.getStateViewArray();
            long stateArrLength = (stateArr == null ? 0 : stateArr.length());
            sb.append(f("Updater Number of Elements", stateArrLength));
            sb.append(fBytes("Updater Memory", stateArrLength * bytesPerElement));
            sumMem += stateArrLength * bytesPerElement;

            List<UpdaterBlock> blocks = u.getUpdaterBlocks();
            for(UpdaterBlock ub : blocks){
                updaterClasses.add(ub.getGradientUpdater().getClass().getName());
            }

            sb.append("Updater Classes:").append("\n");
            for(String s : updaterClasses ){
                sb.append("  ").append(s).append("\n");
            }
        }
        sb.append(fBytes("Params + Gradient + Updater Memory", sumMem));
            //Iter/epoch
        sb.append(f("Iteration Count", BaseOptimizer.getIterationCount(net)));
        sb.append(f("Epoch Count", BaseOptimizer.getEpochCount(net)));

            //Workspaces, backprop type, layer info, activation info, helper info
        if(isMLN) {
            sb.append(f("Backprop Type", mln.getLayerWiseConfigurations().getBackpropType()));
            if(mln.getLayerWiseConfigurations().getBackpropType() == BackpropType.TruncatedBPTT){
                sb.append(f("TBPTT Length", mln.getLayerWiseConfigurations().getTbpttFwdLength() + "/" + mln.getLayerWiseConfigurations().getTbpttBackLength()));
            }
            sb.append(f("Workspace Mode: Training", mln.getLayerWiseConfigurations().getTrainingWorkspaceMode()));
            sb.append(f("Workspace Mode: Inference", mln.getLayerWiseConfigurations().getInferenceWorkspaceMode()));
            appendLayerInformation(sb, mln.getLayers(), bytesPerElement);
            appendHelperInformation(sb, mln.getLayers());
            appendActivationShapes(mln, sb, bytesPerElement);
        } else {
            sb.append(f("Backprop Type", cg.getConfiguration().getBackpropType()));
            if(cg.getConfiguration().getBackpropType() == BackpropType.TruncatedBPTT){
                sb.append(f("TBPTT Length", cg.getConfiguration().getTbpttFwdLength() + "/" + cg.getConfiguration().getTbpttBackLength()));
            }
            sb.append(f("Workspace Mode: Training", cg.getConfiguration().getTrainingWorkspaceMode()));
            sb.append(f("Workspace Mode: Inference", cg.getConfiguration().getInferenceWorkspaceMode()));
            appendLayerInformation(sb, cg.getLayers(), bytesPerElement);
            appendHelperInformation(sb, cg.getLayers());
            appendActivationShapes(cg, sb, bytesPerElement);
        }

        //Listener info:
        Collection<TrainingListener> listeners;
        if(isMLN){
            listeners = mln.getListeners();
        } else {
            listeners = cg.getListeners();
        }

        sb.append("\n----- Network Training Listeners -----\n");
        sb.append(f("Number of Listeners", (listeners == null ? 0 : listeners.size())));
        int lCount = 0;
        if(listeners != null && !listeners.isEmpty()){
            for(TrainingListener tl : listeners) {
                sb.append(f("Listener " + (lCount++), tl));
            }
        }
        
        return sb.toString();
    }

    private static String f(String s1, Object o){
        return String.format(FORMAT, s1, (o == null ? "null" : o.toString())) + "\n";
    }

    private static String fBytes(long bytes){
        String s = StringUtils.TraditionalBinaryPrefix.long2String(bytes, "B", 2);
        String format = "%10s";
        s = String.format(format, s);
        if(bytes >= 1024){
            s += " (" + bytes + ")";
        }
        return s;
    }

    private static String fBytes(String s1, long bytes){
        String s = fBytes(bytes);
        return f(s1, s);
    }

    private static StringBuilder genericMemoryStatus(){

        StringBuilder sb = new StringBuilder();

        sb.append("========== Memory Information ==========\n");
        sb.append("----- Version Information -----\n");
        Pair<String,String> pair = inferVersion();
        sb.append(f("Deeplearning4j Version", (pair.getFirst() == null ? "<could not determine>" : pair.getFirst())));
        sb.append(f("Deeplearning4j CUDA", (pair.getSecond() == null ? "<not present>" : pair.getSecond())));

        sb.append("\n----- System Information -----\n");
        SystemInfo sys = new SystemInfo();
        OperatingSystem os = sys.getOperatingSystem();
        String procName = sys.getHardware().getProcessor().getName();
        long totalMem = sys.getHardware().getMemory().getTotal();

        sb.append(f("Operating System", os.getManufacturer() + " " + os.getFamily() + " " + os.getVersion().getVersion()));
        sb.append(f("CPU", procName));
        sb.append(f("CPU Cores - Physical", sys.getHardware().getProcessor().getPhysicalProcessorCount()));
        sb.append(f("CPU Cores - Logical", sys.getHardware().getProcessor().getLogicalProcessorCount()));
        sb.append(fBytes("Total System Memory", totalMem));

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int nDevices = nativeOps.getAvailableDevices();
        if (nDevices > 0) {
            sb.append(f("Number of GPUs Detected", nDevices));
            //Name CC, Total memory, current memory, free memory
            String fGpu = "  %-30s %-5s %24s %24s %24s";
            sb.append(String.format(fGpu, "Name", "CC", "Total Memory", "Used Memory", "Free Memory")).append("\n");
            for (int i = 0; i < nDevices; i++) {
                try {
                    Class<?> c = Class.forName("org.nd4j.jita.allocator.pointers.CudaPointer");
                    Constructor<?> constructor = c.getConstructor(long.class);
                    Pointer p = (Pointer) constructor.newInstance((long) i);
                    String name = nativeOps.getDeviceName(p);
                    long total = nativeOps.getDeviceTotalMemory(p);
                    long free = nativeOps.getDeviceFreeMemory(p);
                    long current = total - free;
                    int major = nativeOps.getDeviceMajor(p);
                    int minor = nativeOps.getDeviceMinor(p);

                    sb.append(String.format(fGpu, name, major + "." + minor, fBytes(total), fBytes(current), fBytes(free))).append("\n");
                } catch (Exception e) {
                    sb.append("  Failed to get device info for device ").append(i).append("\n");
                }
            }
        }

        sb.append("\n----- ND4J Environment Information -----\n");
        sb.append(f("Data Type", Nd4j.dataType()));
        Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
        for(String s : p.stringPropertyNames()){
            sb.append(f(s, p.get(s)));
        }

        sb.append("\n----- Memory Configuration -----\n");

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




        return sb;
    }

    private static void appendLayerInformation(StringBuilder sb, org.deeplearning4j.nn.api.Layer[] layers, int bytesPerElement){
        Map<String,Integer> layerClasses = new HashMap<>();
        for(org.deeplearning4j.nn.api.Layer l : layers){
            if(!layerClasses.containsKey(l.getClass().getSimpleName())){
                layerClasses.put(l.getClass().getSimpleName(), 0);
            }
            layerClasses.put(l.getClass().getSimpleName(), layerClasses.get(l.getClass().getSimpleName()) + 1);
        }

        List<String> l = new ArrayList<>(layerClasses.keySet());
        Collections.sort(l);
        sb.append(f("Number of Layers", layers.length));
        sb.append("Layer Counts\n");
        for(String s : l){
            sb.append("  ").append(f(s, layerClasses.get(s)));
        }
        sb.append("Layer Parameter Breakdown\n");
        String format = "  %-3s %-20s %-20s %-20s %-20s";
        sb.append(String.format(format, "Idx", "Name", "Layer Type", "Layer # Parameters", "Layer Parameter Memory")).append("\n");
        for(Layer layer : layers){
            long numParams = layer.numParams();
            sb.append(String.format(format, layer.getIndex(), layer.conf().getLayer().getLayerName(),
                    layer.getClass().getSimpleName(), String.valueOf(numParams), fBytes(numParams * bytesPerElement))).append("\n");
        }

    }

    private static void appendHelperInformation(StringBuilder sb, org.deeplearning4j.nn.api.Layer[] layers){
        sb.append("\n----- Layer Helpers - Memory Use -----\n");

        int helperCount = 0;
        long helperWithMemCount = 0L;
        long totalHelperMem = 0L;

        //Layer index, layer name, layer class, helper class, total memory, breakdown
        String format = "%-3s %-20s %-25s %-30s %-12s %s";
        boolean header = false;
        for(Layer l : layers){
            LayerHelper h = l.getHelper();
            if(h == null)
                continue;

            helperCount++;
            Map<String,Long> mem = h.helperMemoryUse();
            if(mem == null || mem.isEmpty())
                continue;
            helperWithMemCount++;

            long layerTotal = 0;
            for(Long m : mem.values()){
                layerTotal += m;
            }

            int idx = l.getIndex();
            String layerName = l.conf().getLayer().getLayerName();
            if(layerName == null)
                layerName = String.valueOf(idx);


            if(!header){
                sb.append(String.format(format, "#", "Layer Name", "Layer Class", "Helper Class", "Total Memory", "Memory Breakdown"))
                        .append("\n");
                header = true;
            }

            sb.append(String.format(format, idx, layerName, l.getClass().getSimpleName(), h.getClass().getSimpleName(),
                    fBytes(layerTotal), mem.toString())).append("\n");

            totalHelperMem += layerTotal;
        }

        sb.append(f("Total Helper Count", helperCount));
        sb.append(f("Helper Count w/ Memory", helperWithMemCount));
        sb.append(fBytes("Total Helper Persistent Memory Use", totalHelperMem));
    }

    private static void appendActivationShapes(MultiLayerNetwork net, StringBuilder sb, int bytesPerElement){
        INDArray input = net.getInput();
        if(input == null){
            return;
        }

        sb.append("\n----- Network Activations: Inferred Activation Shapes -----\n");
        InputType inputType = inferInputType(input);

        sb.append(f("Current Minibatch Size", input.size(0)));
        sb.append(f("Current Input Shape", Arrays.toString(input.shape())));
        List<InputType> inputTypes = net.getLayerWiseConfigurations().getLayerActivationTypes(inputType);
        String format = "%-3s %-20s %-20s %-42s %-20s %-12s %-12s";
        sb.append(String.format(format, "Idx", "Name", "Layer Type", "Activations Type", "Activations Shape",
                "# Elements", "Memory")).append("\n");
        org.deeplearning4j.nn.api.Layer[] layers = net.getLayers();
        long totalActivationBytes = 0;
        long last = 0;
        for( int i=0; i<inputTypes.size(); i++ ){
            long[] shape = inputTypes.get(i).getShape(true);
            if(shape[0] <= 0){
                shape[0] = input.size(0);
            }
            long numElements = ArrayUtil.prodLong(shape);
            long bytes = numElements*bytesPerElement;
            totalActivationBytes += bytes;
            sb.append(String.format(format, String.valueOf(i), layers[i].conf().getLayer().getLayerName(), layers[i].getClass().getSimpleName(),
                    inputTypes.get(i), Arrays.toString(shape), String.valueOf(numElements), fBytes(bytes))).append("\n");
            last = bytes;
        }
        sb.append(fBytes("Total Activations Memory", totalActivationBytes));
        sb.append(fBytes("Total Activations Memory (per ex)", totalActivationBytes / input.size(0)));

        //Exclude output layer, include input
        long totalActivationGradMem = totalActivationBytes - last + (ArrayUtil.prodLong(input.shape()) * bytesPerElement);
        sb.append(fBytes("Total Activation Gradient Mem.", totalActivationGradMem));
        sb.append(fBytes("Total Activation Gradient Mem. (per ex)", totalActivationGradMem / input.size(0)));
    }

    private static void appendActivationShapes(ComputationGraph net, StringBuilder sb, int bytesPerElement){
        INDArray[] input = net.getInputs();
        if(input == null){
            return;
        }
        for( int i=0; i<input.length; i++ ) {
            if (input[i] == null) {
                return;
            }
        }

        sb.append("\n----- Network Activations: Inferred Activation Shapes -----\n");
        InputType[] inputType = inferInputTypes(input);

        sb.append(f("Current Minibatch Size", input[0].size(0)));
        for( int i=0; i<input.length; i++ ) {
            sb.append(f("Current Input Shape (Input " + i + ")", Arrays.toString(input[i].shape())));
        }
        Map<String,InputType> inputTypes = net.getConfiguration().getLayerActivationTypes(inputType);
        GraphIndices indices = net.calculateIndices();

        String format = "%-3s %-20s %-20s %-42s %-20s %-12s %-12s";
        sb.append(String.format(format, "Idx", "Name", "Layer Type", "Activations Type", "Activations Shape",
                "# Elements", "Memory")).append("\n");
        org.deeplearning4j.nn.api.Layer[] layers = net.getLayers();
        long totalActivationBytes = 0;
        long totalExOutput = 0; //Implicitly includes input already due to input vertices
        int[] topo = indices.getTopologicalSortOrder();
        for( int i=0; i<topo.length; i++ ){
            String layerName = indices.getIdxToName().get(i);
            GraphVertex gv = net.getVertex(layerName);

            InputType it = inputTypes.get(layerName);
            long[] shape = it.getShape(true);
            if(shape[0] <= 0){
                shape[0] = input[0].size(0);
            }
            long numElements = ArrayUtil.prodLong(shape);
            long bytes = numElements*bytesPerElement;
            totalActivationBytes += bytes;
            String className;
            if(gv instanceof LayerVertex){
                className = gv.getLayer().getClass().getSimpleName();
            } else {
                className = gv.getClass().getSimpleName();
            }

            sb.append(String.format(format, String.valueOf(i), layerName, className, it,
                    Arrays.toString(shape), String.valueOf(numElements), fBytes(bytes))).append("\n");

            if(!net.getConfiguration().getNetworkOutputs().contains(layerName)){
                totalExOutput += bytes;
            }
        }
        sb.append(fBytes("Total Activations Memory", totalActivationBytes));
        sb.append(fBytes("Total Activation Gradient Memory", totalExOutput));
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
