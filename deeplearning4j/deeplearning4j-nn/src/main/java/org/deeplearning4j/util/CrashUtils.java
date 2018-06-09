package org.deeplearning4j.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.util.GraphIndices;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.LayerVertex;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.BaseMultiLayerUpdater;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.util.StringUtils;
import org.nd4j.versioncheck.VersionCheck;
import org.nd4j.versioncheck.VersionInfo;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;

import static org.deeplearning4j.nn.conf.inputs.InputType.inferInputType;
import static org.deeplearning4j.nn.conf.inputs.InputType.inferInputTypes;

@Slf4j
public class CrashUtils {

    private CrashUtils(){ }


    public static void writeMemoryCrashDump(Model net, Throwable e){
        long now = System.currentTimeMillis();
        File f = new File("dl4j-memory-crash-dump-" + now + ".txt");
        StringBuilder sb = new StringBuilder();

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
        sb.append("OOM Exception Encountered\nTimestamp: ")
                .append(sdf.format(now)).append("\n\n");

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

        log.warn(">>> Memory crash dump written to: {}", f.getAbsolutePath());
    }

    private static final String FORMAT = "%-40s%s";
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

        sb.append("----- Workspace Information -----\n");
        List<MemoryWorkspace> allWs = Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread();
        sb.append(f("Workspaces: # for current thread", allWs.size()));
        //sb.append(f("Workspaces: # for all threads", allWs.size()));      //TODO
        sb.append("Current thread workspaces:\n");
        //Name, open, size, currently allocated
        String wsFormat = "  %-30s%-20s%-20s";
        sb.append(String.format(wsFormat, "Name", "State", "Size")).append("\n");
        long totalWsSize = 0;
        for(MemoryWorkspace ws : allWs){
            totalWsSize += ws.getCurrentSize();
            sb.append(String.format(wsFormat, ws.getId(),
                    (ws.isScopeActive() ? "OPEN" : "CLOSED"),
                    fBytes(ws.getCurrentSize()))).append("\n");
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
        BaseMultiLayerUpdater u;
        if(isMLN){
            u = (BaseMultiLayerUpdater)mln.getUpdater(false);
        } else {
            u = cg.getUpdater(false);
        }
        if(u == null){
            sb.append(f("Updater","<not initialized>"));
        } else {
            INDArray stateArr = u.getStateViewArray();
            long stateArrLength = (stateArr == null ? 0 : stateArr.length());
            sb.append(f("Updater Number of Elements", stateArrLength));
            sb.append(fBytes("Updater Memory", stateArrLength * bytesPerElement));
            sumMem += stateArrLength * bytesPerElement;
        }
        sb.append(fBytes("Params + Gradient + Updater Memory", sumMem));
        if(isMLN) {
            sb.append(f("Workspace Mode: Training", mln.getLayerWiseConfigurations().getTrainingWorkspaceMode()));
            sb.append(f("Workspace Mode: Inference", mln.getLayerWiseConfigurations().getInferenceWorkspaceMode()));
            appendLayerInformation(sb, mln.getLayers());
            appendActivationShapes(mln, sb, bytesPerElement);
        } else {
            sb.append(f("Workspace Mode: Training", cg.getConfiguration().getTrainingWorkspaceMode()));
            sb.append(f("Workspace Mode: Inference", cg.getConfiguration().getInferenceWorkspaceMode()));
            appendLayerInformation(sb, cg.getLayers());
            appendActivationShapes(cg, sb, bytesPerElement);
        }

        return sb.toString();
    }

    private static String f(String s1, Object o){
        return String.format(FORMAT, s1, (o == null ? "null" : o.toString())) + "\n";
    }

    private static String fBytes(long bytes){
        String s = StringUtils.TraditionalBinaryPrefix.long2String(bytes, "B", 2);
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

        sb.append("\n----- Memory - System Information -----\n");

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


        sb.append("\n----- ND4J Environment Information -----\n");
        sb.append(f("Data Type", Nd4j.dataType()));
        Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
        for(String s : p.stringPropertyNames()){
            sb.append(f(s, p.get(s)));
        }

        return sb;
    }

    private static void appendLayerInformation(StringBuilder sb, org.deeplearning4j.nn.api.Layer[] layers){
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

        //Exclude output layer, include input
        long totalActivationGradMem = totalActivationBytes - last + (ArrayUtil.prodLong(input.shape()) * bytesPerElement);
        sb.append(fBytes("Total Activation Gradient Memory", totalActivationGradMem));
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
