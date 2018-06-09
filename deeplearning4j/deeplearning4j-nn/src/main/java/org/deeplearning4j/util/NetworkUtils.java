package org.deeplearning4j.util;

import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.util.StringUtils;
import org.nd4j.versioncheck.VersionCheck;
import org.nd4j.versioncheck.VersionInfo;

import java.util.*;

public class NetworkUtils {

    private NetworkUtils() {
    }

    /**
     * Convert a MultiLayerNetwork to a ComputationGraph
     *
     * @return ComputationGraph equivalent to this network (including parameters and updater state)
     */
    public static ComputationGraph toComputationGraph(MultiLayerNetwork net) {

        //We rely heavily here on the fact that the topological sort order - and hence the layout of parameters - is
        // by definition the identical for a MLN and "single stack" computation graph. This also has to hold
        // for the updater state...

        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
                .graphBuilder();

        MultiLayerConfiguration origConf = net.getLayerWiseConfigurations().clone();


        int layerIdx = 0;
        String lastLayer = "in";
        b.addInputs("in");
        for (NeuralNetConfiguration c : origConf.getConfs()) {
            String currLayer = String.valueOf(layerIdx);

            InputPreProcessor preproc = origConf.getInputPreProcess(layerIdx);
            b.addLayer(currLayer, c.getLayer(), preproc, lastLayer);

            lastLayer = currLayer;
            layerIdx++;
        }
        b.setOutputs(lastLayer);

        ComputationGraphConfiguration conf = b.build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        cg.setParams(net.params());

        //Also copy across updater state:
        INDArray updaterState = net.getUpdater().getStateViewArray();
        if (updaterState != null) {
            cg.getUpdater().getUpdaterStateViewArray()
                    .assign(updaterState);
        }

        return cg;
    }

    /**
     * Set the learning rate for all layers in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration.
     *
     * @param net   Network to set the LR for
     * @param newLr New learning rate for all layers
     */
    public static void setLearningRate(MultiLayerNetwork net, double newLr) {
        setLearningRate(net, newLr, null);
    }

    /**
     * Set the learning rate schedule for all layers in the network to the specified schedule.
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note that the iteration/epoch counts will <i>not</i> be reset. Use {@link MultiLayerConfiguration#setIterationCount(int)}
     * and {@link MultiLayerConfiguration#setEpochCount(int)} if this is required
     *
     * @param newLrSchedule New learning rate schedule for all layers
     */
    public static void setLearningRate(MultiLayerNetwork net, ISchedule newLrSchedule) {
        setLearningRate(net, Double.NaN, newLrSchedule);
    }

    private static void setLearningRate(MultiLayerNetwork net, double newLr, ISchedule lrSchedule) {
        int nLayers = net.getnLayers();
        for (int i = 0; i < nLayers; i++) {
            setLearningRate(net, i, newLr, lrSchedule, false);
        }
        refreshUpdater(net);
    }

    /**
     * Set the learning rate for a single layer in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration. Note also that
     * {@link #setLearningRate(MultiLayerNetwork, double)} should also be used in preference, when all layers need to be set to a new LR
     *
     * @param layerNumber Number of the layer to set the LR for
     * @param newLr       New learning rate for a single layers
     */
    public static void setLearningRate(MultiLayerNetwork net, int layerNumber, double newLr) {
        setLearningRate(net, layerNumber, newLr, null, true);
    }

    /**
     * Set the learning rate schedule for a single layer in the network to the specified value.<br>
     * Note also that {@link #setLearningRate(MultiLayerNetwork, ISchedule)} should also be used in preference, when all layers need
     * to be set to a new LR schedule.<br>
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note also that the iteration/epoch counts will <i>not</i> be reset. Use {@link MultiLayerConfiguration#setIterationCount(int)}
     * and {@link MultiLayerConfiguration#setEpochCount(int)} if this is required
     *
     * @param layerNumber Number of the layer to set the LR schedule for
     * @param lrSchedule  New learning rate for a single layer
     */
    public static void setLearningRate(MultiLayerNetwork net, int layerNumber, ISchedule lrSchedule) {
        setLearningRate(net, layerNumber, Double.NaN, lrSchedule, true);
    }

    private static void setLearningRate(MultiLayerNetwork net, int layerNumber, double newLr, ISchedule newLrSchedule, boolean refreshUpdater) {

        Layer l = net.getLayer(layerNumber).conf().getLayer();
        if (l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            IUpdater u = bl.getIUpdater();
            if (u != null && u.hasLearningRate()) {
                if (newLrSchedule != null) {
                    u.setLrAndSchedule(Double.NaN, newLrSchedule);
                } else {
                    u.setLrAndSchedule(newLr, null);
                }
            }

            //Need to refresh the updater - if we change the LR (or schedule) we may rebuild the updater blocks, which are
            // built by creating blocks of params with the same configuration
            if (refreshUpdater) {
                refreshUpdater(net);
            }
        }
    }

    private static void refreshUpdater(MultiLayerNetwork net) {
        INDArray origUpdaterState = net.getUpdater().getStateViewArray();
        net.setUpdater(null);
        MultiLayerUpdater u = (MultiLayerUpdater) net.getUpdater();
        u.setStateViewArray(origUpdaterState);
    }


    /**
     * Set the learning rate for all layers in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration.
     *
     * @param net   Network to set the LR for
     * @param newLr New learning rate for all layers
     */
    public static void setLearningRate(ComputationGraph net, double newLr) {
        setLearningRate(net, newLr, null);
    }

    /**
     * Set the learning rate schedule for all layers in the network to the specified schedule.
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note that the iteration/epoch counts will <i>not</i> be reset. Use {@link ComputationGraphConfiguration#setIterationCount(int)}
     * and {@link ComputationGraphConfiguration#setEpochCount(int)} if this is required
     *
     * @param newLrSchedule New learning rate schedule for all layers
     */
    public static void setLearningRate(ComputationGraph net, ISchedule newLrSchedule) {
        setLearningRate(net, Double.NaN, newLrSchedule);
    }

    private static void setLearningRate(ComputationGraph net, double newLr, ISchedule lrSchedule) {
        org.deeplearning4j.nn.api.Layer[] layers = net.getLayers();
        for (int i = 0; i < layers.length; i++) {
            setLearningRate(net, layers[i].conf().getLayer().getLayerName(), newLr, lrSchedule, false);
        }
        refreshUpdater(net);
    }

    /**
     * Set the learning rate for a single layer in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration. Note also that
     * {@link #setLearningRate(ComputationGraph, double)} should also be used in preference, when all layers need to be set to a new LR
     *
     * @param layerName Name of the layer to set the LR for
     * @param newLr     New learning rate for a single layers
     */
    public static void setLearningRate(ComputationGraph net, String layerName, double newLr) {
        setLearningRate(net, layerName, newLr, null, true);
    }

    /**
     * Set the learning rate schedule for a single layer in the network to the specified value.<br>
     * Note also that {@link #setLearningRate(ComputationGraph, ISchedule)} should also be used in preference, when all
     * layers need to be set to a new LR schedule.<br>
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note also that the iteration/epoch counts will <i>not</i> be reset. Use {@link ComputationGraphConfiguration#setIterationCount(int)}
     * and {@link ComputationGraphConfiguration#setEpochCount(int)} if this is required
     *
     * @param layerName  Name of the layer to set the LR schedule for
     * @param lrSchedule New learning rate for a single layer
     */
    public static void setLearningRate(ComputationGraph net, String layerName, ISchedule lrSchedule) {
        setLearningRate(net, layerName, Double.NaN, lrSchedule, true);
    }

    private static void setLearningRate(ComputationGraph net, String layerName, double newLr, ISchedule newLrSchedule, boolean refreshUpdater) {

        Layer l = net.getLayer(layerName).conf().getLayer();
        if (l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            IUpdater u = bl.getIUpdater();
            if (u != null && u.hasLearningRate()) {
                if (newLrSchedule != null) {
                    u.setLrAndSchedule(Double.NaN, newLrSchedule);
                } else {
                    u.setLrAndSchedule(newLr, null);
                }
            }

            //Need to refresh the updater - if we change the LR (or schedule) we may rebuild the updater blocks, which are
            // built by creating blocks of params with the same configuration
            if (refreshUpdater) {
                refreshUpdater(net);
            }
        }
    }

    private static void refreshUpdater(ComputationGraph net) {
        INDArray origUpdaterState = net.getUpdater().getStateViewArray();
        net.setUpdater(null);
        ComputationGraphUpdater u = net.getUpdater();
        u.setStateViewArray(origUpdaterState);
    }



    private static final String FORMAT = "%-40s%s";

    public static String generateMemoryStatus(MultiLayerNetwork net){
        StringBuilder sb = genericMemoryStatus();
        String format = "%40s%s";

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

        long sumMem = 0;
        long nParams = net.params().length();
        sb.append("\n----- Network Information -----\n")
                .append(f("Network # Parameters", nParams))
                .append(fBytes("Parameter Memory", bytesPerElement * nParams));
        if(net.getFlattenedGradients() == null){
            sb.append(f("Parameter Gradients Memory", "<not allocated>"));
        } else {
            sumMem += (net.getFlattenedGradients().length() * bytesPerElement);
            sb.append(fBytes("Parameter Gradients Memory", bytesPerElement * net.getFlattenedGradients().length()));
        }
        MultiLayerUpdater u = (MultiLayerUpdater) net.getUpdater(false);
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
        sb.append(f("Workspace Mode: Training", net.getLayerWiseConfigurations().getTrainingWorkspaceMode()));
        sb.append(f("Workspace Mode: Inference", net.getLayerWiseConfigurations().getInferenceWorkspaceMode()));
        appendLayerInformation(sb, net.getLayers());

        appendActivationShapes(net, sb, bytesPerElement);

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
        if(input.rank() < 2 || input.rank() > 4){
            //TODO 3d CNN support
            return;
        }

        sb.append("\n----- Network Activations: Inferred Activation Shapes -----\n");
        InputType inputType = null;
        switch (input.rank()){
            case 2:
                inputType = InputType.feedForward(input.size(1));
                break;
            case 3:
                inputType = InputType.recurrent(input.size(1), input.size(2));
                break;
            case 4:
                inputType = InputType.convolutional(input.size(2), input.size(3), input.size(1));
                break;
        }

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
