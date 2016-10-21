package org.deeplearning4j.ui.module.training;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.flow.beans.Description;
import org.deeplearning4j.ui.flow.beans.LayerInfo;
import org.deeplearning4j.ui.flow.beans.ModelInfo;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Alex on 17/10/2016.
 */
public class TrainModuleUtils {

    private static final List<String> colors = Collections.unmodifiableList(Arrays.asList("#9966ff", "#ff9933", "#ffff99", "#3366ff", "#0099cc", "#669999", "#66ffff"));
    public static final String INPUT = "INPUT";

    private static final String FORMAT = "%02d:%02d:%02d";
    public static final String LOCALHOST = "localhost";

    public static ModelInfo buildModelInfo(MultiLayerConfiguration conf) {
        ModelInfo modelInfo = new ModelInfo();
        {

            LayerInfo info = new LayerInfo();
            info.setId(0);
            info.setName("Input");
            info.setY(0);
            info.setX(0);
            info.setLayerType(INPUT);
            info.setDescription(new Description());
            info.getDescription().setMainLine("Model input");
            info.getDescription().setText("");
            info.addConnection(0, 1);
            modelInfo.addLayer(info);


            // entry 0 is reserved for inputs
            int y = 1;

            // for MLN x value is always 0
            final int x = 0;
            int nLayers = conf.getConfs().size();
            for (int i = 0; i < nLayers; i++) {
                org.deeplearning4j.nn.conf.layers.Layer layer = conf.getConf(i).getLayer();
                LayerInfo layerInfo = getLayerInfo(null, layer, x, y, y);
                // since it's MLN, we know connections in advance as curLayer + 1
                layerInfo.addConnection(x, y + 1);
                modelInfo.addLayer(layerInfo);
                y++;
            }

            LayerInfo layerInfo = modelInfo.getLayerInfoByCoords(x, y - 1);
            layerInfo.dropConnections();

        }
        // find layers without connections, and mark them as output layers
        for (LayerInfo layerInfo : modelInfo.getLayers()) {
            if (layerInfo.getConnections().size() == 0) layerInfo.setLayerType("OUTPUT");
        }

        // now we apply colors to distinct layer types
        AtomicInteger cnt = new AtomicInteger(0);
        for (String layerType : modelInfo.getLayerTypes()) {
            String curColor = colors.get(cnt.getAndIncrement());
            if (cnt.get() >= colors.size()) cnt.set(0);
            for (LayerInfo layerInfo : modelInfo.getLayersByType(layerType)) {
                if (layerType.equals(INPUT)) {
                    layerInfo.setColor("#99ff66");
                } else if (layerType.equals("OUTPUT")) {
                    layerInfo.setColor("#e6e6e6");
                } else {
                    layerInfo.setColor(curColor);
                }
            }
        }
        return modelInfo;
    }

    public static ModelInfo buildModelInfo2(ComputationGraphConfiguration conf){
        Map<String,GraphVertex> gv = conf.getVertices();

        ModelInfo mi = new ModelInfo();


        //FIRST: work out depths
        Map<String,Integer> vertexDepths = getDepths(conf);

        //Second: given depths, go LTR, TTB
        int maxDepth = 0;
        for(Integer i : vertexDepths.values()){
            maxDepth = Math.max(i, maxDepth);
        }

        int order = 0;
        for( int y=0; y<=maxDepth; y++ ){
            List<String> allForCurrentDepth = new ArrayList<>();
            for(Map.Entry<String,Integer> entry : vertexDepths.entrySet()){
                if(entry.getValue() == y){
                    allForCurrentDepth.add(entry.getKey());
                }
            }

            Collections.sort(allForCurrentDepth);
            int x = 0;
            System.out.println("...");
            for(String s : allForCurrentDepth){
                GraphVertex g = gv.get(s);
                Layer l = null;
                if(g instanceof LayerVertex){
                    l = ((LayerVertex) g).getLayerConf().getLayer();
                }
                LayerInfo info = getLayerInfo(s, l, x++, y, order++);
                mi.addLayer(info);
            }
        }

        System.out.println(vertexDepths);

        return mi;
    }

    private static Map<String,Integer> getDepths(ComputationGraphConfiguration conf){

        Map<String,Integer> depth = new HashMap<>();

        List<String> inputs = conf.getNetworkInputs();
        for(String s : inputs){
            depth.put(s, 0);
        }

        Map<String,List<String>> vertexInputs = conf.getVertexInputs();
        int totalNumVertices = conf.getNetworkInputs().size() + vertexInputs.size();
        int currentDepth = 1;
        while(depth.size() != totalNumVertices){
            int countProcessed = 0;
            Set<String> addedThisRound = new HashSet<>();
            for(Map.Entry<String,List<String>> vInputs : vertexInputs.entrySet()){
                String name = vInputs.getKey();
                if(depth.containsKey(name)) continue;   //Already processed

                List<String> currInputs = vInputs.getValue();
                //If we've seen all these inputs: goes at the current level...
                boolean allInputsSeen = true;
                for(String s : currInputs){
                    if(!depth.containsKey(s) || addedThisRound.contains(s)){
                        allInputsSeen = false;
                        break;
                    }
                }

                if(allInputsSeen){
                    depth.put(name, currentDepth);
                    addedThisRound.add(name);
                    countProcessed++;
                }
            }

            if(countProcessed == 0){
                throw new RuntimeException("Invalid graph structure?"); //TODO
            }
            currentDepth++;
        }

        return depth;
    }



    private static LayerInfo getLayerInfo(String name, Layer layer, int x, int y, int order) {
        LayerInfo info = new LayerInfo();

        // set coordinates
        info.setX(x);
        info.setY(y);

        // if name was set, we should grab it
        if(name != null){
            info.setName(name);
        } else {
            try {
                info.setName(layer.getLayerName());
            } catch (Exception e) {
            }
            if (info.getName() == null || info.getName().isEmpty()) info.setName("unnamed");
        }

        // unique layer id required here
        info.setId(order);

        // set layer description according to layer params
        Description description = new Description();
        info.setDescription(description);

        // set layer type
        if( layer == null){
            info.setLayerType("INPUT"); //TODO OTHER GRAPH VERTEX TYPES
        } else {
            try {
                info.setLayerType(layer.getClass().getSimpleName().replaceAll("Layer$", ""));
            } catch (Exception e) {
                e.printStackTrace();
                info.setLayerType("n/a");
                return info;
            }
        }


        StringBuilder mainLine = new StringBuilder();
        StringBuilder subLine = new StringBuilder();
        StringBuilder fullLine = new StringBuilder();

        //    log.info("Layer: " + info.getName() + " class: " + layer.getClass().getSimpleName());

        if(layer == null){
            mainLine.append("INPUT");   //TODO

        } else {
            if (layer instanceof ConvolutionLayer) {
                org.deeplearning4j.nn.conf.layers.ConvolutionLayer layer1 = (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) layer;
                mainLine.append("K: " + Arrays.toString(layer1.getKernelSize()) + " S: " + Arrays.toString(layer1.getStride()) + " P: " + Arrays.toString(layer1.getPadding()));
                subLine.append("nIn/nOut: [" + layer1.getNIn() + "/" + layer1.getNOut() + "]");
                fullLine.append("Kernel size: ").append(Arrays.toString(layer1.getKernelSize())).append("<br/>");
                fullLine.append("Stride: ").append(Arrays.toString(layer1.getStride())).append("<br/>");
                fullLine.append("Padding: ").append(Arrays.toString(layer1.getPadding())).append("<br/>");
                fullLine.append("Inputs number: ").append(layer1.getNIn()).append("<br/>");
                fullLine.append("Outputs number: ").append(layer1.getNOut()).append("<br/>");
            } else if (layer instanceof SubsamplingLayer) {
                SubsamplingLayer layer1 = (SubsamplingLayer) layer;
                fullLine.append("Kernel size: ").append(Arrays.toString(layer1.getKernelSize())).append("<br/>");
                fullLine.append("Stride: ").append(Arrays.toString(layer1.getStride())).append("<br/>");
                fullLine.append("Padding: ").append(Arrays.toString(layer1.getPadding())).append("<br/>");
                fullLine.append("Pooling type: ").append(layer1.getPoolingType().toString()).append("<br/>");
            } else if (layer instanceof FeedForwardLayer) {
                org.deeplearning4j.nn.conf.layers.FeedForwardLayer layer1 = (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) layer;
                mainLine.append("nIn/nOut: [" + layer1.getNIn() + "/" + layer1.getNOut() + "]");
                subLine.append(info.getLayerType());
                fullLine.append("Inputs number: ").append(layer1.getNIn()).append("<br/>");
                fullLine.append("Outputs number: ").append(layer1.getNOut()).append("<br/>");
            } else {
                // TODO: Introduce Layer.Type.OUTPUT
                if (layer instanceof BaseOutputLayer) {
                    mainLine.append("Outputs: [" + ((org.deeplearning4j.nn.conf.layers.BaseOutputLayer) layer).getNOut() + "]");
                    fullLine.append("Outputs number: ").append(((org.deeplearning4j.nn.conf.layers.BaseOutputLayer) layer).getNOut()).append("<br/>");
                }
            }
        }

        subLine.append(" A: [").append(layer.getActivationFunction()).append("]");
        fullLine.append("Activation function: ").append("<b>").append(layer.getActivationFunction()).append("</b>").append("<br/>");

        description.setMainLine(mainLine.toString());
        description.setSubLine(subLine.toString());
        description.setText(fullLine.toString());

        return info;
    }
}
