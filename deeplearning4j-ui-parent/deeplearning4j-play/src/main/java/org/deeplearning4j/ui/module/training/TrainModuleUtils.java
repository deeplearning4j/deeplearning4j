package org.deeplearning4j.ui.module.training;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.ui.flow.beans.Description;
import org.deeplearning4j.ui.flow.beans.LayerInfo;
import org.deeplearning4j.ui.flow.beans.ModelInfo;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
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
                LayerInfo layerInfo = getLayerInfo(layer, x, y, y);
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

    /*
    protected ModelInfo buildModelInfo(ComputationGraphConfiguration conf) {
        ModelInfo modelInfo = new ModelInfo();
        {

//                we assume that graph starts on input. every layer connected to input - is on y1
//                every layer connected to y1, is on y2 etc.
            List<String> inputs = conf.getNetworkInputs();
            // now we need to add inputs as y0 nodes
            int x = 0;
            for (String input : inputs) {
//                GraphVertex vertex = graph.getVertex(input);
//                INDArray gInput = vertex.getInputs()[0];
//                long tadLength = Shape.getTADLength(gInput.shape(), ArrayUtil.range(1,gInput.rank()));

//                long numSamples = gInput.lengthLong() / tadLength;

                StringBuilder builder = new StringBuilder();
                builder.append("Vertex name: ").append(input).append("<br/>");
                builder.append("Model input").append("<br/>");
//                builder.append("Input size: ").append(tadLength).append("<br/>");
//                builder.append("Batch size: ").append(numSamples).append("<br/>");

                LayerInfo info = new LayerInfo();
                info.setId(0);
                info.setName(input);
                info.setY(0);
                info.setX(x);
                info.setLayerType(INPUT);
                info.setDescription(new Description());
                info.getDescription().setMainLine("Model input");
                info.getDescription().setText(builder.toString());
                modelInfo.addLayer(info);
                x++;
            }

            Map<String,GraphVertex> map = conf.getVertices();
            GraphVertex[] vertices = map.values().toArray(new GraphVertex[map.size()]);

            // filling grid in LTR/TTB direction
            List<String> needle = new ArrayList<>();


            // we assume that max row can't be higher then total number of vertices
            for (int y = 1; y < vertices.length; y++) {
                if (needle.isEmpty()) needle.addAll(inputs);

//                    for each grid row we look for nodes, that are connected to previous layer
                List<LayerInfo> layersForGridY = flattenToY(modelInfo, vertices, needle, y);

                needle.clear();
                for (LayerInfo layerInfo : layersForGridY) {
                    needle.add(layerInfo.getName());
                }
                if (needle.isEmpty()) break;
            }

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
    */

    private static LayerInfo getLayerInfo(Layer layer, int x, int y, int order) {
        LayerInfo info = new LayerInfo();


        // set coordinates
        info.setX(x);
        info.setY(y);

        // if name was set, we should grab it
        try {
            info.setName(layer.getLayerName());
        } catch (Exception e) {
        }
        if (info.getName() == null || info.getName().isEmpty()) info.setName("unnamed");

        // unique layer id required here
        info.setId(order);

        // set layer description according to layer params
        Description description = new Description();
        info.setDescription(description);

        // set layer type
        try {
            info.setLayerType(layer.getClass().getSimpleName().replaceAll("Layer$", ""));
        } catch (Exception e) {
            info.setLayerType("n/a");
            return info;
        }


        StringBuilder mainLine = new StringBuilder();
        StringBuilder subLine = new StringBuilder();
        StringBuilder fullLine = new StringBuilder();

        //    log.info("Layer: " + info.getName() + " class: " + layer.getClass().getSimpleName());


//        if (layer.type().equals(Layer.Type.CONVOLUTIONAL)) {
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

        subLine.append(" A: [").append(layer.getActivationFunction()).append("]");
        fullLine.append("Activation function: ").append("<b>").append(layer.getActivationFunction()).append("</b>").append("<br/>");

        description.setMainLine(mainLine.toString());
        description.setSubLine(subLine.toString());
        description.setText(fullLine.toString());

        return info;
    }

    /*
    protected  List<LayerInfo> flattenToY(ModelInfo model, GraphVertex[] vertices, List<String> currentInput, int currentY) {

        //Given the graph structure: work out position of

        List<LayerInfo> results = new ArrayList<>();
        int x = 0;
        for (int v = 0; v < vertices.length; v++) {
            GraphVertex vertex = vertices[v];
            VertexIndices[] indices = vertex.getInputVertices();

            if (indices != null) for (int i = 0; i < indices.length; i++) {
                GraphVertex cv = vertices[indices[i].getVertexIndex()];
                String inputName = cv.getVertexName();

                for (String input: currentInput) {
                    if (inputName.equals(input)) {
                        // we have match for Vertex
                        //    log.info("Vertex: " + vertex.getVertexName() + " has Input: " + input);
                        try {
                            LayerInfo info = model.getLayerInfoByName(vertex.getVertexName());
                            if (info == null) info = getLayerInfo(vertex.getLayer(), x, currentY, 121);
                            info.setName(vertex.getVertexName());

                            // special case here: vertex isn't a layer
                            if (vertex.getLayer() == null) {
                                info.setLayerType(vertex.getClass().getSimpleName());
                            }
                            if (info.getName().endsWith("-merge")) info.setLayerType("MERGE");
                            if (model.getLayerInfoByName(vertex.getVertexName()) == null) {
                                x++;
                                model.addLayer(info);
                                results.add(info);
                            }

                            // now we should map connections
                            LayerInfo connection = model.getLayerInfoByName(input);
                            if (connection != null) {
                                connection.addConnection(info);
                                //  log.info("Adding connection ["+ connection.getName()+"] -> ["+ info.getName()+"]");
                            } else {
                                // the only reason to have null here, is direct input connection
                                //connection.addConnection(0,0);
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }
        return results;
    }
    */
}
