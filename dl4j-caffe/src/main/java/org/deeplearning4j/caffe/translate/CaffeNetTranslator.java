package org.deeplearning4j.caffe.translate;

import lombok.Data;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.deeplearning4j.caffe.common.NNConfigBuilderContainer;
import org.deeplearning4j.caffe.dag.CaffeNode;
import org.deeplearning4j.caffe.dag.CaffeNode.LayerSubType;
import org.deeplearning4j.caffe.dag.Graph;
import org.deeplearning4j.caffe.proto.Caffe.NetParameter;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

/**
 * @author jeffreytang
 */
@SuppressWarnings("unchecked")
@Data
public class CaffeNetTranslator {
    private Map<LayerSubType, Object> layerMapping;
    private Map<LayerSubType, Map<String, String>> layerParamMapping;

    private void populateLayerMapping() {

        // All the dense layers
        layerMapping = new HashMap<LayerSubType, Object>() {{
            put(LayerSubType.CONVOLUTION, new ConvolutionLayer.Builder().build());
            put(LayerSubType.POOLING, new SubsamplingLayer.Builder().build());
            put(LayerSubType.RELU, new DenseLayer.Builder().activation("relu").build());
            put(LayerSubType.SIGMOID, new DenseLayer.Builder().activation("sigmoid").build());
            put(LayerSubType.TANH, new DenseLayer.Builder().activation("tanh").build());
            put(LayerSubType.SOFTMAX, new DenseLayer.Builder().activation("softmax").build());
            put(LayerSubType.SOFTMAXWITHLOSS, new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                    .activation("softmax").build());
            put(LayerSubType.SOFTMAX_LOSS, new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                    .activation("softmax").build());
            put(LayerSubType.EUCLIDEANLOSS, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).build());
            put(LayerSubType.SIGMOIDCROSSENTROPYLOSS, new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build()); //TODO: Fix loss functions
            put(LayerSubType.FLATTEN, null);
            put(LayerSubType.RESHAPE, null);
            put(LayerSubType.CONCAT, null);
            put(LayerSubType.SLICE, null);
            put(LayerSubType.SPLIT, null);
        }};
    }

    private void populateLayerParamMapping() {
        final Map<String, String> convolutionMapping = new HashMap<String, String>() {{
            put("numOutput_", "nOut");
            put("kernelSize_", "kernelSize");
            put("pad_", "padding");
            put("stride_", "stride");
//            put("weightFiller_", "weightInit");
            put("biasFiller_", "");
            put("group_", "");
        }};

        final Map<String, String> poolMapping = new HashMap<String, String>() {{
            put("pool_", "poolingType");
            put("kernelSize_", "kernelSize");
            put("stride_", "stride");
        }};

        final Map<String, String> innerProductMapping = new HashMap<String, String>() {{
            put("numOutput_", "nOut");
//            put("weightFiller_", "weightInit");
            put("biasFiller_", "");
            put("biasTerm_", "");
        }};

        final Map<String, String> dropoutMapping = new HashMap<String, String>() {{
            put("dropoutRatio_", "dropOut");
        }};

        layerParamMapping = new HashMap<LayerSubType, Map<String, String>>() {{
            put(LayerSubType.CONVOLUTION, convolutionMapping);
            put(LayerSubType.POOLING, poolMapping);
            put(LayerSubType.INNERPRODUCT, innerProductMapping);
            put(LayerSubType.DROPOUT, dropoutMapping);
        }};
    }


    /**
     *
     */
    public CaffeNetTranslator() {
        populateLayerMapping();
        populateLayerParamMapping();
    }

    /**
     *
     * @param fieldName
     * @param fieldValue
     * @return
     */
    public Object convertCaffeValue(String fieldName, Object fieldValue) {
        if (fieldName.equals("kernelSize_")) {
            return new int[]{(Integer) fieldValue, (Integer) fieldValue};
        } else if (fieldName.equals("pad_")) {
            return new int[]{(Integer) fieldValue, (Integer) fieldValue};
        } else if (fieldName.equals("stride_")) {
            return new int[]{(Integer) fieldValue, (Integer) fieldValue};
        } else if (fieldName.equals("pool_")) {
            if (fieldValue.equals("STOCHASTIC")) {
                fieldValue = SubsamplingLayer.PoolingType.MAX;
            } else if (fieldValue.equals("MAX")) {
                fieldValue = SubsamplingLayer.PoolingType.MAX;
            } else {
                fieldValue = SubsamplingLayer.PoolingType.AVG;
            }
            return fieldValue;
        } else {
            return fieldValue;
        }
    }

    public List<org.deeplearning4j.nn.conf.layers.Layer> makeDL4JLayerList(NetParameter net) throws Exception {

        // Convert caffe layers to a graph
        Graph<CaffeNode> caffeGraph = new CaffeLayerGraphConversion(net).convert();
        final CaffeNode firstNode = (CaffeNode) caffeGraph.getStartNodeSet().toArray()[0];
        // Do a breath first search to traverse and convert each node in the graph
        List<CaffeNode> seen = new ArrayList<>();
        Stack<CaffeNode> queue = new Stack<CaffeNode>() {{ add(firstNode); }};
        List<org.deeplearning4j.nn.conf.layers.Layer> dl4jLayerList = new ArrayList<>();
        Map<String,List<CaffeNode>> nodes = new HashMap<>();
        for(CaffeNode node : caffeGraph.getAllNodes()) {
            List<CaffeNode> nodeListForName = nodes.get(node.getName());
            if(nodeListForName == null) {
                nodeListForName = new ArrayList<>();
                nodes.put(node.getName(),nodeListForName);
            }
            nodeListForName.add(node);
        }


        // Execute BFS
        while (!queue.empty()) {
            CaffeNode currentNode = queue.pop();
            // Translate logic
            if (!currentNode.getLayerType().equals(CaffeNode.LayerType.CONNECTOR)) {
                LayerSubType layerSubType = currentNode.getLayerSubType();
                Map<String, Object> caffeFieldMap = currentNode.getMetaData();
                List<CaffeNode> bottomNodes = currentNode.getBottomNodeList();
                //TODO: Allow customized WeightInit

                // If layer have associated parameters
                if (layerMapping.containsKey(layerSubType) && layerMapping.get(layerSubType) != null) {
                    // Get DL4J layer from caffe layer
                    org.deeplearning4j.nn.conf.layers.Layer dl4jLayer =
                            (org.deeplearning4j.nn.conf.layers.Layer) layerMapping.get(layerSubType);

                    // If Caffe layer has parameters (inner product, convolution, etc)
                    if (layerParamMapping.containsKey(layerSubType)) {
                        // Get the param mapping for the particular layer
                        Map<String, String> layerParamMap = layerParamMapping.get(layerSubType);
                        // Loop through caffe layer fields
                        for (Map.Entry<String, Object> entry : caffeFieldMap.entrySet()) {
                            // The name of the caffe field
                            String caffeFieldName = entry.getKey();
                            // Get field value of caffe field
                            Object caffeValue = convertCaffeValue(caffeFieldName, entry.getValue());
                            // Put the caffe value into a map of dl4j field and value
                            if (layerParamMap.containsKey(caffeFieldName)) {
                                String dl4jFieldName = layerParamMap.get(caffeFieldName);
                                if (!dl4jFieldName.isEmpty()) {
                                    // Write the corresponding value into the DL4J layer object
                                    FieldUtils.writeField(dl4jLayer, dl4jFieldName, caffeValue, true);
                                }
                            }
                            else {
                                System.out.println("Unused field " + caffeFieldName);
                            }
                        }
                    } // End of if (writing DL4J layer parameters

                    if(dl4jLayer instanceof ConvolutionLayer) {
                        //set the in and out on the layer to be the size of the data
                        ConvolutionLayer layer = (ConvolutionLayer) dl4jLayer;
                        layer.setNIn(currentNode.getData().get(0).size(1));
                        layer.setNOut(currentNode.getData().get(0).size(0));
                    }


                    // Put DL4J layer into a list
                    dl4jLayerList.add(dl4jLayer);
                }
            }

            if (!seen.contains(currentNode)) {
                seen.add(currentNode);
                queue.addAll(caffeGraph.getNextNodes(currentNode));
            }
        }

        return dl4jLayerList;
    }

    /**
     *
     * @param net
     * @param builderContainer
     * @throws Exception
     */
    public void translate(NetParameter net, NNConfigBuilderContainer builderContainer)
            throws Exception {

        List<org.deeplearning4j.nn.conf.layers.Layer> layerList = makeDL4JLayerList(net);
        NeuralNetConfiguration.Builder builder = builderContainer.getBuilder();
        NeuralNetConfiguration.ListBuilder layerListBuilder = builder.list(layerList.size());

        int layerCount = 0;
        for (org.deeplearning4j.nn.conf.layers.Layer layer : layerList) {
            layerListBuilder = layerListBuilder.layer(layerCount, layer);
            layerCount += 1;
        }

        builderContainer.setListBuilder(layerListBuilder);
    }
}
