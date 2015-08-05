package org.deeplearning4j.caffe.translate;

import com.google.protobuf.GeneratedMessage;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.deeplearning4j.caffe.dag.CaffeNode;
import org.deeplearning4j.caffe.dag.CaffeNode.LayerSubType;
import org.deeplearning4j.caffe.dag.CaffeNode.LayerType;
import org.deeplearning4j.caffe.projo.Caffe.NetParameter;
import org.deeplearning4j.dag.Graph;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author jeffreytang
 */
public class CaffeLayerConversion {

    public CaffeLayerConversion() {}

    private static Map<String, String> subType2MethodMap = new HashMap<String, String>() {{
        put("CONVOLUTION", "getConvolutionParam");
        put("POOLING", "getPoolingParam");
        put("RELU", "getReluParam");
        put("SIGMOID", "getSigmoidParam");
        put("TANH", "getTanhParam");
        put("SOFTMAX", "getSoftmaxParam");
        put("FLATTEN", "getFlattenParam");
        put("RESHAPE", "getReshapeParam");
        put("CONCAT", "getConcatParam");
        put("SLICE", "getSliceParam");
        put("INNERPRODUCT", "getInnerProductParam");
        put("SOFTMAXWITHLOSS", "");
        put("EUCLIDEANLOSS", "");
        put("SIGMOIDCROSSENTROPYLOSS", "");
        put("SPLIT", "");
        put("DATA", "");
        put("MEMORYDATA", "");
        put("HDF5DATA", "");
        put("HDF5OUTPUT", "");
        put("ACCURACY", "");
    }};

    private static Map<LayerSubType, LayerType> subType2TypeMap = new HashMap<LayerSubType, LayerType>(){{
        put(LayerSubType.CONVOLUTION, LayerType.HIDDEN);
        put(LayerSubType.POOLING, LayerType.HIDDEN);
        put(LayerSubType.RELU, LayerType.HIDDEN);
        put(LayerSubType.SIGMOID, LayerType.HIDDEN);
        put(LayerSubType.TANH, LayerType.HIDDEN);
        put(LayerSubType.SOFTMAX, LayerType.HIDDEN);
        put(LayerSubType.SOFTMAXWITHLOSS, LayerType.HIDDEN);
        put(LayerSubType.EUCLIDEANLOSS, LayerType.HIDDEN);
        put(LayerSubType.SIGMOIDCROSSENTROPYLOSS, LayerType.HIDDEN);
        put(LayerSubType.FLATTEN, LayerType.PROCESSING);
        put(LayerSubType.RESHAPE, LayerType.PROCESSING);
        put(LayerSubType.CONCAT, LayerType.PROCESSING);
        put(LayerSubType.SLICE, LayerType.PROCESSING);
        put(LayerSubType.SPLIT, LayerType.PROCESSING);
    }};

    @SuppressWarnings("unchecked")
    private static List<? extends GeneratedMessage> convertNetToLayerList(NetParameter net) {
        int layerCount = net.getLayerCount();
        int layersCount = net.getLayersCount();

        List<? extends GeneratedMessage> finalLayerList;

        if (layerCount > 0 && layersCount > 0) {
            throw new IllegalStateException("Caffe net has both LayerParameter and V1LayerParameter layers.");
        } else if (layerCount == 0 && layersCount == 0) {
            throw new IllegalStateException("Caffe net has neither LayerParameter or V1LayerParameter layers.");
        } else if (layerCount > 0) {
            finalLayerList = net.getLayerList();
        } else if (layersCount > 0) {
            finalLayerList =  net.getLayersList();
        } else {
            throw new IllegalStateException("Illegal state of Caffe net layers.");
        }
        return finalLayerList;
    }

    private static LayerSubType subTypify(String subType) {
        try {
            return LayerSubType.valueOf(subType);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(String.format("The LayerSubType %s is not " +
                    "supported or does not exist.", subType));
        }
    }

    @SuppressWarnings("unchecked")
    private static Method getMethodFromString(String methodString, Class c) throws NoSuchMethodException{
        return c.getDeclaredMethod(methodString);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, List<CaffeNode>> convertLayerToCaffeNodeMap(GeneratedMessage layer)
            throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {

        //// Get methods
        Method getNameMethod = getMethodFromString("getName", layer.getClass());
        Method getTypeMethod = getMethodFromString("getType", layer.getClass());
        Method getTopListMethod = getMethodFromString("getTopList", layer.getClass());
        Method getBottomListMethod = getMethodFromString("getBottomList", layer.getClass());


        //// Map to hold all the node generated from the layer
        Map<String, List<CaffeNode>> caffeNodeMap = new HashMap<>();
        caffeNodeMap.put("current", new ArrayList<CaffeNode>());
        caffeNodeMap.put("top", new ArrayList<CaffeNode>());
        caffeNodeMap.put("bottom", new ArrayList<CaffeNode>());

        //// Current Node
        String layerName = (String) getNameMethod.invoke(layer);
        String layerSubTypeString = (String) getTypeMethod.invoke(layer);
        LayerSubType layerSubType = subTypify(layerSubTypeString);
        LayerType layerType = subType2TypeMap.get(layerSubType);
        String paramMethodString = subType2MethodMap.get(layerSubTypeString);

        // Get the data in the layer
        Map<String, Object> field2valueMap = new HashMap<>();
        if (!paramMethodString.isEmpty()) {
            Method getParamMethod = getMethodFromString(subType2MethodMap.get(layerSubTypeString), layer.getClass());
            Object layerClassParameter = getParamMethod.invoke(layer);
            List<Field> fieldList = FieldUtils.getAllFieldsList(layerClassParameter.getClass());
            for (Field field : fieldList) {
                field.setAccessible(true);
                field2valueMap.put(field.getName(), field.get(layer));
            }
        }
        CaffeNode currentNode = new CaffeNode(layerName, layerType, layerSubType, field2valueMap);
        caffeNodeMap.get("current").add(currentNode);

        //// Top and Bottom
        List<String> bottomList = (List<String>) getBottomListMethod.invoke(layer);
        List<String> topList = (List<String>) getTopListMethod.invoke(layer);

        for (String bottomLayerName : bottomList) {
            CaffeNode bottomNode = new CaffeNode(bottomLayerName, LayerType.CONNECTOR, LayerSubType.CONNECTOR);
            caffeNodeMap.get("bottom").add(bottomNode);
        }
        for (String topLayerName : topList) {
            CaffeNode topNode = new CaffeNode(topLayerName, LayerType.CONNECTOR, LayerSubType.CONNECTOR);
            caffeNodeMap.get("top").add(topNode);
        }

        return caffeNodeMap;
    }

    private static List<Map<String, List<CaffeNode>>> convertNetToNodeMapList(NetParameter net)
            throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {

        List<? extends GeneratedMessage> layerList = convertNetToLayerList(net);

        List<Map<String, List<CaffeNode>>> nodeMapList = new ArrayList<>();
        for (GeneratedMessage layer : layerList) {
            Map<String, List<CaffeNode>> nodeMap = convertLayerToCaffeNodeMap(layer);
            nodeMapList.add(nodeMap);
        }

        return nodeMapList;
    }

    private static void addNodeMapToGraph(Map<String, List<CaffeNode>> nodeMap, Graph graph) {
        CaffeNode currNode = nodeMap.get("current").get(0);
        List<CaffeNode> topNodeList = nodeMap.get("top");
        List<CaffeNode> bottomNodeList = nodeMap.get("bottom");

        for (CaffeNode topNode : topNodeList) {
            graph.addEdge(currNode, topNode);
        }

        for (CaffeNode bottomNode : bottomNodeList) {
            graph.addEdge(bottomNode, currNode);
        }
    }

    private static Graph convertNodeMapListToGraph(List<Map<String, List<CaffeNode>>> nodeMapList) {
        Graph graph = new Graph();
        for (Map<String, List<CaffeNode>> nodeMap : nodeMapList) {
            addNodeMapToGraph(nodeMap, graph);
        }
        return graph;
    }

    public static Graph convertNetToNodeGraph(NetParameter net)
            throws NoSuchMethodException, IllegalAccessException, InvocationTargetException{
        List<Map<String, List<CaffeNode>>> nodeMapList = convertNetToNodeMapList(net);
        return convertNodeMapListToGraph(nodeMapList);
    }
}
