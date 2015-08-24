package org.deeplearning4j.caffe.translate;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.protobuf.GeneratedMessage;
import lombok.Data;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.deeplearning4j.caffe.dag.CaffeNode;
import org.deeplearning4j.caffe.dag.CaffeNode.LayerSubType;
import org.deeplearning4j.caffe.dag.CaffeNode.LayerType;
import org.deeplearning4j.caffe.projo.Caffe.BlobProto;
import org.deeplearning4j.caffe.projo.Caffe.NetParameter;
import org.deeplearning4j.caffe.dag.Graph;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;

/**
 * @author jeffreytang
 */
@Data
@SuppressWarnings("unchecked")
public class CaffeLayerGraphConversion {

    public CaffeLayerGraphConversion() {}

    protected static Logger log = LoggerFactory.getLogger(CaffeLayerGraphConversion.class);

    NetParameter net;
    Map<LayerSubType, Map<String, String>> layerParamMapping;
    Nd4j nd4j = new Nd4j();


    public CaffeLayerGraphConversion(NetParameter net) {
        this.net = net;
        this.layerParamMapping = initLayerParamMap();
    }

    // Get map of layer param mapping
    public Map<LayerSubType, Map<String, String>> initLayerParamMap() {
        CaffeNetTranslator netMap = new CaffeNetTranslator();
        return netMap.getLayerParamMapping();
    }

    // Map of LayerSubType to method getting the layer params
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
        put("DATA", null);
        put("MEMORYDATA", null);
        put("HDF5DATA", null);
        put("HDF5OUTPUT", null);
        put("ACCURACY", null);
    }};

    // Map of the layerSubType to layerType
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

    private List<? extends GeneratedMessage> convertNetToLayerList() {
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

    private LayerSubType subTypify(String subType) {
        return LayerSubType.valueOf(subType);
    }

    private boolean isSupported(String subType) {
        return subType2MethodMap.containsKey(subType);
    }

    private boolean isSubTypeAvaliable(String subType) {

        if (!isSupported(subType)) {
            throw new IllegalArgumentException(
                    String.format("The LayerSubType %s is not " +
                    "supported or does not exist.", subType));
        } else {
            try {
                LayerSubType.valueOf(subType);
                return true;
            } catch (IllegalArgumentException e) {
                return false;
            }
        }
    }

    private Method getMethodFromString(String methodString, Class c) throws NoSuchMethodException{
        return c.getDeclaredMethod(methodString);
    }

    private List<INDArray> convertBlobToDataMap(List<BlobProto> blobProtList) {

        List<INDArray> blobDataList = new ArrayList<>();
        for (BlobProto blobProto : blobProtList) {
            // Shape of the blob
            int[] dims;
            List<Long> shape = blobProto.getShape().getDimList();
            int height = blobProto.getHeight();
            int width = blobProto.getWidth();
            int num = blobProto.getNum();
            int channel = blobProto.getChannels();
            if ((height > 0 || width > 0 || num > 0 || channel > 0) && shape.size() == 0) {
                dims = Ints.toArray(Arrays.asList(num, channel, width, height));
            } else {
                dims = Ints.toArray(shape);
            }
            // Blob data
            DataBuffer blobData = Nd4j.createBuffer(Doubles.toArray(blobProto.getDataList()));
            // Create NDArray of blob data based on shape
            blobDataList.add(Nd4j.create(blobData, dims));
        }
        return blobDataList;
    }

    private Map<String, Set<CaffeNode>> convertLayerToCaffeNodeMap(GeneratedMessage layer)
            throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {

        //// Get methods
        Method getNameMethod = getMethodFromString("getName", layer.getClass());
        Method getTypeMethod = getMethodFromString("getType", layer.getClass());
        Method getTopListMethod = getMethodFromString("getTopList", layer.getClass());
        Method getBottomListMethod = getMethodFromString("getBottomList", layer.getClass());
        Method getBlobListMethod = getMethodFromString("getBlobsList", layer.getClass());


        //// Map to hold all the node generated from the layer
        Map<String, Set<CaffeNode>> caffeNodeMap = new HashMap<>();
        caffeNodeMap.put("current", new HashSet<CaffeNode>());
        caffeNodeMap.put("top", new HashSet<CaffeNode>());
        caffeNodeMap.put("bottom", new HashSet<CaffeNode>());

        //// Current Node
        String layerName = (String) getNameMethod.invoke(layer);
        String layerSubTypeString = ((String) getTypeMethod.invoke(layer)).toUpperCase();

        // Is the current layer supported. If not throw IllegalState
        // Is the current layer subtype supported. If not, ignore layer and return null
        if (!isSubTypeAvaliable(layerSubTypeString)) { return null; }
        //

        LayerSubType layerSubType = subTypify(layerSubTypeString);
        LayerType layerType = subType2TypeMap.get(layerSubType);
        String paramMethodString = subType2MethodMap.get(layerSubTypeString);

        // Get the meta data in the layer
        Map<String, Object> field2valueMap = new HashMap<>();
        if (!paramMethodString.isEmpty()) {
            Method getParamMethod = getMethodFromString(subType2MethodMap.get(layerSubTypeString), layer.getClass());
            Object layerClassParameter = getParamMethod.invoke(layer);
            Set<String> layerStringFields = layerParamMapping.get(layerSubType).keySet();
            for (String stringField : layerStringFields) {
                try {
                    Object val = FieldUtils.readField(layerClassParameter, stringField, true);
                    field2valueMap.put(stringField, val);
                } catch (IllegalArgumentException e) {
                    log.info(String.format("Cannot parse field '%s'", stringField));
                }
            }
        }
        // Get the blob data in the layer
        List<INDArray> data = convertBlobToDataMap((List<BlobProto>) getBlobListMethod.invoke(layer));
        ////

        //// Assign Bottom Nodes
        List<String> bottomList = (List<String>) getBottomListMethod.invoke(layer);
        List<String> topList = (List<String>) getTopListMethod.invoke(layer);

        for (String bottomLayerName : bottomList) {
            CaffeNode bottomNode = new CaffeNode(bottomLayerName, LayerType.CONNECTOR, LayerSubType.CONNECTOR);
            caffeNodeMap.get("bottom").add(bottomNode);
        }
        ////

        //// Assign Current node
        CaffeNode currentNode = new CaffeNode(layerName, layerType, layerSubType, field2valueMap, data,
                caffeNodeMap.get("bottom"));
        caffeNodeMap.get("current").add(currentNode);
        ////

        //// Assign Top nodes
        for (String topLayerName : topList) {
            Set<CaffeNode> bottomNodeSetTop = new HashSet<>();
            bottomNodeSetTop.add(currentNode);
            CaffeNode topNode = new CaffeNode(topLayerName, LayerType.CONNECTOR, LayerSubType.CONNECTOR, bottomNodeSetTop);
            caffeNodeMap.get("top").add(topNode);
        }

        return caffeNodeMap;
    }

    private List<Map<String, Set<CaffeNode>>> convertNetToNodeMapList()
            throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {

        List<? extends GeneratedMessage> layerList = convertNetToLayerList();

        List<Map<String, Set<CaffeNode>>> nodeMapList = new ArrayList<>();
        for (GeneratedMessage layer : layerList) {
            Map<String, Set<CaffeNode>> nodeMap = convertLayerToCaffeNodeMap(layer);
            if (nodeMap != null) {
                nodeMapList.add(nodeMap);
            }
        }

        return nodeMapList;
    }

    private void addNodeMapToGraph(Map<String, Set<CaffeNode>> nodeMap, Graph graph) {
        CaffeNode currNode = nodeMap.get("current").iterator().next();
        Set<CaffeNode> topNodeSet = nodeMap.get("top");
        Set<CaffeNode> bottomNodeSet = nodeMap.get("bottom");

        for (CaffeNode topNode : topNodeSet) {
            graph.addEdge(currNode, topNode);
        }

        for (CaffeNode bottomNode : bottomNodeSet) {
            graph.addEdge(bottomNode, currNode);
        }
    }

    private Graph convertNodeMapSetToGraph(List<Map<String, Set<CaffeNode>>> nodeMapList) {
        Graph graph = new Graph();
        for (Map<String, Set<CaffeNode>> nodeMap : nodeMapList) {
            addNodeMapToGraph(nodeMap, graph);
        }
        addStartEndNodesToGraph(graph);
        return graph;
    }

    private void addStartEndNodesToGraph(Graph graph) {
        Map<CaffeNode, Set<CaffeNode>> adjacencyMap = graph.getAdjacencyListMap();
        for (CaffeNode node : adjacencyMap.keySet()) {
            if (node.getBottomNodeSet() == null || node.getBottomNodeSet().size() == 0) {
                graph.addStartNode(node);
            }
            if (adjacencyMap.get(node).size() == 0){
                graph.addEndNode(node);
            }
        }
        trimStartEndNodes(graph);
    }

    private Set<CaffeNode> forwardTrack(CaffeNode node, Graph graph) {
        return graph.getNeighbors(node);
    }

    private Set<CaffeNode> backwardTrack(CaffeNode node) {
        return node.getBottomNodeSet();
    }


    private void trimStartEndNodes(Graph graph) {
        Set<CaffeNode> startNodeSet = graph.getStartNodeSet();
        Set<CaffeNode> endNodeSet = graph.getEndNodeSet();

        for (CaffeNode startNode : startNodeSet) {
            if (startNode.getLayerSubType().equals(LayerSubType.CONNECTOR)) {
                graph.removeStartNode(startNode);
                Set<CaffeNode> topSet = forwardTrack(startNode, graph);
                for (CaffeNode topNode : topSet) {
                    graph.addStartNode(topNode);
                }
            }
        }

        for (CaffeNode endNode : endNodeSet) {
            if (endNode.getLayerSubType().equals(LayerSubType.CONNECTOR)) {
                graph.removeEndNode(endNode);
                Set<CaffeNode> bottomSet = backwardTrack(endNode);
                for (CaffeNode bottomNode : bottomSet) {
                    graph.addEndNode(bottomNode);
                    graph.removeStartNode(bottomNode);
                }
            }
        }
    }

    public void trimGraph(Graph graph) {
        Map<CaffeNode, Set<CaffeNode>> adjacencyMap = graph.getAdjacencyListMap();
        for (CaffeNode node : adjacencyMap.keySet()) {
            if (node.getBottomNodeSet() == null || node.getBottomNodeSet().size() == 0) {
                graph.addStartNode(node);
            }
            if (adjacencyMap.get(node).size() == 0){
                graph.addEndNode(node);
            }
        }


    }

    // Converts Net to Caffe Node Graph
    public Graph convert()
            throws NoSuchMethodException, IllegalAccessException, InvocationTargetException{
        List<Map<String, Set<CaffeNode>>> nodeMapList = convertNetToNodeMapList();
        Graph graph = convertNodeMapSetToGraph(nodeMapList);
        addStartEndNodesToGraph(graph);
        trimGraph(graph);
        return graph;
    }
}
