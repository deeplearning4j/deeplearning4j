package org.deeplearning4j.caffe.dag;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.dag.Graph;

/**
 * @author jeffreytang
 */
@NoArgsConstructor
@Data
public class CaffeGraphParser {

    public Graph graph = new Graph();

//    public CaffeNode parseLayerDataToNode(CaffeNode node, LayerParameter layer) {
//
//    }

//    public void parseSingleLayer(LayerParameter layer) {
//
//        // Add the current Layer as a layer node
//        CaffeNode currentNode = new CaffeNode(LayerType.LAYER, layer.getName() + "Layer");
//        graph.addNode(currentNode);
//
//        // Get the top and bottom nodes as lists
//        List<String> topLayerList = layer.getTopList();
//        List<String> bottomLayerList = layer.getBottomList();
//
//        if (bottomLayerList.size() == 0) {
//            graph.addRootNode(currentNode);
//        }
//
//        // Parse top and bottom nodes to graph
//        for (String topLayer : topLayerList) {
//            CaffeNode topNode = new CaffeNode(LayerType.BLOB, topLayer + "Blob");
//            graph.addNode(topNode);
//            graph.addEdge(currentNode, topNode);
//        }
//
//        for (String bottomLayer : bottomLayerList) {
//            CaffeNode bottomNode = new CaffeNode(LayerType.BLOB, bottomLayer + "Blob");
//            graph.addNode(bottomNode);
//            graph.addEdge(bottomNode, currentNode);
//        }
//    }
//
//    public void parseMultipleLayers(List<LayerParameter> layerList) {
//        for (LayerParameter layer : layerList) {
//            parseSingleLayer(layer);
//        }
//    }
//
//    public void parseNetParameter(NetParameter net) {
//        List<LayerParameter> layerList;
//        if (net.getLayerCount() > 0) {
//            layerList = net.getLayerList();
//        }
//    }
}
