package org.deeplearning4j.caffe.dag;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.caffe.projo.Caffe.LayerParameter;
import org.deeplearning4j.dag.Graph;
import org.deeplearning4j.caffe.dag.CaffeNode.NodeType;

import java.util.List;

/**
 * @author jeffreytang
 */
@NoArgsConstructor
@Data
public class CaffeGraphParser {

    public Graph graph = new Graph();

    public void parseLayer(LayerParameter layer) {

        // Add the current Layer as a layer node
        CaffeNode currentNode = new CaffeNode(NodeType.LAYER, layer.getName() + "Layer");
        graph.addNode(currentNode);

        // Get the top and bottom nodes as lists
        List<String> topLayerList = layer.getTopList();
        List<String> bottomLayerList = layer.getBottomList();

        if (bottomLayerList.size() == 0) {
            graph.addRootNode(currentNode);
        }

        // Parse top and bottom nodes to graph
        for (String topLayer : topLayerList) {
            CaffeNode topNode = new CaffeNode(NodeType.BLOB, topLayer + "Blob");
            graph.addNode(topNode);
            graph.addEdge(currentNode, topNode);
        }

        for (String bottomLayer : bottomLayerList) {
            CaffeNode bottomNode = new CaffeNode(NodeType.BLOB, bottomLayer + "Blob");
            graph.addNode(bottomNode);
            graph.addEdge(bottomNode, currentNode);
        }
    }

    public void parseLayerList(List<LayerParameter> layerList) {
        for (LayerParameter layer : layerList) {
            parseLayer(layer);
        }
    }
}
