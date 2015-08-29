package org.deeplearning4j.caffe.dag;
import org.deeplearning4j.caffe.dag.CaffeNode.LayerSubType;
import org.deeplearning4j.caffe.dag.CaffeNode.LayerType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author jeffreytang
 */
public class CaffeGraphTest {
    @Test
    public void testCreatingNode() {
        CaffeNode connectorNode = new CaffeNode("connector", LayerType.CONNECTOR, LayerSubType.CONNECTOR);

        System.out.println(connectorNode);
        assertTrue(connectorNode.getName().equals("connector"));
        assertTrue(connectorNode.getLayerType().equals(CaffeNode.LayerType.CONNECTOR));
        assertTrue(connectorNode.getLayerSubType().equals(LayerSubType.CONNECTOR));
    }

    @Test
    public void testCreatingGraph() {

        CaffeNode connectorNode = new CaffeNode("connector", LayerType.CONNECTOR, LayerSubType.CONNECTOR);
        CaffeNode convolutionNode = new CaffeNode("convolution", LayerType.HIDDEN, LayerSubType.CONVOLUTION);
        CaffeNode lossNode = new CaffeNode("loss", LayerType.CONNECTOR, LayerSubType.CONNECTOR);

        Graph graph = new Graph();
        graph.addEdge(convolutionNode, connectorNode);
        graph.addEdge(connectorNode, lossNode);

        System.out.print(graph);
        assertEquals(graph.graphSize(), 3);
        String nextNodeName = ((CaffeNode) graph.getNextNodes(connectorNode).iterator().next()).getName();
        assertEquals(nextNodeName, "loss");
    }

}
