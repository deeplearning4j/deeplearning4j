package org.deeplearning4j.caffedag;

import org.deeplearning4j.caffedag.CaffeNode.NodeType;
import org.deeplearning4j.dag.Graph;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author jeffreytang
 */
public class CaffeGraphTest {
    @Test
    public void testCreatingNode() {
        CaffeNode blobNodeA = new CaffeNode(NodeType.BLOB, "BlobNodeA");
        CaffeNode blobNodeB = new CaffeNode(NodeType.BLOB, "BlobNodeB");

        CaffeNode layerNodeA = new CaffeNode(NodeType.LAYER, "LayerNodeA");
        CaffeNode layerNodeB = new CaffeNode(NodeType.LAYER, "LayerNodeB");

        System.out.print(blobNodeA);
        assertTrue(!blobNodeA.equals(blobNodeB));
        assertTrue(!blobNodeA.equals(layerNodeA));
        assertTrue(!blobNodeA.equals(layerNodeB));
        assertTrue(blobNodeA.getName().equals("BlobNodeA"));
        assertTrue(blobNodeA.getType().equals(NodeType.BLOB));
    }

    @Test
    public void testCreatingGraph() {

        CaffeNode nodeA = new CaffeNode(NodeType.BLOB, "A");
        CaffeNode nodeB = new CaffeNode(NodeType.BLOB, "B");
        CaffeNode nodeC = new CaffeNode(NodeType.LAYER, "C");
        CaffeNode nodeD = new CaffeNode(NodeType.LAYER, "D");

        Graph graph = new Graph();
        graph.addEdge(nodeA, nodeB);
        graph.addEdge(nodeC, nodeB);
        graph.addEdge(nodeC, nodeD);

        System.out.print(graph);
    }

}
