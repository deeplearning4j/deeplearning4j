package org.deeplearning4j.caffe.translate;

import org.deeplearning4j.caffe.CaffeTestUtil;
import org.deeplearning4j.caffe.dag.CaffeNode;
import org.deeplearning4j.caffe.dag.Graph;
import org.deeplearning4j.caffe.proto.Caffe.NetParameter;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author jeffreytang
 */
public class CaffeLayerGraphConversionTest {

    @Test
    public void convertLogisticNetTest() throws Exception {
        NetParameter net = CaffeTestUtil.getLogisticNet();
        Graph graph = new CaffeLayerGraphConversion(net).convert();
        System.out.println(graph);
        assertTrue(graph != null);
        System.out.println("Start Nodes: " + graph.getStartNodeSet());
        System.out.println("End Nodes: " + graph.getEndNodeSet());
        assertEquals(graph.graphSize(), 3);
    }

    @Test
    public void convertImageNetTest() throws Exception {
        NetParameter net = CaffeTestUtil.getImageNetNet();
        Graph<CaffeNode> graph = new CaffeLayerGraphConversion(net).convert();
        System.out.println(graph);
        assertTrue(graph != null);
        System.out.println("Start Nodes: " + graph.getStartNodeSet());
        System.out.println("End Nodes: " + graph.getEndNodeSet());
    }

}
