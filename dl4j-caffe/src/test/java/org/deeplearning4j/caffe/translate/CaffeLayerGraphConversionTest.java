package org.deeplearning4j.caffe.translate;

import org.deeplearning4j.caffe.CaffeTestUtil;
import org.deeplearning4j.caffe.projo.Caffe.NetParameter;
import org.deeplearning4j.caffe.dag.Graph;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author jeffreytang
 */
public class CaffeLayerGraphConversionTest {

    @Test
    public void convertTest() throws Exception {
        NetParameter net = CaffeTestUtil.getNet();
        Graph graph = new CaffeLayerGraphConversion(net).convert();
        System.out.println(graph);
        assertTrue(graph != null);

    }
}
