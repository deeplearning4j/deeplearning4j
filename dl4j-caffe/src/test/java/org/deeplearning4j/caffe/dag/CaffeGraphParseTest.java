package org.deeplearning4j.caffe.dag;

import org.deeplearning4j.caffe.projo.Caffe.LayerParameter;
import org.deeplearning4j.caffe.translate.CaffeTranslateTestUtil;

import java.io.IOException;
import java.util.List;

/**
 * @author jeffreytang
 */
public class CaffeGraphParseTest {

    public List<LayerParameter> getLogisticLayer() throws IOException{
        return CaffeTranslateTestUtil.getSolverNet().getNet().getLayerList();
    }

    public void parseSingleLayerTest() throws IOException {

        CaffeGraphParser graphParser = new CaffeGraphParser();
        LayerParameter layer = getLogisticLayer().get(1);
//        graphParser.parseSingleLayer(layer);
//        Graph graph = graphParser.getGraph();
//        graph.getNode

    }

}
