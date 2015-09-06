package org.deeplearning4j.caffe.translate;

import org.deeplearning4j.caffe.CaffeTestUtil;
import org.deeplearning4j.caffe.proto.Caffe.NetParameter;
import org.deeplearning4j.caffe.proto.Caffe.SolverParameter;
import org.deeplearning4j.caffe.common.CaffeLoader;
import org.deeplearning4j.caffe.common.SolverNetContainer;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author jeffreytang
 */
public class CaffeLoaderTest extends CaffeTestUtil {

    @Test
    public void testCaffeLoaderOne() throws IOException{
        SolverNetContainer container = new CaffeLoader()
                .binaryNet(getLogisticBinaryNetPath())
                .textFormatSolver(getLogisticTextFormatSolverPath())
                .load();
        assertTrue(container.getNet() != null);
        assertTrue(container.getSolver() != null);

        NetParameter net = container.getNet();
        assertEquals(net.getName(), "LogisticRegressionNet");
        assertEquals(net.getLayerCount(), 3);

        SolverParameter solver = container.getSolver();
        assertEquals(solver.getBaseLr(), 0.01, 1e-3);
        assertEquals(solver.getMaxIter(), 1000);
        assertEquals(solver.getLrPolicy(), "inv");
    }

    @Test
    public void testCaffeLoaderTwo() throws IOException{
        SolverNetContainer container = new CaffeLoader()
                .textFormatNet(getLogisticTextFormatNetPath())
                .textFormatSolver(getLogisticTextFormatSolverPath())
                .load();
        assertTrue(container.getNet() != null);
        assertTrue(container.getSolver() != null);

        NetParameter net = container.getNet();
        assertEquals(net.getName(), "LogisticRegressionNet");
        assertEquals(net.getLayerCount(), 5);

        SolverParameter solver = container.getSolver();
        assertEquals(solver.getBaseLr(), 0.01, 1e-3);
        assertEquals(solver.getMaxIter(), 1000);
        assertEquals(solver.getLrPolicy(), "inv");
    }

    @Test
    public void testCaffeLoaderThree() throws IOException{
        SolverNetContainer container = new CaffeLoader()
                .binaryNet(getLogisticBinaryNetPath())
                .textFormatNet(getLogisticTextFormatNetPath())
                .textFormatSolver(getLogisticTextFormatSolverPath())
                .load();
        assertTrue(container.getNet() != null);
        assertTrue(container.getSolver() != null);

        NetParameter net = container.getNet();
        assertEquals(net.getName(), "LogisticRegressionNet");
        assertEquals(net.getLayerCount(), 3);

        SolverParameter solver = container.getSolver();
        assertEquals(solver.getBaseLr(), 0.01, 1e-3);
        assertEquals(solver.getMaxIter(), 1000);
        assertEquals(solver.getLrPolicy(), "inv");
    }

    @Test(expected = IllegalStateException.class)
    public void testCaffeLoaderExceptionOne() throws IOException {
        new CaffeLoader().binaryNet(getLogisticBinaryNetPath()).load();
    }

    @Test(expected = IllegalStateException.class)
    public void testCaffeLoaderExceptionTwo() throws IOException {
        new CaffeLoader().textFormatNet(getImageNetTextFormatNetPath()).load();
    }

    @Test(expected = IllegalStateException.class)
    public void testCaffeLoaderExceptionThree() throws IOException {
        new CaffeLoader()
                .binaryNet(getLogisticBinaryNetPath())
                .textFormatNet(getImageNetTextFormatNetPath())
                .load();
    }

    @Test(expected = IllegalStateException.class)
    public void testCaffeLoaderExceptionFour() throws IOException {
        new CaffeLoader().textFormatSolver(getLogisticTextFormatSolverPath()).load();
    }
}
