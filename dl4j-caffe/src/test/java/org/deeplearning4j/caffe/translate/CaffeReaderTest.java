package org.deeplearning4j.caffe.translate;

import org.deeplearning4j.caffe.CaffeTestUtil;
import org.deeplearning4j.caffe.proto.Caffe;
import org.deeplearning4j.caffe.proto.Caffe.LayerParameter;
import org.deeplearning4j.caffe.proto.Caffe.NetParameter;
import org.deeplearning4j.caffe.proto.Caffe.SolverParameter;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author jeffreytang
 */
public class CaffeReaderTest extends CaffeTestUtil {
    /**
     * Test reading the Logistic model in binary format into a Java Class
     * This includes the network configuration and the pre-trained weights
     * @throws Exception
     */
    @Test
    public void testBinaryCaffeModelToJavaClassLogistic() throws Exception {

        // Read the Binary File to a Java Class
        NetParameter net = reader.readBinaryNet(getLogisticBinaryNetPath(), 1000);

        // Test the binary file is read in correctly
        assertEquals(net.getName(), "LogisticRegressionNet");
        // Overall layer count
        assertEquals(net.getLayerCount(), 3);
        // First layer assertions
        LayerParameter firstLayer = net.getLayer(0);
        assertTrue(firstLayer != null);
        assertEquals(firstLayer.getName(), "data");
        assertEquals(firstLayer.getType(), "HDF5Data");
        List firstLayerTopList = firstLayer.getTopList();
        assertEquals(firstLayerTopList.size(), 2);
        assertEquals(firstLayerTopList.get(0), "data");
        assertEquals(firstLayerTopList.get(1), "label");
        assertEquals(firstLayer.getBottomCount(), 0);
        // Second layer assertions
        LayerParameter secondLayer = net.getLayer(1);
        assertTrue(secondLayer != null);
        assertEquals(secondLayer.getName(), "fc1");
        assertEquals(secondLayer.getType(), "InnerProduct");
        List secondLayerTopList = secondLayer.getTopList();
        assertEquals(secondLayerTopList.size(), 1);
        assertEquals(secondLayerTopList.get(0), "fc1");
        List secondLayerBottomList = secondLayer.getBottomList();
        assertEquals(secondLayerBottomList.size(), 1);
        assertEquals(secondLayerBottomList.get(0), "data");
        // Second Layer data blob assertions
        Caffe.BlobProto secondLayerWeightBlob = secondLayer.getBlobs(0);
        Caffe.BlobProto secondLayerBiasBlob = secondLayer.getBlobs(1);
        assertEquals(secondLayerWeightBlob.getShape().getDimList(), Arrays.asList(3L, 4L));
        assertEquals(secondLayerWeightBlob.getData(0), 0.789, 1e-1);
        assertEquals(secondLayerWeightBlob.getData(11), 2.35, 1e-1);
        assertEquals(secondLayerBiasBlob.getShape().getDimList(), Arrays.asList(3L));
        assertEquals(secondLayerBiasBlob.getData(0), 0.862, 1e-1);
        assertEquals(secondLayerBiasBlob.getData(2), -2.104, 1e-1);
        // Third Layer assertions
        LayerParameter thirdLayer = net.getLayer(2);
        assertTrue(thirdLayer != null);
        assertEquals(thirdLayer.getName(), "loss");
        assertEquals(thirdLayer.getType(), "SoftmaxWithLoss");
        List thirdLayerTopList = thirdLayer.getTopList();
        List thirdLayerBottomList = thirdLayer.getBottomList();
        assertEquals(thirdLayerTopList.get(0), "loss");
        assertEquals(thirdLayerBottomList.get(0), "fc1");
        assertEquals(thirdLayerBottomList.get(1), "label");
    }

    /**
     * Test reading the Logistic Network parameters in a proto-text (non binary format)
     * @throws IOException
     */
    @Test
    public void testTextFormatNetProtoToJavaClassLogistic() throws IOException {

        // Read the Net proto-text File to a Java Class
        NetParameter net = reader.readTextFormatNet(getLogisticTextFormatNetPath());

        // Test the binary file is read in correctly
        assertEquals(net.getName(), "LogisticRegressionNet");
        // Overall layer count
        // Two more layers, since the test data and accuracy layer are included
        assertEquals(net.getLayerCount(), 5);
        // First layer assertions
        LayerParameter firstLayer = net.getLayer(0);
        assertTrue(firstLayer != null);
        assertEquals(firstLayer.getName(), "data");
        assertEquals(firstLayer.getType(), "HDF5Data");
        List firstLayerTopList = firstLayer.getTopList();
        assertEquals(firstLayerTopList.size(), 2);
        assertEquals(firstLayerTopList.get(0), "data");
        assertEquals(firstLayerTopList.get(1), "label");
        assertEquals(firstLayer.getBottomCount(), 0);
        // Second layer assertions
        LayerParameter secondLayer = net.getLayer(2);
        assertTrue(secondLayer != null);
        assertEquals(secondLayer.getName(), "fc1");
        assertEquals(secondLayer.getType(), "InnerProduct");
        List secondLayerTopList = secondLayer.getTopList();
        assertEquals(secondLayerTopList.size(), 1);
        assertEquals(secondLayerTopList.get(0), "fc1");
        List secondLayerBottomList = secondLayer.getBottomList();
        assertEquals(secondLayerBottomList.size(), 1);
        assertEquals(secondLayerBottomList.get(0), "data");
        // Third Layer assertions
        LayerParameter thirdLayer = net.getLayer(3);
        assertTrue(thirdLayer != null);
        assertEquals(thirdLayer.getName(), "loss");
        assertEquals(thirdLayer.getType(), "SoftmaxWithLoss");
        List thirdLayerTopList = thirdLayer.getTopList();
        List thirdLayerBottomList = thirdLayer.getBottomList();
        assertEquals(thirdLayerTopList.get(0), "loss");
        assertEquals(thirdLayerBottomList.get(0), "fc1");
        assertEquals(thirdLayerBottomList.get(1), "label");
    }

    /**
     * Test reading the ImageNet Solver parameters in a proto-text (non binary format) file
     * @throws IOException
     */
    @Test
    public void testTextFormatSolverProtoToJavaClassLogistic() throws IOException {

        // Read the Solver proto-text File to a Java Class
        SolverParameter solver = reader.readTextFormatSolver(getLogisticTextFormatSolverPath());

        assertEquals(solver.getBaseLr(), 0.01, 1e-3);
        assertEquals(solver.getMaxIter(), 1000);
        assertEquals(solver.getLrPolicy(), "inv");
        assertEquals(solver.getGamma(), 1e-4, 1e-4);
        assertEquals(solver.getPower(), 0.75, 1e-2);
        assertEquals(solver.getMomentum(), 0.9f, 1e-3);
        assertEquals(solver.getWeightDecay(), 5e-4, 1e-4);
        assertEquals(solver.getRegularizationType(), "L2");
        assertEquals(solver.getStepsize(), 0);
        assertEquals(solver.getStepvalueCount(), 0);
        assertEquals(solver.getClipGradients(), -1.0, 1e-1);
        assertEquals(solver.getSolverMode().name(), "CPU");
        //For GPU
        assertEquals(solver.getDeviceId(), 0);
        assertEquals(solver.getRandomSeed(), -1);
        assertEquals(solver.getSolverType().name(), "SGD");
        //For ADAGRAD
        assertEquals(solver.getDelta(), 1e-8, 1e-8);
    }


    /**
     * Read binary net ImageNet. Note sometimes it is layers and sometimes it's layer
     *
     * @throws Exception
     */
    @Test
    public void testBinaryCaffeModelToJavaClassImageNet() throws Exception {

        // Read the Binary File to a Java Class
        NetParameter net = reader.readBinaryNet(getImageNetBinaryNetPath(), 1000);

        // Test the binary file is read in correctly
        assertEquals(net.getName(), "CaffeNet");
        assertEquals(net.getLayersCount(), 31);
        assertEquals(net.getLayers(0).getName(), "data");
        assertEquals(net.getLayers(30).getName(), "loss");
        assertEquals(net.getLayers(15).getBlobs(0).getData(0), -0.008252043f, 1e-3);
    }

    /**
     * Read TextFormat ImageNet
     *
     * @throws IOException
     */
    @Test
    public void testTextFormatNetProtoToJavaClassImageNet() throws IOException {


        // Read the Net proto-text File to a Java Class
        NetParameter net = reader.readTextFormatNet(getImageNetTextFormatNetPath());

        assertEquals(net.getName(), "nin_imagenet");
        // Not 31 because there is an extra test data layer and an accuracy layer,
        // otherwise same as the net loaded from the  binary .caffemodel file
        assertEquals(net.getLayersCount(), 33);
        assertEquals(net.getLayers(0).getName(), "data");
        assertEquals(net.getLayers(32).getName(), "loss");
    }

    /**
     * TextFormat Solver ImageNet
     * @throws IOException
     */
    @Test
    public void testTextFormatSolverProtoToJavaClassImageNet() throws IOException {


        // Read the Solver proto-text File to a Java Class
        SolverParameter solver = reader.readTextFormatSolver(getImageNetTextFormatSolverPath());

        assertEquals(solver.getMaxIter(), 450000);
        assertEquals(solver.getMomentum(), 0.9f, 1e-3);
        assertEquals(solver.getBaseLr(), 0.01, 1e-3);
        assertEquals(solver.getRegularizationType(), "L2");
    }
}
