package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe;
import org.deeplearning4j.caffe.Caffe.NetParameter;
import org.springframework.core.io.ClassPathResource;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

/**
 * Created by jeffreytang on 7/11/15.
 */
public class CaffeTest {

    /**
     * Test reading the ImageNet in binary format into a Java Class
     * This includes the network configuration and the pre-trained weights
     * But not the Solver parameters, e.g. max_iterations, momentum, etc
     * @throws Exception
     */
    @Test
    public void testBinaryCaffeModelToJavaClass() throws Exception {
        // caffemodel downloaded from https://gist.github.com/mavenlin/d802a5849de39225bcc6
        String imagenetCaffeModelPath = new ClassPathResource("nin_imagenet/nin_imagenet_conv.caffemodel").getURL().getFile();

        // Read the Binary File to a Java Class
        NetParameter net = CaffeModelToJavaClass.readBinaryCaffeModel(imagenetCaffeModelPath, 1000);

        // Test the binary file is read in correctly
        assertEquals(net.getName(), "CaffeNet");
        assertEquals(net.getLayersCount(), 31);
        assertEquals(net.getLayers(0).getName(), "data");
        assertEquals(net.getLayers(30).getName(), "loss");
        assertEquals(net.getLayers(15).getBlobs(0).getData(0), -0.008252043f, 1e-3);
    }

    /**
     * Test reading the ImageNet Solver parameters in a proto-text (non binary format) file
     * @throws IOException
     */
    @Test
    public void testTextFormatSolverProtoToJavaClass() throws IOException {
        // caffemodel downloaded from https://gist.github.com/mavenlin/d802a5849de39225bcc6
        String imagenetSolverProtoPath = new ClassPathResource("nin_imagenet/solver.prototxt").getURL().getFile();

        // Read the Solver proto-text File to a Java Class
        Caffe.SolverParameter solver = CaffeModelToJavaClass.readTextFormatSolverProto(imagenetSolverProtoPath);

        assertEquals(solver.getMaxIter(), 450000);
        assertEquals(solver.getMomentum(), 0.9f, 1e-3);
        assertEquals(solver.getBaseLr(), 0.01, 1e-3);
        assertEquals(solver.getRegularizationType(), "L2");
    }

    /**
     * Test reading the ImageNet Network parameters in a proto-text (non binary format) file
     * Without the weights and Solver parameters
     * Useful if you do not have the pre-trained weights in a .caffemodel binary file
     * @throws IOException
     */
    @Test
    public void testTextFormatNetProtoToJavaClass() throws IOException {
        // caffemodel downloaded from https://gist.github.com/mavenlin/d802a5849de39225bcc6
        String imagenetNetProtoPath = new ClassPathResource("nin_imagenet/train_val.prototxt").getURL().getFile();

        // Read the Net proto-text File to a Java Class
        Caffe.NetParameter net = CaffeModelToJavaClass.readTextFormatNetProto(imagenetNetProtoPath);

        assertEquals(net.getName(), "nin_imagenet");
        // Not 31 because there is an extra test data layer and an accuracy layer,
        // otherwise same as the net loaded from the  binary .caffemodel file
        assertEquals(net.getLayersCount(), 33);
        assertEquals(net.getLayers(0).getName(), "data");
        assertEquals(net.getLayers(32).getName(), "loss");
    }

}
