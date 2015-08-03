package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe.NetParameter;
import org.deeplearning4j.caffe.Caffe.SolverParameter;
import org.springframework.core.io.ClassPathResource;
import org.junit.Test;
import java.io.IOException;
import static org.junit.Assert.*;

/**
 * @author jeffreytang
 */
public class ReadCaffeTest {

    // Define all the paths as String
    public static String getImageNetBinaryNetPath() throws IOException{
        return new ClassPathResource("nin_imagenet/nin_imagenet_conv.caffemodel").getURL().getFile();
    }
    public static String getImageNetTextFormatNetPath() throws IOException{
        return new ClassPathResource("nin_imagenet/train_val.prototxt").getURL().getFile();
    }
    public static String getImageNetTextFormatSolverPath() throws IOException{
        return new ClassPathResource("nin_imagenet/solver.prototxt").getURL().getFile();
    }

    /**
     * Test reading the ImageNet in binary format into a Java Class
     * This includes the network configuration and the pre-trained weights
     * But not the Solver parameters, e.g. max_iterations, momentum, etc
     * @throws Exception
     */
    @Test
    public void testBinaryCaffeModelToJavaClass() throws Exception {

        // Read the Binary File to a Java Class
        CaffeReader reader = new CaffeReader();
        NetParameter net = reader.readBinaryNet(getImageNetBinaryNetPath(), 1000);

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

        CaffeReader reader = new CaffeReader();

        // Read the Solver proto-text File to a Java Class
        SolverParameter solver = reader.readTextFormatSolver(getImageNetTextFormatSolverPath());

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

        CaffeReader reader = new CaffeReader();

        // Read the Net proto-text File to a Java Class
        NetParameter net = reader.readTextFormatNet(getImageNetTextFormatNetPath());

        assertEquals(net.getName(), "nin_imagenet");
        // Not 31 because there is an extra test data layer and an accuracy layer,
        // otherwise same as the net loaded from the  binary .caffemodel file
        assertEquals(net.getLayersCount(), 33);
        assertEquals(net.getLayers(0).getName(), "data");
        assertEquals(net.getLayers(32).getName(), "loss");
    }

}
