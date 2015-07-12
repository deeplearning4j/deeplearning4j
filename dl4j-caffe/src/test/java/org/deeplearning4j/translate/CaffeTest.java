package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe.NetParameter;
import org.springframework.core.io.ClassPathResource;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Created by jeffreytang on 7/11/15.
 */
public class CaffeTest {

    @Test
    public void testCaffeModelToJavaClass() throws Exception {
        // caffemodel downloaded from https://gist.github.com/mavenlin/d802a5849de39225bcc6
        String imagenetCaffeModelPath = new ClassPathResource("nin_imagenet_conv.caffemodel").getURL().getFile();

        NetParameter net = CaffeModelToJavaClass.readCaffeModel(imagenetCaffeModelPath, 1000);
        assertEquals(net.getName(), "CaffeNet");
        assertEquals(net.getLayersCount(), 31);
        assertEquals(net.getLayers(0).getName(), "data");
        assertEquals(net.getLayers(30).getName(), "loss");
        assertEquals(net.getLayers(15).getBlobs(0).getData(0), -0.008252043f, 1e-1);

    }

}
