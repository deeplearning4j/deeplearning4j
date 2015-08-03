package org.deeplearning4j.translate;

import lombok.Data;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;

/**
 * @author jeffreytang
 */
@Data
public class CaffeTestUtil {

    public static final CaffeReader reader = new CaffeReader();

    // Define all the paths as String
    public static String getImageNetBinaryNetPath() throws IOException {
        return new ClassPathResource("nin_imagenet/nin_imagenet_conv.caffemodel").getURL().getFile();
    }
    public static String getImageNetTextFormatNetPath() throws IOException{
        return new ClassPathResource("nin_imagenet/train_val.prototxt").getURL().getFile();
    }
    public static String getImageNetTextFormatSolverPath() throws IOException{
        return new ClassPathResource("nin_imagenet/solver.prototxt").getURL().getFile();
    }
    public static String getLogisticBinaryNetPath() throws IOException {
        return new ClassPathResource("iris_logistic/iris_logit_iter_1000.caffemodel").getURL().getFile();
    }
    public static String getLogisticTextFormatNetPath() throws IOException{
        return new ClassPathResource("iris_logistic/net.prototxt").getURL().getFile();
    }
    public static String getLogisticTextFormatSolverPath() throws IOException{
        return new ClassPathResource("iris_logistic/solver.prototxt").getURL().getFile();
    }

}
