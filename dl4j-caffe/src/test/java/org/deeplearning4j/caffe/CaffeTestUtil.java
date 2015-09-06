package org.deeplearning4j.caffe;

import lombok.Data;
import org.deeplearning4j.caffe.common.CaffeLoader;
import org.deeplearning4j.caffe.common.CaffeReader;
import org.deeplearning4j.caffe.common.SolverNetContainer;
import org.deeplearning4j.caffe.proto.Caffe.*;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;

/**
 * @author jeffreytang
 */
@Data
public class CaffeTestUtil {

    public CaffeTestUtil() {}

    // Reader for testing CaffeReader
    public static final CaffeReader reader = new CaffeReader();

    // Get a working SolverNetContainer (given the CaffeLoader tests all run)
    public static SolverNetContainer getLogisticSolverNet() throws IOException {
        return new CaffeLoader().binaryNet(getLogisticBinaryNetPath())
                .textFormatSolver(getLogisticTextFormatSolverPath())
                .load();
    }

    public static SolverNetContainer getImageNetSolverNet() throws IOException {
        return new CaffeLoader().binaryNet(getImageNetBinaryNetPath())
                .textFormatSolver(getImageNetTextFormatSolverPath())
                .load();
    }

    public static NetParameter getLogisticNet() throws IOException {
        return getLogisticSolverNet().getNet();
    }

    public static SolverParameter getLogisticSolver() throws IOException {
        return getLogisticSolverNet().getSolver();
    }


    public static NetParameter getImageNetNet() throws IOException {
        return getImageNetSolverNet().getNet();
    }

    public static SolverParameter getImageNetSolver() throws IOException {
        return getImageNetSolverNet().getSolver();
    }

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
