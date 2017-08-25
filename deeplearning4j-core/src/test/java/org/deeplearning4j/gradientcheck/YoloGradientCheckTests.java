package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Created by nyghtowl on 9/1/15.
 */
public class YoloGradientCheckTests {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testYoloOutputLayer() {
        int depthIn = 2;


        int[] minibatchSizes = {2};
        int w = 5;
        int h = 5;
        int c = 5;
        int b = 4;

        int yoloDepth = 5*b + c;

        INDArray bbPrior = Nd4j.rand(b, 2).muliRowVector(Nd4j.create(new double[]{w, h}));
        Activation a = Activation.TANH;

        Nd4j.getRandom().setSeed(12345);

        for (int mb : minibatchSizes) {
            INDArray input = Nd4j.rand(new int[]{mb, depthIn, h, w});
            INDArray labels = yoloLabels(mb, c, h, w);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                    .learningRate(1.0).updater(Updater.SGD)
                    .activation(a)
                    .convolutionMode(ConvolutionMode.Same)
                    .list()
                    .layer(new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1)
                            .nIn(depthIn).nOut(yoloDepth).build())//output: (5-2+0)/1+1 = 4
                    .layer(new Yolo2OutputLayer.Builder()
                            .boundingBoxePriors(bbPrior)
                            .build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            String msg = "YOLO, mb = " + mb;
            System.out.println(msg);

            boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

            assertTrue(msg, gradOK);
        }
    }

    private static INDArray yoloLabels(int mb, int c, int h, int w){
        int labelDepth = 4 + c;
        INDArray labels = Nd4j.zeros(mb, labelDepth, h, w);
        //put 1 object per minibatch, at positions (0,0), (1,1) etc.
        //Positions for label boxes: (1,1) to (2,2), (2,2) to (4,4) etc

        for( int i=0; i<mb; i++ ){
            //Class labels
            labels.putScalar(i, 4 + i%c, i%h, i%w, 1);

            //BB coordinates (top left, bottom right)
            labels.putScalar(i, 0, 0, 0, i%w);
            labels.putScalar(i, 1, 0, 0, i%h);
            labels.putScalar(i, 2, 0, 0, (i%w)+1);
            labels.putScalar(i, 3, 0, 0, (i%h)+1);
        }

        return labels;
    }
}
