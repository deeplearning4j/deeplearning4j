package org.deeplearning4j.gradientcheck;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 12/09/2016.
 */
@Slf4j
public class LossFunctionGradientCheck {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-5;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-10;

    @Test
    public void lossFunctionGradientCheck(){

        ILossFunction[] lossFunctions = new ILossFunction[]{
                new LossBinaryXENT(),
                new LossCosineProximity(),
                new LossHinge(),
                new LossKLD(),
                new LossKLD(),
                new LossL1(),
                new LossL1(),
                new LossL2(),
                new LossL2(),
                new LossMAE(),
                new LossMAE(),
                new LossMAPE(),
                new LossMAPE(),
                new LossMCXENT(),
                new LossMSE(),
                new LossMSE(),
                new LossMSLE(),
                new LossMSLE(),
                new LossNegativeLogLikelihood(),
                new LossNegativeLogLikelihood(),
                new LossPoisson(),
                new LossSquaredHinge()
        };

        String[] outputActivationFn = new String[]{
                "sigmoid",  //xent
                "tanh",     //cosine
                "tanh",     //hinge -> trying to predict 1 or -1
                "sigmoid",  //kld -> probab so should be between 0 and 1
                "softmax",  //kld + softmax
                "tanh",     //l1
                "softmax",  //l1 + softmax
                "tanh",     //l2
                "softmax",  //l2 + softmax
                "identity", //mae
                "softmax",  //mae + softmax
                "identity", //mape
                "softmax",  //mape + softmax
                "softmax",  //mcxent
                "identity", //mse
                "softmax",  //mse + softmax
                "sigmoid",  //msle  -   requires positive labels/activations due to log
                "softmax",  //msle + softmax
                "sigmoid",  //nll
                "softmax",  //nll + softmax
                "sigmoid",  //poisson - requires positive predictions due to log... not sure if this is the best option
                "tanh"      //squared hinge
        };

        int[] nOut = new int[]{
                1,          //xent
                5,          //cosine
                3,          //hinge
                3,          //kld
                3,          //kld + softmax
                3,          //l1
                3,          //l1 + softmax
                3,          //l2
                3,          //l2 + softmax
                3,          //mae
                3,          //mae + softmax
                3,          //mape
                3,          //mape + softmax
                3,          //mcxent
                3,          //mse
                3,          //mse + softmax
                3,          //msle
                3,          //msle + softmax
                3,          //nll
                3,          //nll + softmax
                3,          //poisson
                3           //squared hinge
        };

        int[] minibatchSizes = new int[]{1, 3};
//        int[] minibatchSizes = new int[]{3};


        List<String> passed = new ArrayList<>();
        List<String> failed = new ArrayList<>();

        for( int i=0; i<lossFunctions.length; i++ ){
            for( int j=0; j<minibatchSizes.length; j++ ) {
                String testName = lossFunctions[i] + " - " + outputActivationFn[i] + " - minibatchSize = " + minibatchSizes[j];

                Nd4j.getRandom().setSeed(12345);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .seed(12345)
                        .updater(Updater.NONE)
                        .regularization(false)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(-2, 2))
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(4).activation("tanh").build())
                        .layer(1, new OutputLayer.Builder()
                                .lossFunction(lossFunctions[i])
                                .activation(outputActivationFn[i])
                                .nIn(4).nOut(nOut[i])
                                .build())
                        .pretrain(false).backprop(true).build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                INDArray[] inOut = getFeaturesAndLabels(lossFunctions[i], minibatchSizes[j], 4, nOut[i], 12345);
                INDArray input = inOut[0];
                INDArray labels = inOut[1];

                log.info(" ***** Starting test: {} *****", testName);
//                System.out.println(Arrays.toString(labels.data().asDouble()));
//                System.out.println(Arrays.toString(net.output(input,false).data().asDouble()));
//                System.out.println(net.score(new DataSet(input,labels)));

                boolean gradOK;
                try{
                    gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                            PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);
                } catch(Exception e){
                    e.printStackTrace();
                    failed.add(testName + "\t" + "EXCEPTION");
                    continue;
                }

                if(gradOK){
                    passed.add(testName);
                } else {
                    failed.add(testName);
                }

                System.out.println("\n\n");
            }
        }


        System.out.println("---- Passed ----");
        for(String s : passed){
            System.out.println(s);
        }

        System.out.println("---- Failed ----");
        for(String s : failed){
            System.out.println(s);
        }

        assertEquals("Tests failed", 0, failed.size());
    }

    private static INDArray[] getFeaturesAndLabels(ILossFunction l, int minibatch, int nIn, int nOut, long seed){
        Nd4j.getRandom().setSeed(seed);
        Random r = new Random(seed);
        INDArray[] ret = new INDArray[2];

        ret[0] = Nd4j.rand(minibatch, nIn);

        switch(l.getClass().getSimpleName()){
            case "LossBinaryXENT":
                //Want binary vector labels
                ret[1] = Nd4j.rand(minibatch, nOut);
                BooleanIndexing.replaceWhere(ret[1],0, Conditions.lessThanOrEqual(0.5));
                BooleanIndexing.replaceWhere(ret[1],1, Conditions.greaterThanOEqual(0.5));
                break;
            case "LossCosineProximity":
                //Should be real-valued??
                ret[1] = Nd4j.rand(minibatch, nOut).subi(0.5);
                break;
            case "LossKLD":
                //KL divergence: should be a probability distribution for labels??
                ret[1] = Nd4j.rand(minibatch, nOut);
                Nd4j.getExecutioner().exec(new SoftMax(ret[1]),1);
                break;
            case "LossMCXENT":
            case "LossNegativeLogLikelihood":
                ret[1] = Nd4j.zeros(minibatch, nOut);
                for( int i=0; i<minibatch; i++ ){
                    ret[1].putScalar(i, r.nextInt(nOut), 1.0);
                }
                break;
            case "LossHinge":
            case "LossSquaredHinge":
                ret[1] = Nd4j.ones(minibatch, nOut);
                for( int i=0; i<minibatch; i++ ){
                    ret[1].putScalar(i, r.nextInt(nOut), -1.0);
                }
                break;
            case "LossMAPE":
                //requires non-zero values for actual...
                ret[1] = Nd4j.rand(minibatch, nOut).addi(1.0);      //1 to 2
                break;
            case "LossMAE":
            case "LossMSE":
            case "LossL1":
            case "LossL2":
                ret[1] = Nd4j.rand(minibatch, nOut).muli(2).subi(1);
                break;
            case "LossMSLE":
                //Requires positive labels/activations due to log
                ret[1] = Nd4j.rand(minibatch, nOut);
                break;
            case "LossPoisson":
                //Binary vector labels should be OK here??
                ret[1] = Nd4j.rand(minibatch, nOut);
                BooleanIndexing.replaceWhere(ret[1],0, Conditions.lessThanOrEqual(0.5));
                BooleanIndexing.replaceWhere(ret[1],1, Conditions.greaterThanOEqual(0.5));
                break;
            default:
                throw new IllegalArgumentException("Unknown class: " + l.getClass().getSimpleName());
        }

        return ret;
    }

}
