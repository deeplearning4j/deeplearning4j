package org.deeplearning4j.gradientcheck;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.OldSoftMax;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 12/09/2016.
 */
@Slf4j
public class LossFunctionGradientCheck extends BaseDL4JTest {

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-5;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    @Test
    public void lossFunctionGradientCheck() {

        ILossFunction[] lossFunctions = new ILossFunction[] {new LossBinaryXENT(), new LossBinaryXENT(),
                        new LossCosineProximity(), new LossHinge(), new LossKLD(), new LossKLD(), new LossL1(),
                        new LossL1(), new LossL1(), new LossL2(), new LossL2(), new LossMAE(), new LossMAE(),
                        new LossMAPE(), new LossMAPE(), new LossMCXENT(), new LossMSE(), new LossMSE(), new LossMSLE(),
                        new LossMSLE(), new LossNegativeLogLikelihood(), new LossNegativeLogLikelihood(),
                        new LossPoisson(), new LossSquaredHinge(), new LossFMeasure(), new LossFMeasure(2.0),
                        new LossFMeasure(), new LossFMeasure(2.0),
                        LossMixtureDensity.builder().gaussians(2).labelWidth(3).build(),
                        LossMixtureDensity.builder().gaussians(2).labelWidth(3).build(),
                        new LossMultiLabel(),
        };

        Activation[] outputActivationFn = new Activation[] {Activation.SIGMOID, //xent
                        Activation.SIGMOID, //xent
                        Activation.TANH, //cosine
                        Activation.TANH, //hinge -> trying to predict 1 or -1
                        Activation.SIGMOID, //kld -> probab so should be between 0 and 1
                        Activation.SOFTMAX, //kld + softmax
                        Activation.TANH, //l1
                        Activation.RATIONALTANH, //l1
                        Activation.SOFTMAX, //l1 + softmax
                        Activation.TANH, //l2
                        Activation.SOFTMAX, //l2 + softmax
                        Activation.IDENTITY, //mae
                        Activation.SOFTMAX, //mae + softmax
                        Activation.IDENTITY, //mape
                        Activation.SOFTMAX, //mape + softmax
                        Activation.SOFTMAX, //mcxent
                        Activation.IDENTITY, //mse
                        Activation.SOFTMAX, //mse + softmax
                        Activation.SIGMOID, //msle  -   requires positive labels/activations due to log
                        Activation.SOFTMAX, //msle + softmax
                        Activation.SIGMOID, //nll
                        Activation.SOFTMAX, //nll + softmax
                        Activation.SIGMOID, //poisson - requires positive predictions due to log... not sure if this is the best option
                        Activation.TANH, //squared hinge
                        Activation.SIGMOID, //f-measure (binary, single sigmoid output)
                        Activation.SIGMOID, //f-measure (binary, single sigmoid output)
                        Activation.SOFTMAX, //f-measure (binary, 2-label softmax output)
                        Activation.SOFTMAX, //f-measure (binary, 2-label softmax output)
                        Activation.IDENTITY, // MixtureDensity
                        Activation.TANH, // MixtureDensity + tanh
                        Activation.TANH, // MultiLabel, doesn't require any special activation, but tanh was used in paper
        };

        int[] nOut = new int[] {1, //xent
                        3, //xent
                        5, //cosine
                        3, //hinge
                        3, //kld
                        3, //kld + softmax
                        3, //l1
                        3, //l1
                        3, //l1 + softmax
                        3, //l2
                        3, //l2 + softmax
                        3, //mae
                        3, //mae + softmax
                        3, //mape
                        3, //mape + softmax
                        3, //mcxent
                        3, //mse
                        3, //mse + softmax
                        3, //msle
                        3, //msle + softmax
                        3, //nll
                        3, //nll + softmax
                        3, //poisson
                        3, //squared hinge
                        1, //f-measure (binary, single sigmoid output)
                        1, //f-measure (binary, single sigmoid output)
                        2, //f-measure (binary, 2-label softmax output)
                        2, //f-measure (binary, 2-label softmax output)
                        10, // Mixture Density
                        10, // Mixture Density + tanh
                        10, // MultiLabel
        };

        int[] minibatchSizes = new int[] {1, 3};


        List<String> passed = new ArrayList<>();
        List<String> failed = new ArrayList<>();

        for (int i = 0; i < lossFunctions.length; i++) {
            for (int j = 0; j < minibatchSizes.length; j++) {
                String testName = lossFunctions[i] + " - " + outputActivationFn[i] + " - minibatchSize = "
                                + minibatchSizes[j];

                Nd4j.getRandom().setSeed(12345);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(12345)
                                .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                .dist(new UniformDistribution(-2, 2)).list()
                                .layer(0, new DenseLayer.Builder().nIn(4).nOut(4).activation(Activation.TANH).build())
                                .layer(1, new OutputLayer.Builder().lossFunction(lossFunctions[i])
                                                .activation(outputActivationFn[i]).nIn(4).nOut(nOut[i]).build())
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
                try {
                    gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);
                } catch (Exception e) {
                    e.printStackTrace();
                    failed.add(testName + "\t" + "EXCEPTION");
                    continue;
                }

                if (gradOK) {
                    passed.add(testName);
                } else {
                    failed.add(testName);
                }

                System.out.println("\n\n");
                TestUtils.testModelSerialization(net);
            }
        }


        System.out.println("---- Passed ----");
        for (String s : passed) {
            System.out.println(s);
        }

        System.out.println("---- Failed ----");
        for (String s : failed) {
            System.out.println(s);
        }

        assertEquals("Tests failed", 0, failed.size());
    }

    @Test
    public void lossFunctionGradientCheckLossLayer() {

        ILossFunction[] lossFunctions = new ILossFunction[] {new LossBinaryXENT(), new LossBinaryXENT(),
                        new LossCosineProximity(), new LossHinge(), new LossKLD(), new LossKLD(), new LossL1(),
                        new LossL1(), new LossL2(), new LossL2(), new LossMAE(), new LossMAE(), new LossMAPE(),
                        new LossMAPE(), new LossMCXENT(), new LossMSE(), new LossMSE(), new LossMSLE(), new LossMSLE(),
                        new LossNegativeLogLikelihood(), new LossNegativeLogLikelihood(), new LossPoisson(),
                        new LossSquaredHinge(), new LossFMeasure(), new LossFMeasure(2.0), new LossFMeasure(),
                        new LossFMeasure(2.0), LossMixtureDensity.builder().gaussians(2).labelWidth(3).build(),
                        LossMixtureDensity.builder().gaussians(2).labelWidth(3).build(),
                        new LossMultiLabel()
        };

        Activation[] outputActivationFn = new Activation[] {Activation.SIGMOID, //xent
                        Activation.SIGMOID, //xent
                        Activation.TANH, //cosine
                        Activation.TANH, //hinge -> trying to predict 1 or -1
                        Activation.SIGMOID, //kld -> probab so should be between 0 and 1
                        Activation.SOFTMAX, //kld + softmax
                        Activation.TANH, //l1
                        Activation.SOFTMAX, //l1 + softmax
                        Activation.TANH, //l2
                        Activation.SOFTMAX, //l2 + softmax
                        Activation.IDENTITY, //mae
                        Activation.SOFTMAX, //mae + softmax
                        Activation.IDENTITY, //mape
                        Activation.SOFTMAX, //mape + softmax
                        Activation.SOFTMAX, //mcxent
                        Activation.IDENTITY, //mse
                        Activation.SOFTMAX, //mse + softmax
                        Activation.SIGMOID, //msle  -   requires positive labels/activations due to log
                        Activation.SOFTMAX, //msle + softmax
                        Activation.SIGMOID, //nll
                        Activation.SOFTMAX, //nll + softmax
                        Activation.SIGMOID, //poisson - requires positive predictions due to log... not sure if this is the best option
                        Activation.TANH, //squared hinge
                        Activation.SIGMOID, //f-measure (binary, single sigmoid output)
                        Activation.SIGMOID, //f-measure (binary, single sigmoid output)
                        Activation.SOFTMAX, //f-measure (binary, 2-label softmax output)
                        Activation.SOFTMAX, //f-measure (binary, 2-label softmax output)
                        Activation.IDENTITY, // MixtureDensity
                        Activation.TANH, // MixtureDensity + tanh
                        Activation.TANH, // MultiLabel
        };

        int[] nOut = new int[] {1, //xent
                        3, //xent
                        5, //cosine
                        3, //hinge
                        3, //kld
                        3, //kld + softmax
                        3, //l1
                        3, //l1 + softmax
                        3, //l2
                        3, //l2 + softmax
                        3, //mae
                        3, //mae + softmax
                        3, //mape
                        3, //mape + softmax
                        3, //mcxent
                        3, //mse
                        3, //mse + softmax
                        3, //msle
                        3, //msle + softmax
                        3, //nll
                        3, //nll + softmax
                        3, //poisson
                        3, //squared hinge
                        1, //f-measure (binary, single sigmoid output)
                        1, //f-measure (binary, single sigmoid output)
                        2, //f-measure (binary, 2-label softmax output)
                        2, //f-measure (binary, 2-label softmax output)
                        10, // Mixture Density
                        10, // Mixture Density + tanh
                        10, // MultiLabel
        };

        int[] minibatchSizes = new int[] {1, 3};
        //        int[] minibatchSizes = new int[]{3};


        List<String> passed = new ArrayList<>();
        List<String> failed = new ArrayList<>();

        for (int i = 0; i < lossFunctions.length; i++) {
            for (int j = 0; j < minibatchSizes.length; j++) {
                String testName = lossFunctions[i] + " - " + outputActivationFn[i] + " - minibatchSize = "
                                + minibatchSizes[j];

                // Serialize and de-serialize loss function
                // to ensure that we carry the parameters through
                // the serializer.
                try {
                    ObjectMapper m = NeuralNetConfiguration.mapper();
                    String s = m.writeValueAsString(lossFunctions[i]);
                    ILossFunction lf2 = m.readValue(s, lossFunctions[i].getClass());
                    lossFunctions[i] = lf2;
                } catch (IOException ex) {
                    ex.printStackTrace();
                    assertEquals("Tests failed: serialization of " + lossFunctions[i], 0, 1);
                }
                Nd4j.getRandom().setSeed(12345);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(12345)
                                .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                .dist(new UniformDistribution(-2, 2)).list()
                                .layer(0, new DenseLayer.Builder().nIn(4).nOut(nOut[i]).activation(Activation.TANH)
                                                .build())
                                .layer(1, new LossLayer.Builder().lossFunction(lossFunctions[i])
                                                .activation(outputActivationFn[i]).build())
                                .pretrain(false).backprop(true).build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                assertTrue(((LossLayer) net.getLayer(1).conf().getLayer()).getLossFn().getClass() == lossFunctions[i]
                                .getClass());

                INDArray[] inOut = getFeaturesAndLabels(lossFunctions[i], minibatchSizes[j], 4, nOut[i], 12345);
                INDArray input = inOut[0];
                INDArray labels = inOut[1];

                log.info(" ***** Starting test: {} *****", testName);
                //                System.out.println(Arrays.toString(labels.data().asDouble()));
                //                System.out.println(Arrays.toString(net.output(input,false).data().asDouble()));
                //                System.out.println(net.score(new DataSet(input,labels)));

                boolean gradOK;
                try {
                    gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);
                } catch (Exception e) {
                    e.printStackTrace();
                    failed.add(testName + "\t" + "EXCEPTION");
                    continue;
                }

                if (gradOK) {
                    passed.add(testName);
                } else {
                    failed.add(testName);
                }

                System.out.println("\n\n");
                TestUtils.testModelSerialization(net);
            }
        }


        System.out.println("---- Passed ----");
        for (String s : passed) {
            System.out.println(s);
        }

        System.out.println("---- Failed ----");
        for (String s : failed) {
            System.out.println(s);
        }

        assertEquals("Tests failed", 0, failed.size());
    }

    public static INDArray[] getFeaturesAndLabels(ILossFunction l, long minibatch, long nIn, long nOut, long seed) {
        return getFeaturesAndLabels(l, new long[] {minibatch, nIn}, new long[] {minibatch, nOut}, seed);
    }

    public static INDArray[] getFeaturesAndLabels(ILossFunction l, int[] featuresShape, int[] labelsShape, long seed) {
        return getFeaturesAndLabels(l, ArrayUtil.toLongArray(featuresShape), ArrayUtil.toLongArray(labelsShape), seed);
    }

    public static INDArray[] getFeaturesAndLabels(ILossFunction l, long[] featuresShape, long[] labelsShape, long seed) {
        Nd4j.getRandom().setSeed(seed);
        Random r = new Random(seed);
        INDArray[] ret = new INDArray[2];

        ret[0] = Nd4j.rand(featuresShape);

        switch (l.getClass().getSimpleName()) {
            case "LossBinaryXENT":
                //Want binary vector labels
                ret[1] = Nd4j.rand(labelsShape);
                BooleanIndexing.replaceWhere(ret[1], 0, Conditions.lessThanOrEqual(0.5));
                BooleanIndexing.replaceWhere(ret[1], 1, Conditions.greaterThanOrEqual(0.5));
                break;
            case "LossCosineProximity":
                //Should be real-valued??
                ret[1] = Nd4j.rand(labelsShape).subi(0.5);
                break;
            case "LossKLD":
                //KL divergence: should be a probability distribution for labels??
                ret[1] = Nd4j.rand(labelsShape);
                Nd4j.getExecutioner().exec(new OldSoftMax(ret[1]), 1);
                break;
            case "LossMCXENT":
            case "LossNegativeLogLikelihood":
                ret[1] = Nd4j.zeros(labelsShape);
                if (labelsShape.length == 2) {
                    for (int i = 0; i < labelsShape[0]; i++) {
                        // FIXME: int cast
                        ret[1].putScalar(i, r.nextInt((int) labelsShape[1]), 1.0);
                    }
                } else if (labelsShape.length == 3) {
                    for (int i = 0; i < labelsShape[0]; i++) {
                        for (int j = 0; j < labelsShape[2]; j++) {
                            // FIXME: int cast
                            ret[1].putScalar(i, r.nextInt((int) labelsShape[1]), j, 1.0);
                        }
                    }
                } else {
                    throw new UnsupportedOperationException();
                }

                break;
            case "LossHinge":
            case "LossSquaredHinge":
                ret[1] = Nd4j.ones(labelsShape);
                if (labelsShape.length == 2) {
                    for (int i = 0; i < labelsShape[0]; i++) {
                        // FIXME: int cast
                        ret[1].putScalar(i, r.nextInt((int) labelsShape[1]), -1.0);
                    }
                } else if (labelsShape.length == 3) {
                    for (int i = 0; i < labelsShape[0]; i++) {
                        for (int j = 0; j < labelsShape[2]; j++) {
                            // FIXME: int cast
                            ret[1].putScalar(i, r.nextInt((int) labelsShape[1]), j, -1.0);
                        }
                    }
                } else {
                    throw new UnsupportedOperationException();
                }
                break;
            case "LossMAPE":
                //requires non-zero values for actual...
                ret[1] = Nd4j.rand(labelsShape).addi(1.0); //1 to 2
                break;
            case "LossMAE":
            case "LossMSE":
            case "LossL1":
            case "LossL2":
                ret[1] = Nd4j.rand(labelsShape).muli(2).subi(1);
                break;
            case "LossMSLE":
                //Requires positive labels/activations due to log
                ret[1] = Nd4j.rand(labelsShape);
                break;
            case "LossPoisson":
                //Binary vector labels should be OK here??
                ret[1] = Nd4j.rand(labelsShape);
                BooleanIndexing.replaceWhere(ret[1], 0, Conditions.lessThanOrEqual(0.5));
                BooleanIndexing.replaceWhere(ret[1], 1, Conditions.greaterThanOrEqual(0.5));
                break;
            case "LossFMeasure":
                if (labelsShape[1] == 1) {
                    //single binary output case
                    ret[1] = Nd4j.getExecutioner()
                                    .exec(new BernoulliDistribution(Nd4j.createUninitialized(labelsShape), 0.5));
                    if (labelsShape[0] >= 2) {
                        //Ensure we have at least one "0" and one "1"
                        int count = ret[1].sumNumber().intValue();
                        if (count == 0) {
                            ret[1].putScalar(0, 0, 1.0);
                        } else if (count == ret[1].size(0)) {
                            ret[1].putScalar(0, 0, 0.0);
                        }
                    }
                } else {
                    //"softmax style" binary output case
                    ret[1] = Nd4j.create(labelsShape);
                    for (int i = 0; i < labelsShape[0]; i++) {
                        ret[1].putScalar(i, i % labelsShape[1], 1.0);
                    }
                }
                break;
            case "LossMixtureDensity":
                LossMixtureDensity lmd = (LossMixtureDensity) l;
                int labelWidth = lmd.getLabelWidth();
                ret[1] = Nd4j.rand(new long[] {labelsShape[0], labelWidth});
                break;
            case "LossMultiLabel":
                ret[1] = Nd4j.rand(labelsShape).lti(0.3);
                // ensure that there is no example that is all ones or all zeros
                final INDArray sum = ret[1].sum(0);
                for (int i = 0; i < labelsShape[0]; i++) {
                    final int rowSum = sum.getInt(i);
                    if (rowSum == 0) {
                        ret[1].putScalar(i, 0, 1);
                    } else if (rowSum == labelsShape[1]) {
                        ret[1].putScalar(i, 0, 0);
                    }
                }

                break;
            default:
                throw new IllegalArgumentException("Unknown class: " + l.getClass().getSimpleName());
        }

        return ret;
    }


    @Test
    public void lossFunctionWeightedGradientCheck() {
        Nd4j.getRandom().setSeed(12345);

        INDArray[] weights = new INDArray[] {Nd4j.create(new double[] {0.2, 0.3, 0.5}),
                        Nd4j.create(new double[] {1.0, 0.5, 2.0})};


        List<String> passed = new ArrayList<>();
        List<String> failed = new ArrayList<>();

        for (INDArray w : weights) {

            ILossFunction[] lossFunctions = new ILossFunction[] {new LossBinaryXENT(w), new LossL1(w), new LossL1(w),
                            new LossL2(w), new LossL2(w), new LossMAE(w), new LossMAE(w), new LossMAPE(w),
                            new LossMAPE(w), new LossMCXENT(w), new LossMSE(w), new LossMSE(w), new LossMSLE(w),
                            new LossMSLE(w), new LossNegativeLogLikelihood(w), new LossNegativeLogLikelihood(w),};

            Activation[] outputActivationFn = new Activation[] {Activation.SIGMOID, //xent
                            Activation.TANH, //l1
                            Activation.SOFTMAX, //l1 + softmax
                            Activation.TANH, //l2
                            Activation.SOFTMAX, //l2 + softmax
                            Activation.IDENTITY, //mae
                            Activation.SOFTMAX, //mae + softmax
                            Activation.IDENTITY, //mape
                            Activation.SOFTMAX, //mape + softmax
                            Activation.SOFTMAX, //mcxent
                            Activation.IDENTITY, //mse
                            Activation.SOFTMAX, //mse + softmax
                            Activation.SIGMOID, //msle  -   requires positive labels/activations due to log
                            Activation.SOFTMAX, //msle + softmax
                            Activation.SIGMOID, //nll
                            Activation.SOFTMAX, //nll + softmax
            };

            int[] minibatchSizes = new int[] {1, 3};

            for (int i = 0; i < lossFunctions.length; i++) {
                for (int j = 0; j < minibatchSizes.length; j++) {
                    String testName = lossFunctions[i] + " - " + outputActivationFn[i] + " - minibatchSize = "
                                    + minibatchSizes[j] + "; weights = " + w;

                    Nd4j.getRandom().setSeed(12345);
                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(12345)
                                    .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
//                                    .dist(new UniformDistribution(-3, 3))
                                    .dist(new NormalDistribution(0, 1))
                                    .list()
                                    .layer(0, new DenseLayer.Builder().nIn(4).nOut(4).activation(Activation.TANH)
                                                    .build())
                                    .layer(1, new OutputLayer.Builder().lossFunction(lossFunctions[i])
                                                    .activation(outputActivationFn[i]).nIn(4).nOut(3).build())
                                    .pretrain(false).backprop(true).build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    //Check params to avoid test flakiness on small or large params
                    INDArray params = net.params();
                    for( int x=0; x<params.length(); x++ ){
                        while(Math.abs(params.getDouble(x)) < 0.01 || Math.abs(params.getDouble(x)) > 1.5){
                            double d = Nd4j.getRandom().nextDouble();
                            params.putScalar(x, -1.5 + d * 3);
                        }
                    }

                    INDArray[] inOut = getFeaturesAndLabels(lossFunctions[i], minibatchSizes[j], 4, 3, 12345);
                    INDArray input = inOut[0];
                    INDArray labels = inOut[1];

                    log.info(" ***** Starting test: {} *****", testName);
                    //                System.out.println(Arrays.toString(labels.data().asDouble()));
                    //                System.out.println(Arrays.toString(net.output(input,false).data().asDouble()));
                    //                System.out.println(net.score(new DataSet(input,labels)));

                    boolean gradOK;
                    try {
                        gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);
                    } catch (Exception e) {
                        e.printStackTrace();
                        failed.add(testName + "\t" + "EXCEPTION");
                        continue;
                    }

                    if (gradOK) {
                        passed.add(testName);
                    } else {
                        failed.add(testName);
                    }

                    System.out.println("\n\n");
                }
            }
        }

        System.out.println("---- Passed ----");
        for (String s : passed) {
            System.out.println(s);
        }

        System.out.println("---- Failed ----");
        for (String s : failed) {
            System.out.println(s);
        }

        assertEquals("Tests failed", 0, failed.size());
    }
}
