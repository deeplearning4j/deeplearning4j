package org.deeplearning4j.gradientcheck;

import java.io.IOException;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 * Created by Alex on 12/09/2016.
 */
@Slf4j
public class LossFunctionGradientCheck {

    static {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
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
                        LossMixtureDensity.builder().gaussians(2).labelWidth(3).build(),};

        String[] outputActivationFn = new String[] {"sigmoid", //xent
                        "sigmoid", //xent
                        "tanh", //cosine
                        "tanh", //hinge -> trying to predict 1 or -1
                        "sigmoid", //kld -> probab so should be between 0 and 1
                        "softmax", //kld + softmax
                        "tanh", //l1
                        "rationaltanh", //l1
                        "softmax", //l1 + softmax
                        "tanh", //l2
                        "softmax", //l2 + softmax
                        "identity", //mae
                        "softmax", //mae + softmax
                        "identity", //mape
                        "softmax", //mape + softmax
                        "softmax", //mcxent
                        "identity", //mse
                        "softmax", //mse + softmax
                        "sigmoid", //msle  -   requires positive labels/activations due to log
                        "softmax", //msle + softmax
                        "sigmoid", //nll
                        "softmax", //nll + softmax
                        "sigmoid", //poisson - requires positive predictions due to log... not sure if this is the best option
                        "tanh", //squared hinge
                        "sigmoid", //f-measure (binary, single sigmoid output)
                        "sigmoid", //f-measure (binary, single sigmoid output)
                        "softmax", //f-measure (binary, 2-label softmax output)
                        "softmax", //f-measure (binary, 2-label softmax output)
                        "identity", // MixtureDensity
                        "tanh", // MixtureDensity + tanh
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
        };

        int[] minibatchSizes = new int[] {1, 3};


        List<String> passed = new ArrayList<>();
        List<String> failed = new ArrayList<>();

        for (int i = 0; i < lossFunctions.length; i++) {
            for (int j = 0; j < minibatchSizes.length; j++) {
                String testName = lossFunctions[i] + " - " + outputActivationFn[i] + " - minibatchSize = "
                                + minibatchSizes[j];

                Nd4j.getRandom().setSeed(12345);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().iterations(1)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(12345)
                                .updater(Updater.NONE).regularization(false).weightInit(WeightInit.DISTRIBUTION)
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
                        LossMixtureDensity.builder().gaussians(2).labelWidth(3).build(),};

        String[] outputActivationFn = new String[] {"sigmoid", //xent
                        "sigmoid", //xent
                        "tanh", //cosine
                        "tanh", //hinge -> trying to predict 1 or -1
                        "sigmoid", //kld -> probab so should be between 0 and 1
                        "softmax", //kld + softmax
                        "tanh", //l1
                        "softmax", //l1 + softmax
                        "tanh", //l2
                        "softmax", //l2 + softmax
                        "identity", //mae
                        "softmax", //mae + softmax
                        "identity", //mape
                        "softmax", //mape + softmax
                        "softmax", //mcxent
                        "identity", //mse
                        "softmax", //mse + softmax
                        "sigmoid", //msle  -   requires positive labels/activations due to log
                        "softmax", //msle + softmax
                        "sigmoid", //nll
                        "softmax", //nll + softmax
                        "sigmoid", //poisson - requires positive predictions due to log... not sure if this is the best option
                        "tanh", //squared hinge
                        "sigmoid", //f-measure (binary, single sigmoid output)
                        "sigmoid", //f-measure (binary, single sigmoid output)
                        "softmax", //f-measure (binary, 2-label softmax output)
                        "softmax", //f-measure (binary, 2-label softmax output)
                        "identity", // MixtureDensity
                        "tanh", // MixtureDensity + tanh
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
                    ObjectMapper m = new ObjectMapper();
                    String s = m.writeValueAsString(lossFunctions[i]);
                    ILossFunction lf2 = m.readValue(s, lossFunctions[i].getClass());
                    lossFunctions[i] = lf2;
                } catch (IOException ex) {
                    ex.printStackTrace();
                    assertEquals("Tests failed: serialization of " + lossFunctions[i], 0, 1);
                }
                Nd4j.getRandom().setSeed(12345);
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().iterations(1)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(12345)
                                .updater(Updater.NONE).regularization(false).weightInit(WeightInit.DISTRIBUTION)
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

    public static INDArray[] getFeaturesAndLabels(ILossFunction l, int minibatch, int nIn, int nOut, long seed) {
        return getFeaturesAndLabels(l, new int[] {minibatch, nIn}, new int[] {minibatch, nOut}, seed);
    }

    public static INDArray[] getFeaturesAndLabels(ILossFunction l, int[] featuresShape, int[] labelsShape, long seed) {
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
                Nd4j.getExecutioner().exec(new SoftMax(ret[1]), 1);
                break;
            case "LossMCXENT":
            case "LossNegativeLogLikelihood":
                ret[1] = Nd4j.zeros(labelsShape);
                if (labelsShape.length == 2) {
                    for (int i = 0; i < labelsShape[0]; i++) {
                        ret[1].putScalar(i, r.nextInt(labelsShape[1]), 1.0);
                    }
                } else if (labelsShape.length == 3) {
                    for (int i = 0; i < labelsShape[0]; i++) {
                        for (int j = 0; j < labelsShape[2]; j++) {
                            ret[1].putScalar(i, r.nextInt(labelsShape[1]), j, 1.0);
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
                        ret[1].putScalar(i, r.nextInt(labelsShape[1]), -1.0);
                    }
                } else if (labelsShape.length == 3) {
                    for (int i = 0; i < labelsShape[0]; i++) {
                        for (int j = 0; j < labelsShape[2]; j++) {
                            ret[1].putScalar(i, r.nextInt(labelsShape[1]), j, -1.0);
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
                ret[1] = Nd4j.rand(new int[] {labelsShape[0], labelWidth});
                break;
            default:
                throw new IllegalArgumentException("Unknown class: " + l.getClass().getSimpleName());
        }

        return ret;
    }


    @Test
    public void lossFunctionWeightedGradientCheck() {

        INDArray[] weights = new INDArray[] {Nd4j.create(new double[] {0.2, 0.3, 0.5}),
                        Nd4j.create(new double[] {1.0, 0.5, 2.0})};


        List<String> passed = new ArrayList<>();
        List<String> failed = new ArrayList<>();

        for (INDArray w : weights) {

            ILossFunction[] lossFunctions = new ILossFunction[] {new LossBinaryXENT(w), new LossL1(w), new LossL1(w),
                            new LossL2(w), new LossL2(w), new LossMAE(w), new LossMAE(w), new LossMAPE(w),
                            new LossMAPE(w), new LossMCXENT(w), new LossMSE(w), new LossMSE(w), new LossMSLE(w),
                            new LossMSLE(w), new LossNegativeLogLikelihood(w), new LossNegativeLogLikelihood(w),};

            String[] outputActivationFn = new String[] {"sigmoid", //xent
                            "tanh", //l1
                            "softmax", //l1 + softmax
                            "tanh", //l2
                            "softmax", //l2 + softmax
                            "identity", //mae
                            "softmax", //mae + softmax
                            "identity", //mape
                            "softmax", //mape + softmax
                            "softmax", //mcxent
                            "identity", //mse
                            "softmax", //mse + softmax
                            "sigmoid", //msle  -   requires positive labels/activations due to log
                            "softmax", //msle + softmax
                            "sigmoid", //nll
                            "softmax", //nll + softmax
            };

            int[] minibatchSizes = new int[] {1, 3};

            for (int i = 0; i < lossFunctions.length; i++) {
                for (int j = 0; j < minibatchSizes.length; j++) {
                    String testName = lossFunctions[i] + " - " + outputActivationFn[i] + " - minibatchSize = "
                                    + minibatchSizes[j] + "; weights = " + w;

                    Nd4j.getRandom().setSeed(12345);
                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().iterations(1)
                                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(12345)
                                    .updater(Updater.NONE).regularization(false).weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new UniformDistribution(-3, 3)).list()
                                    .layer(0, new DenseLayer.Builder().nIn(4).nOut(4).activation(Activation.TANH)
                                                    .build())
                                    .layer(1, new OutputLayer.Builder().lossFunction(lossFunctions[i])
                                                    .activation(outputActivationFn[i]).nIn(4).nOut(3).build())
                                    .pretrain(false).backprop(true).build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

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
