package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.OpValidationSuite;
import org.nd4j.autodiff.loss.LossFunctions;
import org.nd4j.autodiff.loss.LossInfo;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

@Slf4j
public class LossOpValidation extends BaseOpValidation {
    public LossOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testLossSimple2d() {
        OpValidationSuite.ignoreFailing();

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

            for (LossFunctions.Reduction reduction : new LossFunctions.Reduction[]{
                    LossFunctions.Reduction.MEAN_BY_COUNT, LossFunctions.Reduction.MEAN_BY_WEIGHT, LossFunctions.Reduction.SUM}) {

                SameDiff sd = SameDiff.create();

                int nOut = 4;
                int minibatch = 10;
                SDVariable input = sd.var("in", new int[]{-1, nOut});
                SDVariable labels = sd.var("labels", new int[]{-1, nOut});

                INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
                INDArray labelsArr = Nd4j.randn(minibatch, nOut).muli(100);

                LossInfo lossInfo;
                INDArray expOut;
                switch (fn) {
                    case "mse":
                        lossInfo = LossFunctions.mse("out", input, labels, null, reduction, 1);
                        expOut = inputArr.sub(labelsArr);
                        expOut.muli(expOut);
                        expOut = expOut.mean(Integer.MAX_VALUE);
                        break;
                    case "l1":
                        lossInfo = LossFunctions.l1("out", input, labels, null, reduction, 1);
                        //L1 = sum abs error
                        expOut = Transforms.abs(inputArr.sub(labelsArr)).sum(1);
                        expOut = expOut.mean(Integer.MAX_VALUE);
                        break;
                    case "l2":
                        lossInfo = LossFunctions.l2("out", input, labels, null, reduction, 1);
                        //L2 = sum squared error
                        expOut = Transforms.pow(inputArr.sub(labelsArr), 2.0).sum(1).mean(Integer.MAX_VALUE);
                        break;
                    case "mcxent":
                        lossInfo = LossFunctions.mcxent("out", input, labels, null, reduction, 1);
                        //mcxent = sum label * log(prob)
                        expOut = labelsArr.mul(Transforms.log(inputArr)).sum(1).mean(Integer.MAX_VALUE);
                        break;
                    default:
                        throw new RuntimeException();
                }


                String msg = "test: " + lossInfo.getLossName() + ", reduction=" + reduction;
                log.info("*** Starting test: " + msg);


                sd.associateArrayWithVariable(inputArr, input);
                sd.associateArrayWithVariable(labelsArr, labels);

                TestCase tc = new TestCase(sd)
                        .expectedOutput("out", expOut);

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testLossWeights2d() {
        OpValidationSuite.ignoreFailing();

        String[] weightTypes = new String[]{"none", "per-example", "per-output", "per-example-output"};

        Nd4j.getRandom().setSeed(12345);

        int nOut = 4;
        int minibatch = 10;
        List<String> failed = new ArrayList<>();

        for (String weightType : weightTypes) {

            for (boolean binary : new boolean[]{true, false}) {  //Binary mask (like DL4J) or arbitrary weights?

                int[] weightShape;
                switch (weightType) {
                    case "none":
                        weightShape = null;
                        break;
                    case "per-example":
                        weightShape = new int[]{minibatch, 1};
                        break;
                    case "per-output":
                        weightShape = new int[]{1, nOut};
                        break;
                    case "per-example-output":
                        weightShape = new int[]{minibatch, nOut};
                        break;
                    default:
                        throw new RuntimeException("Unknown type: " + weightType);
                }

                INDArray weightArr = null;
                if (!"none".equals(weightType)) {
                    if (binary) {
                        weightArr = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(weightShape), 0.5));
                    } else {
                        weightArr = Nd4j.rand(weightShape).muli(2.0);
                    }
                }

                for (LossFunctions.Reduction reduction : new LossFunctions.Reduction[]{
                        LossFunctions.Reduction.MEAN_BY_COUNT, LossFunctions.Reduction.MEAN_BY_WEIGHT, LossFunctions.Reduction.SUM}) {

                    for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

                        SameDiff sd = SameDiff.create();


                        SDVariable input = sd.var("in", new int[]{-1, nOut});
                        SDVariable labels = sd.var("labels", new int[]{-1, nOut});
                        SDVariable weight = null;
                        if (!"none".equals(weightType)) {
                            weight = sd.var("weights", weightArr);
                        }

                        INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
                        INDArray labelsArr = Nd4j.randn(minibatch, nOut).muli(100);

                        LossInfo lossInfo;
                        switch (fn) {
                            case "mse":
                                lossInfo = LossFunctions.mse("out", input, labels, weight, reduction, 1);
                                break;
                            case "l1":
                                lossInfo = LossFunctions.l1("out", input, labels, weight, reduction, 1);
                                //L1 = sum abs error
                                break;
                            case "l2":
                                lossInfo = LossFunctions.l2("out", input, labels, weight, reduction, 1);
                                //L2 = sum squared error
                                break;
                            case "mcxent":
                                lossInfo = LossFunctions.mcxent("out", input, labels, weight, reduction, 1);
                                //mcxent = sum label * log(prob)
                                break;
                            default:
                                throw new RuntimeException();
                        }


                        String msg = "lossFn=" + fn + ", reduction=" + reduction + ", weightType=" + weightType + ", binaryWeight=" + binary;
                        log.info("*** Starting test: " + msg);

                        sd.associateArrayWithVariable(inputArr, input);
                        sd.associateArrayWithVariable(labelsArr, labels);
                        if (weight != null) {
                            sd.associateArrayWithVariable(weightArr, weight);
                        }

                        TestCase tc = new TestCase(sd);
                        String error = OpValidation.validate(tc);
                        if(error != null){
                            failed.add(name);
                        }
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testLossWeights3d() {
        OpValidationSuite.ignoreFailing();

        String[] weightTypes = new String[]{"none", "per-example", "per-output", "per-timestep",
                "per-example-output", "per-example-timestep", "per-output-timestep", "per-all"};

        Nd4j.getRandom().setSeed(12345);

        int nOut = 4;
        int minibatch = 10;
        int tsLength = 5;

        List<String> failed = new ArrayList<>();

        for (String weightType : weightTypes) {

            for (boolean binary : new boolean[]{true, false}) {  //Binary mask (like DL4J) or arbitrary weights?

                int[] weightShape;
                switch (weightType) {
                    case "none":
                        weightShape = null;
                        break;
                    case "per-example":
                        weightShape = new int[]{minibatch, 1, 1};
                        break;
                    case "per-output":
                        weightShape = new int[]{1, nOut, 1};
                        break;
                    case "per-timestep":
                        weightShape = new int[]{1,1, tsLength};
                        break;
                    case "per-example-output":
                        weightShape = new int[]{minibatch, nOut, 1};
                        break;
                    case "per-example-timestep":
                        weightShape = new int[]{minibatch, 1, nOut};
                        break;
                    case "per-output-timestep":
                        weightShape = new int[]{1, nOut, tsLength};
                        break;
                    case "per-all":
                        weightShape = new int[]{minibatch, nOut, tsLength};
                        break;
                    default:
                        throw new RuntimeException("Unknown type: " + weightType);
                }

                INDArray weightArr = null;
                if (!"none".equals(weightType)) {
                    if (binary) {
                        weightArr = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(weightShape), 0.5));
                    } else {
                        weightArr = Nd4j.rand(weightShape).muli(2.0);
                    }
                }

                for (LossFunctions.Reduction reduction : new LossFunctions.Reduction[]{
                        LossFunctions.Reduction.MEAN_BY_COUNT, LossFunctions.Reduction.MEAN_BY_WEIGHT, LossFunctions.Reduction.SUM}) {

                    for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

                        SameDiff sd = SameDiff.create();


                        SDVariable input = sd.var("in", new int[]{-1, nOut, -1});
                        SDVariable labels = sd.var("labels", new int[]{-1, nOut, -1});
                        SDVariable weight = null;
                        if (!"none".equals(weightType)) {
                            weight = sd.var("weights", weightArr);
                        }

                        INDArray inputArr = Nd4j.randn(new int[]{minibatch, nOut, tsLength}).muli(10);
                        INDArray labelsArr = Nd4j.randn(new int[]{minibatch, nOut, tsLength}).muli(10);

                        LossInfo lossInfo;
                        switch (fn) {
                            case "mse":
                                lossInfo = LossFunctions.mse("out", input, labels, weight, reduction, 1, 2);
                                break;
                            case "l1":
                                lossInfo = LossFunctions.l1("out", input, labels, weight, reduction, 1, 2);
                                //L1 = sum abs error
                                break;
                            case "l2":
                                lossInfo = LossFunctions.l2("out", input, labels, weight, reduction, 1, 2);
                                //L2 = sum squared error
                                break;
                            case "mcxent":
                                lossInfo = LossFunctions.mcxent("out", input, labels, weight, reduction, 1, 2);
                                //mcxent = sum label * log(prob)
                                break;
                            default:
                                throw new RuntimeException();
                        }


                        String msg = "lossFn=" + fn + ", reduction=" + reduction + ", weightType=" + weightType + ", binaryWeight=" + binary;
                        log.info("*** Starting test: " + msg);

                        sd.associateArrayWithVariable(inputArr, input);
                        sd.associateArrayWithVariable(labelsArr, labels);
                        if (weight != null) {
                            sd.associateArrayWithVariable(weightArr, weight);
                        }

                        INDArray out = sd.execAndEndResult();
                        assertEquals(1, out.length());

                        TestCase tc = new TestCase(sd);
                        String error = OpValidation.validate(tc);
                        if(error != null){
                            failed.add(name);
                        }
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testLossWeights4d() {
        OpValidationSuite.ignoreFailing();

        String[] weightTypes = new String[]{"none", "per-example", "per-depth", "per-height", "per-width",
                "per-height-width", "per-depth-height", "per-depth-width", "per-example-depth", "per-example-height",
                "per-example-height-width", "per-all"};

        Nd4j.getRandom().setSeed(12345);

        //Assume NCHW format here
        int minibatch = 10;
        int depth = 4;
        int h = 5;
        int w = 6;
        List<String> failed = new ArrayList<>();

        for (String weightType : weightTypes) {

            for (boolean binary : new boolean[]{true, false}) {  //Binary mask (like DL4J) or arbitrary weights?

                int[] weightShape;
                switch (weightType) {
                    case "none":
                        weightShape = null;
                        break;
                    case "per-example":
                        weightShape = new int[]{minibatch, 1, 1, 1};
                        break;
                    case "per-depth":
                        weightShape = new int[]{1, depth, 1, 1};
                        break;
                    case "per-height":
                        weightShape = new int[]{1,1, h, 1};
                        break;
                    case "per-width":
                        weightShape = new int[]{1, 1, 1, w};
                        break;
                    case "per-height-width":
                        weightShape = new int[]{1, 1, h, w};
                        break;
                    case "per-depth-height":
                        weightShape = new int[]{1,depth, h, 1};
                        break;
                    case "per-depth-width":
                        weightShape = new int[]{1,depth, 1, w};
                        break;
                    case "per-example-depth":
                        weightShape = new int[]{minibatch, depth, 1, 1};
                        break;
                    case "per-example-height":
                        weightShape = new int[]{minibatch, 1, h, 1};
                        break;
                    case "per-example-height-width":
                        weightShape = new int[]{minibatch, 1, h, w};
                        break;
                    case "per-all":
                        weightShape = new int[]{minibatch, depth, h, w};
                        break;
                    default:
                        throw new RuntimeException("Unknown type: " + weightType);
                }

                INDArray weightArr = null;
                if (!"none".equals(weightType)) {
                    if (binary) {
                        weightArr = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(weightShape), 0.5));
                    } else {
                        weightArr = Nd4j.rand(weightShape).muli(2.0);
                    }
                }

                for (LossFunctions.Reduction reduction : new LossFunctions.Reduction[]{
                        LossFunctions.Reduction.MEAN_BY_COUNT, LossFunctions.Reduction.MEAN_BY_WEIGHT, LossFunctions.Reduction.SUM}) {

                    for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

                        SameDiff sd = SameDiff.create();


                        SDVariable input = sd.var("in", new int[]{-1, depth, -1, -1});
                        SDVariable labels = sd.var("labels", new int[]{-1, depth, -1, -1});
                        SDVariable weight = null;
                        if (!"none".equals(weightType)) {
                            weight = sd.var("weights", weightArr);
                        }

                        INDArray inputArr = Nd4j.randn(new int[]{minibatch, depth, h, w}).muli(10);
                        INDArray labelsArr = Nd4j.randn(new int[]{minibatch, depth, h, w}).muli(10);

                        LossInfo lossInfo;
                        switch (fn) {
                            case "mse":
                                lossInfo = LossFunctions.mse("out", input, labels, weight, reduction, 1, 2, 3);
                                break;
                            case "l1":
                                lossInfo = LossFunctions.l1("out", input, labels, weight, reduction, 1, 2, 3);
                                //L1 = sum abs error
                                break;
                            case "l2":
                                lossInfo = LossFunctions.l2("out", input, labels, weight, reduction, 1, 2, 3);
                                //L2 = sum squared error
                                break;
                            case "mcxent":
                                lossInfo = LossFunctions.mcxent("out", input, labels, weight, reduction, 1, 2, 3);
                                //mcxent = sum label * log(prob)
                                break;
                            default:
                                throw new RuntimeException();
                        }


                        String msg = "lossFn=" + fn + ", reduction=" + reduction + ", weightType=" + weightType + ", binaryWeight=" + binary;
                        log.info("*** Starting test: " + msg);

                        sd.associateArrayWithVariable(inputArr, input);
                        sd.associateArrayWithVariable(labelsArr, labels);
                        if (weight != null) {
                            sd.associateArrayWithVariable(weightArr, weight);
                        }

                        INDArray out = sd.execAndEndResult();
                        assertEquals(1, out.length());

                        TestCase tc = new TestCase(sd);

                        String error = OpValidation.validate(tc);
                        if(error != null){
                            failed.add(name);
                        }
                    }
                }
            }
        }
        assertEquals(failed.toString(), 0, failed.size());
    }
}
