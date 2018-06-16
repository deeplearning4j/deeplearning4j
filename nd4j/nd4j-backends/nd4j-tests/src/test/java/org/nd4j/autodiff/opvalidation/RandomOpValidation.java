package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.random.custom.DistributionUniform;
import org.nd4j.linalg.api.ops.random.custom.RandomBernoulli;
import org.nd4j.linalg.api.ops.random.custom.RandomExponential;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

@Slf4j
public class RandomOpValidation extends BaseOpValidation {

    public RandomOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testRandomOpsSDVarShape() {

        List<String> failed = new ArrayList<>();

        for (double[] shape : Arrays.asList(new double[]{1000.0}, new double[]{100, 10}, new double[]{40, 5, 5})) {

            for (int i = 0; i < 4; i++) {
                INDArray arr = Nd4j.trueVector(shape);

                Nd4j.getRandom().setSeed(12345);
                SameDiff sd = SameDiff.create();
                SDVariable shapeVar = sd.var("shape", arr);

                SDVariable rand;
                Function<INDArray, String> checkFn;
                String name;
                switch (i) {
                    case 0:
                        name = "randomUniform";
                        rand = sd.randomUniform(1, 2, shapeVar);
                        checkFn = in -> {
                            double min = in.minNumber().doubleValue();
                            double max = in.maxNumber().doubleValue();
                            double mean = in.meanNumber().doubleValue();
                            if (min >= 1 && max <= 2 && (in.length() == 1 || Math.abs(mean - 1.5) < 0.1))
                                return null;
                            return "Failed: min = " + min + ", max = " + max + ", mean = " + mean;
                        };
                        break;
                    case 1:
                        name = "randomNormal";
                        rand = sd.randomNormal(1, 1, shapeVar);
                        checkFn = in -> {
                            double mean = in.meanNumber().doubleValue();
                            double stdev = in.std(true).getDouble(0);
                            if (in.length() == 1 || (Math.abs(mean - 1) < 0.1 && Math.abs(stdev - 1) < 0.1))
                                return null;
                            return "Failed: mean = " + mean + ", stdev = " + stdev;
                        };
                        break;
                    case 2:
                        name = "randomBernoulli";
                        rand = sd.randomBernoulli(0.5, shapeVar);
                        checkFn = in -> {
                            double mean = in.meanNumber().doubleValue();
                            double min = in.minNumber().doubleValue();
                            double max = in.maxNumber().doubleValue();
                            int sum0 = Transforms.not(in.dup()).sumNumber().intValue();
                            int sum1 = in.sumNumber().intValue();
                            if ((in.length() == 1 && min == max && (min == 0 || min == 1)) ||
                                    (Math.abs(mean - 0.5) < 0.1 && min == 0 && max == 1 && (sum0 + sum1) == in.length()))
                                return null;
                            return "Failed: bernoulli - sum0 = " + sum0 + ", sum1 = " + sum1;
                        };
                        break;
                    case 3:
                        name = "randomExponential";
                        final double lambda = 2;
                        rand = sd.randomExponential(lambda, shapeVar);
                        checkFn = in -> {
                            double mean = in.meanNumber().doubleValue();
                            double min = in.minNumber().doubleValue();
                            double std = in.stdNumber().doubleValue();
                            //mean: 1/lambda; std: 1/lambda
                            if ((in.length() == 1 && min > 0) || (Math.abs(mean - 1 / lambda) < 0.1 && min >= 0 && Math.abs(std - 1 / lambda) < 0.1))
                                return null;
                            return "Failed: exponential: mean=" + mean + ", std = " + std + ", min=" + min;
                        };
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable loss;
                if (shape.length > 0) {
                    loss = rand.std(true);
                } else {
                    loss = rand.mean();
                }

                String msg = name + " - " + Arrays.toString(shape);
                TestCase tc = new TestCase(sd)
                        .gradCheckSkipVariables("shape")
                        .testName(msg)
                        .expected(rand, checkFn);

                log.info("TEST: " + msg);

                String err = OpValidation.validate(tc, true);
                if (err != null) {
                    failed.add(err);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testUniformRankSimple() {

        INDArray arr = Nd4j.trueVector(new double[]{100.0});
//        OpTestCase tc = new OpTestCase(DynamicCustomOp.builder("randomuniform")
//                .addInputs(arr)
//                .addOutputs(Nd4j.createUninitialized(new long[]{100}))
//                .addFloatingPointArguments(0.0, 1.0)
//                .build());

//        OpTestCase tc = new OpTestCase(new DistributionUniform(arr, Nd4j.createUninitialized(new long[]{100}), 0, 1));
        OpTestCase tc = new OpTestCase(new RandomBernoulli(arr, Nd4j.createUninitialized(new long[]{100}), 0.5));

        tc.expectedOutput(0, new long[]{100}, in -> {
            double min = in.minNumber().doubleValue();
            double max = in.maxNumber().doubleValue();
            double mean = in.meanNumber().doubleValue();
            if (min >= 0 && max <= 1 && (in.length() == 1 || Math.abs(mean - 0.5) < 0.2))
                return null;
            return "Failed: min = " + min + ", max = " + max + ", mean = " + mean;
        });

        String err = OpValidation.validate(tc);
        assertNull(err);

        double d = arr.getDouble(0);

        assertEquals(100.0, d, 0.0);

    }


    @Test
    public void testRandomExponential() {
        long length = 1_000_000;
        INDArray shape = Nd4j.trueVector(new double[]{length});
        INDArray out = Nd4j.createUninitialized(new long[]{length});
        double lambda = 2;
        RandomExponential op = new RandomExponential(shape, out, lambda);

        Nd4j.getExecutioner().exec(op);

        double min = out.minNumber().doubleValue();
        double mean = out.meanNumber().doubleValue();
        double std = out.stdNumber().doubleValue();

        double expMean = 1.0/lambda;
        double expStd = 1.0/lambda;

        assertTrue(min >= 0.0);
        assertEquals("mean", expMean, mean, 0.1);
        assertEquals("std", expStd, std, 0.1);
    }
}
