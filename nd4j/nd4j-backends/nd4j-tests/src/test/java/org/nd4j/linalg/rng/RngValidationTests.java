package org.nd4j.linalg.rng;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform;
import org.nd4j.linalg.api.ops.random.impl.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

@Slf4j
public class RngValidationTests {

    @Builder(builderClassName = "TestCaseBuilder")
    @Data
    public static class TestCase {
        private String opType;
        private DataType dataType;
        @Builder.Default private long rngSeed = 12345;
        private long[] shape;
        private double minValue;
        private double maxValue;
        private boolean minValueInclusive;
        private boolean maxValueInclusive;
        private Double expectedMean;
        private Double expectedStd;
        @Builder.Default private double meanRelativeErrorTolerance = 0.01;
        @Builder.Default private double stdRelativeErrorTolerance = 0.01;
        private Double meanMinAbsErrorTolerance;    //Consider relative error between 0 and 0.001: relative error is 1.0, but absolute error is small
        private Double stdMinAbsErrorTolerance;
        @Builder.Default private Map<String,Object> args = new LinkedHashMap<>();

        public static class TestCaseBuilder {

            public TestCaseBuilder arg(String arg, Object value){
                if(args == null) {
                    args(new LinkedHashMap<>());
                }
                args.put(arg, value);
                return this;
            }

            public TestCaseBuilder shape(long... shape){
                this.shape = shape;
                return this;
            }
        }

        public INDArray arr(){
            Preconditions.checkState(shape != null, "Shape is null");
            INDArray arr = Nd4j.createUninitialized(dataType, shape);
            arr.assign(Double.NaN);     //Assign NaNs to help detect implementation issues
            return arr;
        }

        public <T> T prop(String s){
            Preconditions.checkState(args != null && args.containsKey(s), "Property \"%s\" not found. All properties: %s", s, args);
            return (T)args.get(s);
        }
    }


    @Test
    public void validateRngDistributions(){

        List<TestCase> testCases = new ArrayList<>();
        for(DataType type : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            //Legacy (non-custom) RNG ops:
            testCases.add(TestCase.builder().opType("bernoulli").dataType(type).shape(new long[0]).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.5).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("bernoulli").dataType(type).shape(1000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.5)
                    .expectedMean(0.5).expectedStd(Math.sqrt(0.5*0.5) /*var = p*(1-p)*/).build());
            testCases.add(TestCase.builder().opType("bernoulli").dataType(type).shape(100,10000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.2)
                    .expectedMean(0.2).expectedStd(Math.sqrt(0.2*(1-0.2)) /*var = p*(1-p)*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            testCases.add(TestCase.builder().opType("uniform").dataType(type).shape(new long[0]).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("min", 0.0).arg("max", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("uniform").dataType(type).shape(1000).minValue(1).maxValue(2).minValueInclusive(true).maxValueInclusive(true).arg("min", 1.0).arg("max",2.0)
                    .expectedMean((1+2)/2.0).expectedStd(Math.sqrt(1/12.0 * Math.pow(2.0-1.0, 2)) /*Var: 1/12 * (b-a)^2*/).build());
            testCases.add(TestCase.builder().opType("uniform").dataType(type).shape(100,10000).minValue(-4).maxValue(-2).minValueInclusive(true).maxValueInclusive(true).arg("min", -4.0).arg("max",-2.0)
                    .expectedMean(-3.0).expectedStd(Math.sqrt(1/12.0 * Math.pow(-4.0+2.0, 2)) /*Var: 1/12 * (b-a)^2*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            testCases.add(TestCase.builder().opType("gaussian").dataType(type).shape(new long[0]).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("gaussian").dataType(type).shape(1000).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0)
                    .expectedMean(0.0).expectedStd(1.0).meanMinAbsErrorTolerance(0.05).stdMinAbsErrorTolerance(0.05).build());
            testCases.add(TestCase.builder().opType("gaussian").dataType(type).shape(100,1000).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mean", 2.0).arg("std", 0.5)
                    .expectedMean(2.0).expectedStd(0.5).meanRelativeErrorTolerance(0.01).stdRelativeErrorTolerance(0.01).meanMinAbsErrorTolerance(0.001).build());

            testCases.add(TestCase.builder().opType("binomial").dataType(type).shape(new long[0]).minValue(0).maxValue(5).minValueInclusive(true).maxValueInclusive(true).arg("n", 5).arg("p",0.5).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("binomial").dataType(type).shape(1000).minValue(0).maxValue(10).minValueInclusive(true).maxValueInclusive(true).arg("n", 10).arg("p",0.5)
                    .expectedMean(10*0.5).expectedStd(Math.sqrt(10*0.5*(1-0.5)) /*var = np(1-p)*/).build());
            testCases.add(TestCase.builder().opType("binomial").dataType(type).shape(100,10000).minValue(0).maxValue(20).minValueInclusive(true).maxValueInclusive(true).arg("n", 20).arg("p",0.2)
                    .expectedMean(20*0.2).expectedStd(Math.sqrt(20*0.2*(1-0.2)) /*var = np(1-p)*/).meanRelativeErrorTolerance(0.001).stdRelativeErrorTolerance(0.01).build());

                //truncated normal clips at (mean-2*std, mean+2*std). Mean for equal 2-sided clipping about mean is same as original mean. Variance is difficult to calculate...
                //Assume variance is similar to non-truncated normal (should be a bit less in practice) but use large relative error here
            testCases.add(TestCase.builder().opType("truncated_normal").dataType(type).shape(new long[0]).minValue(-2.0).maxValue(2.0).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("truncated_normal").dataType(type).shape(1000).minValue(-2.0).maxValue(2.0).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0)
                    .expectedMean(0.0).expectedStd(1.0).stdRelativeErrorTolerance(0.2).meanMinAbsErrorTolerance(0.01).build());
            testCases.add(TestCase.builder().opType("truncated_normal").dataType(type).shape(100,10000).minValue(1.0).maxValue(3.0).minValueInclusive(true).maxValueInclusive(true).arg("mean", 2.0).arg("std", 0.5)
                    .expectedMean(2.0).expectedStd(0.5).meanRelativeErrorTolerance(0.001).stdRelativeErrorTolerance(0.2).meanMinAbsErrorTolerance(0.001).build());
        }


        int count = 1;
        for(TestCase tc : testCases){
            log.info("Starting test case: {} of {}", count, testCases.size());
            log.info("{}", tc);

            Op op = getOp(tc);
            Nd4j.getRandom().setSeed(tc.getRngSeed());
            Nd4j.getExecutioner().exec(op);

            //Check for NaNs, Infs, etc
            int countNaN = Nd4j.getExecutioner().execAndReturn(new MatchConditionTransform(op.z(), Nd4j.create(DataType.BOOL, op.z().shape()), Conditions.isNan())).sumNumber().intValue();
            int countInf = Nd4j.getExecutioner().execAndReturn(new MatchConditionTransform(op.z(), Nd4j.create(DataType.BOOL, op.z().shape()), Conditions.isInfinite())).sumNumber().intValue();
            assertEquals("NaN - expected 0 values", 0, countNaN);
            assertEquals("Infinite - expected 0 values", 0, countInf);

            //Check min/max values
            double min = op.z().minNumber().doubleValue();
            if ((tc.isMinValueInclusive() && min < tc.getMinValue()) || (!tc.isMinValueInclusive() && min <= tc.getMinValue())) {
                fail("Minimum value (" + min + ") is less than allowed minimum value (" + tc.getMinValue() + ", inclusive=" + tc.isMinValueInclusive() + "): test case: " + tc);
            }

            double max = op.z().maxNumber().doubleValue();
            if ((tc.isMaxValueInclusive() && max > tc.getMaxValue()) || (!tc.isMaxValueInclusive() && max >= tc.getMaxValue())) {
                fail("Maximum value (" + max + ") is greater than allowed maximum value (" + tc.getMaxValue() + ", inclusive=" + tc.isMaxValueInclusive() + "): test case: " + tc);
            }

            //Check RNG seed repeatability
            Op op2 = getOp(tc);
            Nd4j.getRandom().setSeed(tc.getRngSeed());
            Nd4j.getExecutioner().exec(op);
            INDArray out1 = op.z();
            INDArray out2 = op.z();
            assertEquals(out1, out2);

            //Check mean, stdev
            if(tc.getExpectedMean() != null){
                double mean = op.z().meanNumber().doubleValue();
                double re = relError(tc.getExpectedMean(), mean);
                double ae = Math.abs(tc.getExpectedMean() - mean);
                if(re > tc.getMeanRelativeErrorTolerance() && (tc.getMeanMinAbsErrorTolerance() == null || ae > tc.getMeanMinAbsErrorTolerance())){
                    fail("Relative error for mean (" + re + ") exceeds maximum (" + tc.getMeanRelativeErrorTolerance() +
                            ") - expected mean = " + tc.getExpectedMean() + " vs. observed mean = " + mean + " - test: " + tc);
                }
            }
            if(tc.getExpectedStd() != null){
                double std = op.z().std(true).getDouble(0);
                double re = relError(tc.getExpectedStd(), std);
                double ae = Math.abs(tc.getExpectedStd() - std);
                if(re > tc.getStdRelativeErrorTolerance() && (tc.getStdMinAbsErrorTolerance() == null || ae > tc.getStdMinAbsErrorTolerance())){
                    fail("Relative error for stdev (" + re + ") exceeds maximum (" + tc.getStdRelativeErrorTolerance() +
                            ") - expected stdev = " + tc.getExpectedStd() + " vs. observed stdev = " + std + " - test: " + tc);
                }
            }

            count++;
        }


    }

    private static double minValue(DataType dataType){
       switch (dataType){
           case DOUBLE:
               return -Double.MAX_VALUE;
           case FLOAT:
               return -Float.MAX_VALUE;
           case HALF:
               return 0;
           default:
               throw new RuntimeException("Dtype not supported: " + dataType);
       }
    }

    private static double maxValue(DataType dataType){
        switch (dataType){
            case DOUBLE:
                return Double.MAX_VALUE;
            case FLOAT:
                return Float.MAX_VALUE;
            case HALF:
                return 0;
            default:
                throw new RuntimeException("Dtype not supported: " + dataType);
        }
    }


    private static Op getOp(TestCase tc){

        switch (tc.getOpType()){
            //Legacy (non-custom) RNG ops
            case "bernoulli":
                return new BernoulliDistribution(tc.arr(), (double)tc.prop("prob"));
            case "uniform":
                return new UniformDistribution(tc.arr(), tc.prop("min"), tc.prop("max"));
            case "gaussian":
                return new GaussianDistribution(tc.arr(), (double)tc.prop("mean"), tc.prop("std"));
            case "binomial":
                return new BinomialDistribution(tc.arr(), tc.prop("n"), (double)tc.prop("p"));
            case "truncated_normal":
                return new TruncatedNormalDistribution(tc.arr(), (double)tc.prop("mean"), (double)tc.prop("std") );
            default:
                throw new RuntimeException("Not yet implemented: " + tc.getOpType());
        }
    }

    private static double relError(double x, double y){
        return Math.abs(x-y) / (Math.abs(x) + Math.abs(y));
    }

}
