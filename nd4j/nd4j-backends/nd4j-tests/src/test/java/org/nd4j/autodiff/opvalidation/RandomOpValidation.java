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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.function.Function;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

@Slf4j
public class RandomOpValidation extends BaseOpValidation {

    public RandomOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testRandomOpsSDVarShape(){

        List<String> failed = new ArrayList<>();

        for(double[] shape : Arrays.asList(new double[]{100}, new double[]{10,10}, new double[]{4,5,5})) {

            for (int i = 0; i < 2; i++) {
                INDArray arr;
                if(shape.length == 0){
                    arr = Nd4j.trueScalar(1);
                } else {
                    arr = Nd4j.trueVector(shape);
                }
                
                Nd4j.getRandom().setSeed(12345);
                SameDiff sd = SameDiff.create();
                SDVariable shapeVar = sd.var("shape", arr);

                SDVariable rand;
                Function<INDArray,String> checkFn;
                String name;
                switch (i){
                    case 0:
                        rand = sd.randomUniform(1, 2, shapeVar);
                        checkFn = in -> {
                            double min = in.minNumber().doubleValue();
                            double max = in.maxNumber().doubleValue();
                            double mean = in.meanNumber().doubleValue();
                            if(min >= 1 && max <= 2 && (in.length() == 1 || Math.abs(mean-1.5) < 0.1))
                                return null;
                            return "Failed: min = " + min + ", max = " + max + ", mean = " + mean;
                        };
                        name = "randomUniform";
                        break;
                    case 1:
                        rand = sd.randomNormal(1, 1, shapeVar);
                        checkFn = in -> {
                            double mean = in.meanNumber().doubleValue();
                            double stdev = in.std(true).getDouble(0);
                            if(Math.abs(mean - 1) < 0.1 && Math.abs(stdev - 1) < 0.1)
                                return null;
                            return "Failed: mean = " + mean + ", stdev = " + stdev;
                        };
                        name = "randomNormal";
                    default:
                        throw new RuntimeException();
                }

                SDVariable loss;
                if(shape.length > 0) {
                    loss = rand.std(true);
                } else {
                    loss = rand.mean();
                }

                String msg = name + " - " + Arrays.toString(shape);
                TestCase tc = new TestCase(sd)
                        .testName(msg)
                        .expected(rand, checkFn);

                log.info("TEST: " + msg);

                String err = OpValidation.validate(tc, true);
                if(err != null){
                    failed.add(err);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testUniformRankSimple(){
        OpTestCase tc = new OpTestCase(DynamicCustomOp.builder("randomuniform")
//                .addInputs(Nd4j.trueVector(new double[]{100}))
                .addInputs(Nd4j.create(new double[]{100}))
                .addOutputs(Nd4j.createUninitialized(new long[]{100}))
//                .addFloatingPointArguments(0.0, 1.0)
                .addFloatingPointArguments(1.0, 2.0)
                .build());

        tc.expectedOutput(0, new long[]{100}, in -> {
            double min = in.minNumber().doubleValue();
            double max = in.maxNumber().doubleValue();
            double mean = in.meanNumber().doubleValue();
            if(min >= 0 && max <= 1 && (in.length() == 1 || Math.abs(mean-0.5) < 0.1))
                return null;
            return "Failed: min = " + min + ", max = " + max + ", mean = " + mean;
        });

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

}
