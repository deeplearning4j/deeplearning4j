package org.deeplearning4j.lstm;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.layers.dropout.CudnnDropoutHelper;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ValidateCudnnDropout extends BaseDL4JTest {

    @Test
    public void testCudnnDropoutSimple() {
        for (int[] shape : new int[][]{{10, 10}, {5, 2, 5, 2}}) {

            Nd4j.getRandom().setSeed(12345);
            INDArray in = Nd4j.ones(shape);
            double pRetain = 0.25;
            double valueIfKept = 1.0 / pRetain;

            CudnnDropoutHelper d = new CudnnDropoutHelper();

            INDArray out = Nd4j.createUninitialized(shape);
            d.applyDropout(in, out, pRetain);

            int countZero = Nd4j.getExecutioner().execAndReturn(new MatchCondition(out, Conditions.equals(0.0))).z().getInt(0);
            int countNonDropped = Nd4j.getExecutioner().execAndReturn(new MatchCondition(out, Conditions.equals(valueIfKept))).z().getInt(0);
//            System.out.println(countZero);
//            System.out.println(countNonDropped);

            assertTrue(String.valueOf(countZero), countZero >= 5 && countZero <= 90);
            assertTrue(String.valueOf(countNonDropped), countNonDropped >= 5 && countNonDropped <= 95);
            assertEquals(100, countZero + countNonDropped);

            //Test repeatability:
            for (int i = 0; i < 10; i++) {
                Nd4j.getRandom().setSeed(12345);

                INDArray outNew = Nd4j.createUninitialized(shape);
                d.applyDropout(in, outNew, pRetain);

                assertEquals(out, outNew);
            }

            //Test backprop:
            INDArray gradAtOut = Nd4j.ones(shape);
            INDArray gradAtInput = Nd4j.createUninitialized(shape);
            d.backprop(gradAtOut, gradAtInput);

            //If dropped: expect 0. Otherwise: expect 1/pRetain, i.e., output for 1s input
            assertEquals(out, gradAtInput);
        }
    }

}
