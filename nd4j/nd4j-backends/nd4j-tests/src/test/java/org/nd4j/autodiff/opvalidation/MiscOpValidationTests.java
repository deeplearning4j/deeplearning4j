package org.nd4j.autodiff.opvalidation;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNull;

public class MiscOpValidationTests {

    @Test
    public void testXwPlusB() {
        Nd4j.getRandom().setSeed(12345);

        for(boolean rank1Bias : new boolean[]{false, true}) {

            SameDiff sameDiff = SameDiff.create();
            INDArray input = Nd4j.rand(new long[]{2, 3});
            INDArray weights = Nd4j.rand(new long[]{3, 4});
            INDArray b = Nd4j.rand( rank1Bias ? new long[]{4} : new long[]{1, 4});

            SDVariable sdInput = sameDiff.var("input", input);
            SDVariable sdWeights = sameDiff.var("weights", weights);
            SDVariable sdBias = sameDiff.var("bias", b);

            SDVariable res = sameDiff.xwPlusB(sdInput, sdWeights, sdBias);
            SDVariable loss = sameDiff.standardDeviation(res, true);

            INDArray exp = input.mmul(weights).addiRowVector(b);

            TestCase tc = new TestCase(sameDiff)
                    .gradientCheck(true)
                    .expectedOutput(res.getVarName(), exp);


            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

}
