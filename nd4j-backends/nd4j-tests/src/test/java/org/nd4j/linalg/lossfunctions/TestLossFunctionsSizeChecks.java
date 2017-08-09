package org.nd4j.linalg.lossfunctions;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * This is intended to ensure that if the incorrect size of data is given
 * to the loss functions, that the report this fact through an appropriate
 * exception.  This functionality used to be performed at the 'deeplearning4j'
 * level, but it was discovered that many loss functions perform mappings
 * involving 'label' sizes different than 'output' sizes.  Such an example
 * would be the Bishop Mixture Density Network.  Hence the testing for
 * loss function size was moved to being the responsibility of the loss function
 * to enforce.
 * 
 * @author Jonathan S. Arney.
 */
public class TestLossFunctionsSizeChecks {
    @Test
    public void testL2() {
        LossFunction[] lossFunctionList = {LossFunction.MSE, LossFunction.L1, LossFunction.EXPLL, LossFunction.XENT,
                        LossFunction.MCXENT, LossFunction.SQUARED_LOSS, LossFunction.RECONSTRUCTION_CROSSENTROPY,
                        LossFunction.NEGATIVELOGLIKELIHOOD, LossFunction.COSINE_PROXIMITY, LossFunction.HINGE,
                        LossFunction.SQUARED_HINGE, LossFunction.KL_DIVERGENCE, LossFunction.MEAN_ABSOLUTE_ERROR,
                        LossFunction.L2, LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
                        LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR, LossFunction.POISSON};

        testLossFunctions(lossFunctionList);
    }

    public void testLossFunctions(LossFunction[] lossFunctions) {
        for (LossFunction loss : lossFunctions) {
            testLossFunctionScoreSizeMismatchCase(loss.getILossFunction());
        }
    }

    /**
     * This method checks that the given loss function will give an assertion
     * if the labels and output vectors are of different sizes.
     * @param loss Loss function to verify.
     */
    public void testLossFunctionScoreSizeMismatchCase(ILossFunction loss) {

        try {
            INDArray labels = Nd4j.create(100, 32);
            INDArray preOutput = Nd4j.create(100, 44);
            double score = loss.computeScore(labels, preOutput, Activation.IDENTITY.getActivationFunction(), null,
                            true);
            Assert.assertFalse(
                            "Loss function " + loss.toString()
                                            + "did not check for size mismatch.  This should fail to compute an activation function because the sizes of the vectors are not equal",
                            true);
        } catch (IllegalArgumentException ex) {
            String exceptionMessage = ex.getMessage();
            Assert.assertTrue(
                            "Loss function exception " + loss.toString()
                                            + " did not indicate size mismatch when vectors of incorrect size were used.",
                            exceptionMessage.contains("does not match"));
        }

        try {
            INDArray labels = Nd4j.create(100, 32);
            INDArray preOutput = Nd4j.create(100, 44);
            INDArray gradient =
                            loss.computeGradient(labels, preOutput, Activation.IDENTITY.getActivationFunction(), null);
            Assert.assertFalse(
                            "Loss function " + loss.toString()
                                            + "did not check for size mismatch.  This should fail to compute an activation function because the sizes of the vectors are not equal",
                            true);
        } catch (IllegalArgumentException ex) {
            String exceptionMessage = ex.getMessage();
            Assert.assertTrue(
                            "Loss function exception " + loss.toString()
                                            + " did not indicate size mismatch when vectors of incorrect size were used.",
                            exceptionMessage.contains("does not match"));
        }

    }
}
