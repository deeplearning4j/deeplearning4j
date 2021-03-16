/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.lossfunctions;

import org.junit.Assert;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class TestLossFunctionsSizeChecks extends BaseNd4jTestWithBackends {


    @Override
    public char ordering(){
        return 'c';
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testL2(Nd4jBackend backend) {
        LossFunction[] lossFunctionList = {LossFunction.MSE, LossFunction.L1, LossFunction.XENT,
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
                    exceptionMessage.contains("shapes"));
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
                    exceptionMessage.contains("shapes"));
        }

    }
}
