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

package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.common.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Getter
@EqualsAndHashCode
public class LossFMeasure implements ILossFunction {

    public static final double DEFAULT_BETA = 1.0;

    private final double beta;

    public LossFMeasure() {
        this(DEFAULT_BETA);
    }

    public LossFMeasure(@JsonProperty("beta") double beta) {
        if (beta <= 0) {
            throw new UnsupportedOperationException("Invalid value: beta must be > 0. Got: " + beta);
        }
        this.beta = beta;
    }


    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {
        labels = labels.castTo(preOutput.dataType());   //No-op if already correct dtype
        double[] d = computeScoreNumDenom(labels, preOutput, activationFn, mask, average);
        double numerator = d[0];
        double denominator = d[1];

        if (numerator == 0.0 && denominator == 0.0) {
            return 0.0;
        }

        return 1.0 - numerator / denominator;
    }

    private double[] computeScoreNumDenom(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        long n = labels.size(1);
        if (n != 1 && n != 2) {
            throw new UnsupportedOperationException(
                            "For binary classification: expect output size of 1 or 2. Got: " + n);
        }

        //First: determine positives and negatives
        INDArray isPositiveLabel;
        INDArray isNegativeLabel;
        INDArray pClass0;
        INDArray pClass1;
        if (n == 1) {
            isPositiveLabel = labels;
            isNegativeLabel = isPositiveLabel.rsub(1.0);
            pClass0 = output.rsub(1.0);
            pClass1 = output;
        } else {
            isPositiveLabel = labels.getColumn(1);
            isNegativeLabel = labels.getColumn(0);
            pClass0 = output.getColumn(0);
            pClass1 = output.getColumn(1);
        }

        if (mask != null) {
            isPositiveLabel = isPositiveLabel.mulColumnVector(mask);
            isNegativeLabel = isNegativeLabel.mulColumnVector(mask);
        }

        double tp = isPositiveLabel.mul(pClass1).sumNumber().doubleValue();
        double fp = isNegativeLabel.mul(pClass1).sumNumber().doubleValue();
        double fn = isPositiveLabel.mul(pClass0).sumNumber().doubleValue();

        double numerator = (1.0 + beta * beta) * tp;
        double denominator = (1.0 + beta * beta) * tp + beta * beta * fn + fp;

        return new double[] {numerator, denominator};
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        throw new UnsupportedOperationException("Cannot compute score array for FMeasure loss function: loss is only "
                        + "defined for minibatches");
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        labels = labels.castTo(preOutput.dataType());   //No-op if already correct dtype
        double[] d = computeScoreNumDenom(labels, preOutput, activationFn, mask, false);
        double numerator = d[0];
        double denominator = d[1];

        if (numerator == 0.0 && denominator == 0.0) {
            //Zero score -> zero gradient
            return Nd4j.create(preOutput.shape());
        }

        double secondTerm = numerator / (denominator * denominator);

        INDArray dLdOut;
        if (labels.size(1) == 1) {
            //Single binary output case
            dLdOut = labels.mul(1 + beta * beta).divi(denominator).subi(secondTerm);
        } else {
            //Softmax case: the getColumn(1) here is to account for the fact that we're using prob(class1)
            // only in the score function; column(1) is equivalent to output for the single output case
            dLdOut = Nd4j.create(labels.shape());
            dLdOut.getColumn(1).assign(labels.getColumn(1).mul(1 + beta * beta).divi(denominator).subi(secondTerm));
        }

        //Negate relative to description in paper, as we want to *minimize* 1.0-fMeasure, which is equivalent to
        // maximizing fMeasure
        dLdOut.negi();

        INDArray dLdPreOut = activationFn.backprop(preOutput, dLdOut).getFirst();

        if (mask != null) {
            dLdPreOut.muliColumnVector(mask);
        }

        return dLdPreOut;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                    INDArray mask, boolean average) {
        //TODO optimize
        return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
                        computeGradient(labels, preOutput, activationFn, mask));
    }

    /**
     * The opName of this function
     *
     * @return
     */
    @Override
    public String name() {
        return "floss";
    }

    @Override
    public String toString() {
        return "LossFMeasure(beta=" + beta + ")";
    }
}
