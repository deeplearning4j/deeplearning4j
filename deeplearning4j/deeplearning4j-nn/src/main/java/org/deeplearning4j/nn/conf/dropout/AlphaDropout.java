/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.ToString;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldMulOp;
import org.nd4j.linalg.api.ops.random.impl.AlphaDropOut;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * AlphaDropout is a dropout technique proposed by Klaumbauer et al. 2017 - Self-Normalizing Neural Networks
 * <a href="https://arxiv.org/abs/1706.02515">https://arxiv.org/abs/1706.02515</a> <br>
 * <br>
 * This dropout technique was designed specifically for self-normalizing neural networks - i.e., networks using
 * {@link org.nd4j.linalg.activations.impl.ActivationSELU} / {@link org.nd4j.linalg.activations.Activation#SELU}
 * activation function, combined with the N(0,stdev=1/sqrt(fanIn)) "SNN" weight initialization,
 * {@link org.deeplearning4j.nn.weights.WeightInit#NORMAL}<br>
 * <br>
 * In conjuction with the aforementioned activation function and weight initialization, AlphaDropout attempts to keep
 * both the mean and variance of the post-dropout activations to the the same (in expectation) as before alpha
 * dropout was applied.<br>
 * Specifically, AlphaDropout implements a * (x * d + alphaPrime * (1-d)) + b, where d ~ Bernoulli(p), i.e., d \in {0,1}.
 * Where x is the input activations, a, b, alphaPrime are constants determined from the SELU alpha/lambda parameters.
 * Users should use the default alpha/lambda values in virtually all cases.<br>
 * <br>
 * Dropout schedules (i.e., varying probability p as a function of iteration/epoch) are also supported.<br>
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"lastPValue","alphaPrime","a","b", "mask"})
@ToString(exclude = {"lastPValue","alphaPrime","a","b"})
@JsonIgnoreProperties({"lastPValue", "alphaPrime", "a", "b", "mask"})
public class AlphaDropout implements IDropout {

    public static final double DEFAULT_ALPHA =  1.6732632423543772;
    public static final double DEFAULT_LAMBDA = 1.0507009873554804;


    private final double p;
    private final ISchedule pSchedule;
    private final double alpha;
    private final double lambda;

    private transient double lastPValue;
    private double alphaPrime;
    private double a;
    private double b;

    private transient INDArray mask;

    /**
     * @param activationRetainProbability Probability of retaining an activation. See {@link AlphaDropout} javadoc
     */
    public AlphaDropout(double activationRetainProbability){
        this(activationRetainProbability, null, DEFAULT_ALPHA, DEFAULT_LAMBDA);
        if(activationRetainProbability < 0.0){
            throw new IllegalArgumentException("Activation retain probability must be > 0. Got: " + activationRetainProbability);
        }
        if(activationRetainProbability == 0.0){
            throw new IllegalArgumentException("Invalid probability value: Dropout with 0.0 probability of retaining "
                    + "activations is not supported");
        }
    }

    /**
     * @param activationRetainProbabilitySchedule Schedule for the probability of retaining an activation. See
     *  {@link AlphaDropout} javadoc
     */
    public AlphaDropout(@NonNull ISchedule activationRetainProbabilitySchedule){
        this(Double.NaN, activationRetainProbabilitySchedule, DEFAULT_ALPHA, DEFAULT_LAMBDA);
    }

    protected AlphaDropout(@JsonProperty("p")double activationRetainProbability,
                           @JsonProperty("pSchedule") ISchedule activationRetainProbabilitySchedule,
                           @JsonProperty("alpha") double alpha, @JsonProperty("lambda") double lambda ){
        this.p = activationRetainProbability;
        this.pSchedule = activationRetainProbabilitySchedule;
        this.alpha = alpha;
        this.lambda = lambda;

        this.alphaPrime = -lambda * alpha;
        if(activationRetainProbabilitySchedule == null){
            this.lastPValue = p;
            this.a = a(p);
            this.b = b(p);
        }
    }

    @Override
    public INDArray applyDropout(INDArray inputActivations, INDArray output, int iteration, int epoch, LayerWorkspaceMgr workspaceMgr) {
        //https://arxiv.org/pdf/1706.02515.pdf pg6
        // "...we propose “alpha dropout”, that randomly sets inputs to α'"
        // "The affine transformation a(xd + α'(1−d))+b allows to determine parameters a and b such that mean and
        // variance are kept to their values"

        double pValue;
        if(pSchedule != null){
            pValue = pSchedule.valueAt(iteration, epoch);
        } else {
            pValue = p;
        }

        if(pValue != lastPValue){
            a = a(pValue);
            b = b(pValue);
        }
        lastPValue = pValue;

        mask = workspaceMgr.createUninitialized(ArrayType.INPUT, output.shape(), output.ordering());
        Nd4j.getExecutioner().exec(new BernoulliDistribution(mask, pValue));

        //a * (x * d + alphaPrime * (1-d)) + b
        INDArray aPOneMinusD = Transforms.not(mask).muli(alphaPrime);
        Nd4j.getExecutioner().exec(new OldMulOp(inputActivations, mask, output));   //out = x * d
        output.addi(aPOneMinusD).muli(a).addi(b);

        //Nd4j.getExecutioner().exec(new AlphaDropOut(inputActivations, output, p, a, alphaPrime, b));
        return output;
    }

    @Override
    public INDArray backprop(INDArray gradAtOutput, INDArray gradAtInput, int iteration, int epoch) {
        Preconditions.checkState(mask != null, "Cannot perform backprop: Dropout mask array is absent (already cleared?)");
        //dL/dIn = dL/dOut * dOut/dIn
        // dOut/dIn = 0 if dropped (d=0), or a otherwise (d=1)
        mask.muli(a);
        Nd4j.getExecutioner().exec(new OldMulOp(gradAtOutput, mask, gradAtInput));
        mask = null;
        return gradAtInput;
    }

    @Override
    public void clear() {
        mask = null;
    }

    @Override
    public AlphaDropout clone() {
        return new AlphaDropout(p, pSchedule == null ? null : pSchedule.clone(), alpha, lambda);
    }

    public double a(double p){
        return 1.0 / Math.sqrt(p + alphaPrime*alphaPrime * p * (1-p));
    }

    public double b(double p){
        return -a(p) * (1-p)*alphaPrime;
    }
}
