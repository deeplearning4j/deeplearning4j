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
package org.nd4j.linalg.learning;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Max;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

public class UpdaterJavaCode {

    private UpdaterJavaCode(){ }

    public static void applyAdaDeltaUpdater(INDArray gradient, INDArray msg, INDArray msdx, double rho, double epsilon){

        //Line 4 of Algorithm 1: https://arxiv.org/pdf/1212.5701v1.pdf
        //E[g^2]_t = rho * E[g^2]_{t-1} + (1-rho)*g^2_t
        msg.muli(rho).addi(gradient.mul(gradient).muli(1 - rho));

        //Calculate update:
        //dX = - g * RMS[delta x]_{t-1} / RMS[g]_t
        //Note: negative is applied in the DL4J step function: params -= update rather than params += update
        INDArray rmsdx_t1 = Transforms.sqrt(msdx.add(epsilon), false);
        INDArray rmsg_t = Transforms.sqrt(msg.add(epsilon), false);
        INDArray update = gradient.muli(rmsdx_t1.divi(rmsg_t));

        //Accumulate gradients: E[delta x^2]_t = rho * E[delta x^2]_{t-1} + (1-rho)* (delta x_t)^2
        msdx.muli(rho).addi(update.mul(update).muli(1 - rho));
    }

    public static void applyAdaGradUpdater(INDArray gradient, INDArray state, double learningRate, double epsilon){
        state.addi(gradient.mul(gradient));

        INDArray sqrtHistory = sqrt(state.dup('c'), false).addi(epsilon);
        // lr * gradient / (sqrt(sumSquaredGradients) + epsilon)
        gradient.muli(sqrtHistory.rdivi(learningRate));
    }


    public static void applyAdamUpdater(INDArray gradient, INDArray m, INDArray v, double learningRate, double beta1, double beta2,
                                                         double epsilon, int iteration){

        INDArray oneMinusBeta1Grad = gradient.mul(1.0 - beta1);
        m.muli(beta1).addi(oneMinusBeta1Grad);

        INDArray oneMinusBeta2GradSquared = gradient.mul(gradient).muli(1 - beta2);
        v.muli(beta2).addi(oneMinusBeta2GradSquared);

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);

        double alphat = learningRate * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0)
            alphat = epsilon;
        INDArray sqrtV = Transforms.sqrt(v.dup('c'), false).addi(epsilon);

        gradient.assign(m).muli(alphat).divi(sqrtV);
    }

    public static void applyAdaMaxUpdater(INDArray gradient, INDArray m, INDArray v, double learningRate, double beta1, double beta2,
                                        double epsilon, int iteration){

        //m = B_1 * m + (1-B_1)*grad
        m.muli(beta1).addi(gradient.mul(1 - beta1));

        //u = max(B_2 * u, |grad|)
        v.muli(beta2);
        Transforms.abs(gradient, false); //In-place should be OK here, original gradient values aren't used again later
        Nd4j.getExecutioner().exec(new Max(v, gradient, v));

        double beta1t = FastMath.pow(beta1, iteration + 1);

        double alphat = learningRate / (1.0 - beta1t);
        if (Double.isNaN(alphat) || Double.isInfinite(alphat) || alphat == 0.0) {
            alphat = epsilon;
        }

        v.addi(1e-32); // prevent NaNs in params
        gradient.assign(m).muli(alphat).divi(v);
    }

    public static void applyAmsGradUpdater(INDArray gradient, INDArray m, INDArray v, INDArray vHat, double learningRate, double beta1, double beta2,
                                           double epsilon, int iteration){
        //m_t = b_1 * m_{t-1} + (1-b_1) * g_t       eq 1 pg 3
        INDArray oneMinusBeta1Grad = gradient.mul(1.0 - beta1);
        m.muli(beta1).addi(oneMinusBeta1Grad);

        //v_t = b_2 * v_{t-1} + (1-b_2) * (g_t)^2   eq 1 pg 3
        INDArray oneMinusBeta2GradSquared = gradient.mul(gradient).muli(1 - beta2);
        v.muli(beta2).addi(oneMinusBeta2GradSquared);

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);

        //vHat_t = max(vHat_{t-1}, v_t)
        Transforms.max(vHat, v, false);

        double alphat = learningRate * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0)
            alphat = epsilon;

        //gradient array contains: sqrt(vHat) + eps
        Nd4j.getExecutioner().exec(new Sqrt(vHat, gradient)).addi(epsilon);

        //gradient = alphat * m_t / (sqrt(vHat) + eps)
        gradient.rdivi(m).muli(alphat);
    }

    public static void applyNadamUpdater(INDArray gradient, INDArray m, INDArray v, double learningRate, double beta1, double beta2,
                                        double epsilon, int iteration){

        INDArray oneMinusBeta1Grad = gradient.mul(1.0 - beta1);
        m.muli(beta1).addi(oneMinusBeta1Grad);

        INDArray oneMinusBeta2GradSquared = gradient.mul(gradient).muli(1.0 - beta2);
        v.muli(beta2).addi(oneMinusBeta2GradSquared);

        double beta1t = FastMath.pow(beta1, iteration + 1);

        INDArray biasCorrectedEstimateOfMomentum = m.mul(beta1).divi(1.0 - beta1t);
        INDArray secondTerm = oneMinusBeta1Grad.divi(1 - beta1t);

        INDArray alphat = biasCorrectedEstimateOfMomentum.add(secondTerm).muli(learningRate);

        INDArray sqrtV = Transforms.sqrt(v.dup('c'), false).addi(epsilon);

        gradient.assign(alphat).divi(sqrtV);
    }

    public static void applyNesterovsUpdater(INDArray gradient, INDArray v, double lr, double momentum){
        //reference https://cs231n.github.io/neural-networks-3/#sgd 2nd equation
        //DL4J default is negative step function thus we flipped the signs:
        // x += mu * v_prev + (-1 - mu) * v
        //i.e., we do params -= updatedGradient, not params += updatedGradient

        //v = mu * v - lr * gradient
        INDArray vPrev = v.dup('c');
        v.muli(momentum).subi(gradient.dup('c').muli(lr)); //Modify state array in-place

        /*
        Next line is equivalent to:
        INDArray ret = vPrev.muli(momentum).addi(v.mul(-momentum - 1));
        gradient.assign(ret);
        */
        Nd4j.getExecutioner().exec(new AddOp(vPrev.muli(momentum), v.mul(-momentum - 1), gradient));
    }

    public static void applyRmsProp(INDArray gradient, INDArray lastGradient, double learningRate, double rmsDecay, double epsilon){
        lastGradient.muli(rmsDecay).addi(gradient.mul(gradient).muli(1 - rmsDecay));
        // lr * gradient / (sqrt(cache) + 1e-8)
        gradient.muli(learningRate).divi(Transforms.sqrt(lastGradient.dup('c'), false).addi(epsilon));
    }

    public static void applySgd(INDArray gradient, double lr){
        gradient.muli(lr);
    }
}
