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

package org.nd4j.finitedifferences;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * Simple 2 point finite difference approximation
 * to compute the partial derivatives wrt the 2 given  points
 * based on:
 * https://github.com/apache/commons-math/blob/master/src/main/java/org/apache/commons/math4/analysis/interpolation/BicubicInterpolator.java
 *
 *
 *
 * @author Adam Gibson
 */
public class TwoPointApproximation {


    /**
     * Prepare the boundaries for processing
     * @param bounds the bounds
     * @param x the input in to the approximation
     * @return the lower and upper bounds as an array of ndarrays
     * (in that order) of the same shape as x
     */
    public static INDArray[] prepareBounds(INDArray bounds,INDArray x) {
        return new INDArray[]  {Nd4j.valueArrayOf(x.shape(),bounds.getDouble(0)),
                Nd4j.valueArrayOf(x.shape(),bounds.getDouble(1))};
    }

    /**
     * Adjust final scheme to presence of bounds
     *
     * Returns (in this order):
     * adjusted hypothesis, whether to use onesided as an int mask array
     * @param x the point to estimate the derivative
     * @param h the finite difference steps
     * @param numSteps Number of h steps in 1 direction
     *                 to implement finite difference scheme.
     *
     * @param lowerBound Lower bounds for independent variable variable
     * @param upperBound Upper bounds for independent variable
     * @return
     */
    public static INDArray[] adjustSchemeToBounds(INDArray x,INDArray h,int numSteps,INDArray lowerBound,INDArray upperBound) {
        INDArray oneSided = Nd4j.onesLike(h);
        if(and(lowerBound.eq(Double.NEGATIVE_INFINITY),upperBound.eq(Double.POSITIVE_INFINITY)).sumNumber().doubleValue() > 0) {
            return new INDArray[] {h,oneSided};
        }
        INDArray hTotal = h.mul(numSteps);
        INDArray hAdjusted = h.dup();
        INDArray lowerDist = x.sub(lowerBound);
        INDArray upperBound2 = upperBound.sub(x);

        INDArray central = and(greaterThanOrEqual(lowerDist,hTotal),greaterThanOrEqual(upperBound2,hTotal));
        INDArray forward = and(greaterThanOrEqual(upperBound,lowerDist),not(central));
        hAdjusted.put(forward,min(h.get(forward),upperBound2.get(forward).mul(0.5).divi(numSteps)));
        oneSided.put(forward,Nd4j.scalar(1.0));

        INDArray backward = and(upperBound2.lt(lowerBound),not(central));
        hAdjusted.put(backward,min(h.get(backward),lowerDist.get(backward).mul(0.5).divi(numSteps)));
        oneSided.put(backward,Nd4j.scalar(1.0));

        INDArray minDist = min(upperBound2,lowerDist).divi(numSteps);
        INDArray adjustedCentral = and(not(central),lessThanOrEqual(abs(hAdjusted),minDist));
        hAdjusted.put(adjustedCentral,minDist.get(adjustedCentral));
        oneSided.put(adjustedCentral,Nd4j.scalar(0.0));
        return new INDArray[] {hAdjusted,oneSided};
    }

    /**
     *
     * @param x
     * @return
     */
    public static INDArray computeAbsoluteStep(INDArray x) {
        INDArray relStep = pow(Nd4j.scalar(Nd4j.EPS_THRESHOLD),0.5);
        return computeAbsoluteStep(relStep,x);
    }

    public static double getEpsRelativeTo(INDArray data) {
        if(data.data().dataType() == DataBuffer.Type.FLOAT)
            return 1.1920929e-07;
            return 2.220446049250313e-16;
    }


    /**
     *
     * @param relStep
     * @param x
     * @return
     */
    public static INDArray computeAbsoluteStep(INDArray relStep,INDArray x) {
        if(relStep == null) {
            relStep = pow(Nd4j.scalar(getEpsRelativeTo(x)),0.5);
        }
        INDArray signX0 = x.gte(0).muli(2).subi(1);
        return signX0.mul(relStep).muli(max(abs(x),1.0));
    }

    /**
     *
     * @param f
     * @param x
     * @param relStep
     * @param f0
     * @param bounds
     * @return
     */
    public static INDArray approximateDerivative(Function<INDArray,INDArray> f,
                                                 INDArray x,
                                                 INDArray relStep,INDArray f0,
                                                 INDArray bounds)  {
        if(x.rank() > 2) {
            throw new ND4JIllegalArgumentException("Argument must be a vector or scalar");
        }

        INDArray h = computeAbsoluteStep(relStep,x);
        INDArray[] upperAndLower = prepareBounds(bounds, x);
        INDArray[] boundaries = adjustSchemeToBounds(x,h,1,upperAndLower[0],upperAndLower[1]);
        return denseDifference(f,x,f0,h,boundaries[1]);

    }


    /**
     *
     * @param func
     * @param x0
     * @param f0
     * @param h
     * @param oneSided
     * @return
     */
    public static INDArray denseDifference(Function<INDArray,INDArray> func,
                                           INDArray x0,INDArray f0,
                                           INDArray h,INDArray oneSided) {
        INDArray hVecs = Nd4j.diag(h.reshape(1,h.length()));
        INDArray dx,df,x;
        INDArray jTransposed = Nd4j.create(x0.length(),f0.length());
        for(int i = 0; i < h.length(); i++) {
            INDArray hVecI = hVecs.slice(i);
            x = (x0.add(hVecI));
            dx = x.slice(i).sub(x0.slice(i));
            df = func.apply(x).sub(f0);
            INDArray div = df.div(dx);
            jTransposed.putSlice(i,div);
        }

        if(f0.length() == 1)
            jTransposed = jTransposed.ravel();

            return jTransposed;

    }

}
