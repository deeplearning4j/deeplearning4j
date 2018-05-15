/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.weights;


import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.OrthogonalDistribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;


/**
 * Weight initialization utility
 *
 * @author Adam Gibson
 */
public class WeightInitUtil {

    /**
     * Default order for the arrays created by WeightInitUtil.
     */
    public static final char DEFAULT_WEIGHT_INIT_ORDER = 'f';

    private WeightInitUtil() {}

    public static INDArray initWeights(int[] shape, float min, float max) {
        return Nd4j.rand(shape, min, max, Nd4j.getRandom());
    }


    /**
     * Initializes a matrix with the given weight initialization scheme.
     * Note: Defaults to fortran ('f') order arrays for the weights. Use {@link #initWeights(int[], WeightInit, Distribution, char, INDArray)}
     * to control this
     *
     * @param shape      the shape of the matrix
     * @param initScheme the scheme to use
     * @return a matrix of the specified dimensions with the specified
     * distribution based on the initialization scheme
     */
    public static INDArray initWeights(double fanIn, double fanOut, int[] shape, WeightInit initScheme,
                    Distribution dist, INDArray paramView) {
        return initWeights(fanIn, fanOut, shape, initScheme, dist, DEFAULT_WEIGHT_INIT_ORDER, paramView);
    }

    public static INDArray initWeights(double fanIn, double fanOut, int[] shape, WeightInit initScheme,
                    Distribution dist, char order, INDArray paramView) {
        switch (initScheme) {
            case DISTRIBUTION:
                if (dist instanceof OrthogonalDistribution) {
                    dist.sample(paramView.reshape(order, shape));
                } else {
                    dist.sample(paramView);
                }
                break;
            case RELU:
                Nd4j.randn(paramView).muli(FastMath.sqrt(2.0 / fanIn)); //N(0, 2/nIn)
                break;
            case RELU_UNIFORM:
                double u = Math.sqrt(6.0 / fanIn);
                Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-u, u)); //U(-sqrt(6/fanIn), sqrt(6/fanIn)
                break;
            case SIGMOID_UNIFORM:
                double r = 4.0 * Math.sqrt(6.0 / (fanIn + fanOut));
                Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-r, r));
                break;
            case UNIFORM:
                double a = 1.0 / Math.sqrt(fanIn);
                Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-a, a));
                break;
            case LECUN_UNIFORM:
                double b = 3.0 / Math.sqrt(fanIn);
                Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-b, b));
                break;
            case XAVIER:
                Nd4j.randn(paramView).muli(FastMath.sqrt(2.0 / (fanIn + fanOut)));
                break;
            case XAVIER_UNIFORM:
                //As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
                //Eq 16: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
                double s = Math.sqrt(6.0) / Math.sqrt(fanIn + fanOut);
                Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-s, s));
                break;
            case LECUN_NORMAL:  //Fall through: these 3 are equivalent
            case NORMAL:
            case XAVIER_FAN_IN:
                Nd4j.randn(paramView).divi(FastMath.sqrt(fanIn));
                break;
            case XAVIER_LEGACY:
                Nd4j.randn(paramView).divi(FastMath.sqrt(shape[0] + shape[1]));
                break;
            case ZERO:
                paramView.assign(0.0);
                break;
            case ONES:
                paramView.assign(1.0);
                break;
            case IDENTITY:
                if(shape.length != 2 || shape[0] != shape[1]){
                    throw new IllegalStateException("Cannot use IDENTITY init with parameters of shape "
                            + Arrays.toString(shape) + ": weights must be a square matrix for identity");
                }
                INDArray ret;
                if(order == Nd4j.order()){
                    ret = Nd4j.eye(shape[0]);
                } else {
                    ret = Nd4j.createUninitialized(shape, order).assign(Nd4j.eye(shape[0]));
                }
                INDArray flat = Nd4j.toFlattened(order, ret);
                paramView.assign(flat);
                break;
            case VAR_SCALING_NORMAL_FAN_IN:
                // TODO: needs to be truncated normal to match keras.
                Nd4j.randn(paramView).divi(FastMath.sqrt(fanIn));
                break;
            case VAR_SCALING_NORMAL_FAN_OUT:
                Nd4j.randn(paramView).divi(FastMath.sqrt(fanOut));
                break;
            case VAR_SCALING_NORMAL_FAN_AVG:
                Nd4j.randn(paramView).divi(FastMath.sqrt((fanIn + fanOut) / 2));
                break;
            case VAR_SCALING_UNIFORM_FAN_IN:
                double scalingFanIn = 3.0 / Math.sqrt(fanIn);
                Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-scalingFanIn, scalingFanIn));
                break;
            case VAR_SCALING_UNIFORM_FAN_OUT:
                double scalingFanOut = 3.0 / Math.sqrt(fanOut);
                Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-scalingFanOut, scalingFanOut));
                break;
            case VAR_SCALING_UNIFORM_FAN_AVG:
                double scalingFanAvg = 3.0 / Math.sqrt((fanIn + fanOut) / 2);
                Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-scalingFanAvg, scalingFanAvg));
                break;
            default:
                throw new IllegalStateException("Illegal weight init value: " + initScheme);
        }

        return paramView.reshape(order, shape);
    }


    /**
     * Reshape the parameters view, without modifying the paramsView array values.
     *
     * @param shape      Shape to reshape
     * @param paramsView Parameters array view
     */
    public static INDArray reshapeWeights(int[] shape, INDArray paramsView) {
        return reshapeWeights(shape, paramsView, DEFAULT_WEIGHT_INIT_ORDER);
    }

    /**
     * Reshape the parameters view, without modifying the paramsView array values.
     *
     * @param shape           Shape to reshape
     * @param paramsView      Parameters array view
     * @param flatteningOrder Order in which parameters are flattened/reshaped
     */
    public static INDArray reshapeWeights(int[] shape, INDArray paramsView, char flatteningOrder) {
        return paramsView.reshape(flatteningOrder, shape);
    }


}
