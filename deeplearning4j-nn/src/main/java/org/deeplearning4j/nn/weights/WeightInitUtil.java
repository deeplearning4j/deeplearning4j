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
        //Note: using f order here as params get flattened to f order

        INDArray ret;
        switch (initScheme) {
            case DISTRIBUTION:
                ret = dist.sample(shape);
                break;
            case RELU:
                ret = Nd4j.randn(order, shape).muli(FastMath.sqrt(2.0 / fanIn)); //N(0, 2/nIn)
                break;
            case RELU_UNIFORM:
                double u = Math.sqrt(6.0 / fanIn);
                ret = Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-u, u)); //U(-sqrt(6/fanIn), sqrt(6/fanIn)
                break;
            case SIGMOID_UNIFORM:
                double r = 4.0 * Math.sqrt(6.0 / (fanIn + fanOut));
                ret = Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-r, r));
                break;
            case UNIFORM:
                double a = 1.0 / Math.sqrt(fanIn);
                ret = Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-a, a));
                break;
            case XAVIER:
                ret = Nd4j.randn(order, shape).muli(FastMath.sqrt(2.0 / (fanIn + fanOut)));
                break;
            case XAVIER_UNIFORM:
                //As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
                //Eq 16: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
                double s = Math.sqrt(6.0) / Math.sqrt(fanIn + fanOut);
                ret = Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-s, s));
                break;
            case XAVIER_FAN_IN:
                ret = Nd4j.randn(order, shape).divi(FastMath.sqrt(fanIn));
                break;
            case XAVIER_LEGACY:
                ret = Nd4j.randn(order, shape).divi(FastMath.sqrt(shape[0] + shape[1]));
                break;
            case ZERO:
                ret = Nd4j.create(shape, order);
                break;

            default:
                throw new IllegalStateException("Illegal weight init value: " + initScheme);
        }

        INDArray flat = Nd4j.toFlattened(order, ret);
        if (flat.length() != paramView.length())
            throw new RuntimeException("ParamView length does not match initialized weights length (view length: "
                            + paramView.length() + ", view shape: " + Arrays.toString(paramView.shape())
                            + "; flattened length: " + flat.length());

        paramView.assign(flat);

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
