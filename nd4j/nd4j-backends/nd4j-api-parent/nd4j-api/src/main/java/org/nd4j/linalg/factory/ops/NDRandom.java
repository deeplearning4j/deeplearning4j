/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NDRandom {
  public NDRandom() {
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Bernoulli distribution,<br>
   * with the specified probability. Array values will have value 1 with probability P and value 0 with probability<br>
   * 1-P.<br>
   *
   * @param p Probability of value 1
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public INDArray bernoulli(double p, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution(p, datatype, shape));
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Binomial distribution,<br>
   * with the specified number of trials and probability.<br>
   *
   * @param nTrials Number of trials parameter for the binomial distribution
   * @param p Probability of success for each trial
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public INDArray binomial(int nTrials, double p, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.BinomialDistribution(nTrials, p, datatype, shape));
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a exponential distribution:<br>
   * P(x) = lambda * exp(-lambda * x)<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be positive: lambda > 0<br>
   *
   * @param lambda lambda parameter
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public INDArray exponential(double lambda, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    Preconditions.checkArgument(lambda > 0, "Must be positive");
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.custom.RandomExponential(lambda, datatype, shape))[0];
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Log Normal distribution,<br>
   * i.e., {@code log(x) ~ N(mean, stdev)}<br>
   *
   * @param mean Mean value for the random array
   * @param stddev Standard deviation for the random array
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public INDArray logNormal(double mean, double stddev, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.LogNormalDistribution(mean, stddev, datatype, shape));
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Gaussian (normal) distribution,<br>
   * N(mean, stdev)<br>
   *
   * @param mean Mean value for the random array
   * @param stddev Standard deviation for the random array
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public INDArray normal(double mean, double stddev, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.GaussianDistribution(mean, stddev, datatype, shape));
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Gaussian (normal) distribution,<br>
   * N(mean, stdev). However, any values more than 1 standard deviation from the mean are dropped and re-sampled<br>
   *
   * @param mean Mean value for the random array
   * @param stddev Standard deviation for the random array
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public INDArray normalTruncated(double mean, double stddev, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.TruncatedNormalDistribution(mean, stddev, datatype, shape));
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a uniform distribution,<br>
   * U(min,max)<br>
   *
   * @param min Minimum value
   * @param max Maximum value.
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public INDArray uniform(double min, double max, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.UniformDistribution(min, max, datatype, shape));
  }
}
