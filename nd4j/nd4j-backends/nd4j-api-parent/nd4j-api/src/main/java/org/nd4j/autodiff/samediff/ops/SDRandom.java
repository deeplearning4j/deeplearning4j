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

package org.nd4j.autodiff.samediff.ops;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;

public class SDRandom extends SDOps {
  public SDRandom(SameDiff sameDiff) {
    super(sameDiff);
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
  public SDVariable bernoulli(double p, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return new org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution(sd,p, datatype, shape).outputVariable();
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Bernoulli distribution,<br>
   * with the specified probability. Array values will have value 1 with probability P and value 0 with probability<br>
   * 1-P.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param p Probability of value 1
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public SDVariable bernoulli(String name, double p, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution(sd,p, datatype, shape).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable binomial(int nTrials, double p, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return new org.nd4j.linalg.api.ops.random.impl.BinomialDistribution(sd,nTrials, p, datatype, shape).outputVariable();
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Binomial distribution,<br>
   * with the specified number of trials and probability.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param nTrials Number of trials parameter for the binomial distribution
   * @param p Probability of success for each trial
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public SDVariable binomial(String name, int nTrials, double p, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.BinomialDistribution(sd,nTrials, p, datatype, shape).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable exponential(double lambda, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    Preconditions.checkArgument(lambda > 0, "Must be positive");
    return new org.nd4j.linalg.api.ops.random.custom.RandomExponential(sd,lambda, datatype, shape).outputVariable();
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a exponential distribution:<br>
   * P(x) = lambda * exp(-lambda * x)<br>
   *
   * Inputs must satisfy the following constraints: <br>
   * Must be positive: lambda > 0<br>
   *
   * @param name name May be null. Name for the output variable
   * @param lambda lambda parameter
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public SDVariable exponential(String name, double lambda, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    Preconditions.checkArgument(lambda > 0, "Must be positive");
    SDVariable out =  new org.nd4j.linalg.api.ops.random.custom.RandomExponential(sd,lambda, datatype, shape).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable logNormal(double mean, double stddev, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return new org.nd4j.linalg.api.ops.random.impl.LogNormalDistribution(sd,mean, stddev, datatype, shape).outputVariable();
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Log Normal distribution,<br>
   * i.e., {@code log(x) ~ N(mean, stdev)}<br>
   *
   * @param name name May be null. Name for the output variable
   * @param mean Mean value for the random array
   * @param stddev Standard deviation for the random array
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public SDVariable logNormal(String name, double mean, double stddev, DataType datatype,
      long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.LogNormalDistribution(sd,mean, stddev, datatype, shape).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable normal(double mean, double stddev, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return new org.nd4j.linalg.api.ops.random.impl.GaussianDistribution(sd,mean, stddev, datatype, shape).outputVariable();
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Gaussian (normal) distribution,<br>
   * N(mean, stdev)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param mean Mean value for the random array
   * @param stddev Standard deviation for the random array
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public SDVariable normal(String name, double mean, double stddev, DataType datatype,
      long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.GaussianDistribution(sd,mean, stddev, datatype, shape).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable normalTruncated(double mean, double stddev, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return new org.nd4j.linalg.api.ops.random.impl.TruncatedNormalDistribution(sd,mean, stddev, datatype, shape).outputVariable();
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a Gaussian (normal) distribution,<br>
   * N(mean, stdev). However, any values more than 1 standard deviation from the mean are dropped and re-sampled<br>
   *
   * @param name name May be null. Name for the output variable
   * @param mean Mean value for the random array
   * @param stddev Standard deviation for the random array
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public SDVariable normalTruncated(String name, double mean, double stddev, DataType datatype,
      long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.TruncatedNormalDistribution(sd,mean, stddev, datatype, shape).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable uniform(double min, double max, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    return new org.nd4j.linalg.api.ops.random.impl.UniformDistribution(sd,min, max, datatype, shape).outputVariable();
  }

  /**
   * Generate a new random INDArray, where values are randomly sampled according to a uniform distribution,<br>
   * U(min,max)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param min Minimum value
   * @param max Maximum value.
   * @param datatype Data type of the output variable
   * @param shape Shape of the new random INDArray, as a 1D array (Size: AtLeast(min=0))
   * @return output Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution (NUMERIC type)
   */
  public SDVariable uniform(String name, double min, double max, DataType datatype, long... shape) {
    Preconditions.checkArgument(shape.length >= 0, "shape has incorrect size/length. Expected: shape.length >= 0, got %s", shape.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.UniformDistribution(sd,min, max, datatype, shape).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}
