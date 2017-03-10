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

/**
 * Weight initialization scheme
 * <p>
 * <b>DISTRIBUTION</b>: Sample weights from a provided distribution<br>
 * <p>
 * <b>ZERO</b>: Generate weights as zeros<br>
 * <p>
 * <b>SIGMOID_UNIFORM</b>: A version of XAVIER_UNIFORM for sigmoid activation functions. U(-r,r) with r=4*sqrt(6/(fanIn + fanOut))
 * <p>
 * <b>UNIFORM</b>: Uniform U[-a,a] with a=1/sqrt(fanIn). "Commonly used heuristic" as per Glorot and Bengio 2010
 * <p>
 * <b>XAVIER</b>: As per Glorot and Bengio 2010: Gaussian distribution with mean 0, variance 2.0/(fanIn + fanOut)
 * <p>
 * <b>XAVIER_UNIFORM</b>: As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
 * <p>
 * <b>XAVIER_FAN_IN</b>: Similar to Xavier, but 1/fanIn -> Caffe originally used this.
 * <p>
 * <b>XAVIER_LEGACY</b>: Xavier weight init in DL4J up to 0.6.0. XAVIER should be preferred.
 * <p>
 * <b>RELU</b>: He et al. (2015), "Delving Deep into Rectifiers". Normal distribution with variance 2.0/nIn
 * <p>
 * <b>RELU_UNIFORM</b>: He et al. (2015), "Delving Deep into Rectifiers". Uniform distribution U(-s,s) with s = sqrt(6/fanIn)
 * <p>
 *
 * @author Adam Gibson
 */
public enum WeightInit {
    DISTRIBUTION, ZERO, SIGMOID_UNIFORM, UNIFORM, XAVIER, XAVIER_UNIFORM, XAVIER_FAN_IN, XAVIER_LEGACY, RELU, RELU_UNIFORM
}
