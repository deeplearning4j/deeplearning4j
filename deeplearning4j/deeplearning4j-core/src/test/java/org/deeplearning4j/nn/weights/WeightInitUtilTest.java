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
package org.deeplearning4j.nn.weights;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Weight Init Util Test")
class WeightInitUtilTest extends BaseDL4JTest {

    protected int fanIn = 3;

    protected int fanOut = 2;

    protected int[] shape = new int[] { fanIn, fanOut };

    protected Distribution dist = Distributions.createDistribution(new GaussianDistribution(0.0, 0.1));

    @BeforeEach
    void doBefore() {
        Nd4j.getRandom().setSeed(123);
    }

    @Test
    @DisplayName("Test Distribution")
    void testDistribution() {
        INDArray params = Nd4j.create(shape, 'f');
        // fan in/out not used
        INDArray weightsActual = WeightInitUtil.initWeights(-1, -1, shape, WeightInit.DISTRIBUTION, dist, params);
        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected = dist.sample(params);
        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    @DisplayName("Test Relu")
    void testRelu() {
        INDArray params = Nd4j.create(shape, 'f');
        INDArray weightsActual = WeightInitUtil.initWeights(fanIn, fanOut, shape, WeightInit.RELU, dist, params);
        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected = Nd4j.randn('f', shape).muli(FastMath.sqrt(2.0 / fanIn));
        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    @DisplayName("Test Sigmoid Uniform")
    void testSigmoidUniform() {
        INDArray params = Nd4j.create(shape, 'f');
        INDArray weightsActual = WeightInitUtil.initWeights(fanIn, fanOut, shape, WeightInit.SIGMOID_UNIFORM, dist, params);
        // expected calculation
        Nd4j.getRandom().setSeed(123);
        double min = -4.0 * Math.sqrt(6.0 / (double) (shape[0] + shape[1]));
        double max = 4.0 * Math.sqrt(6.0 / (double) (shape[0] + shape[1]));
        INDArray weightsExpected = Nd4j.getDistributions().createUniform(min, max).sample(Nd4j.createUninitialized(shape, 'f'));
        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    @DisplayName("Test Uniform")
    void testUniform() {
        INDArray params = Nd4j.create(shape, 'f');
        INDArray weightsActual = WeightInitUtil.initWeights(fanIn, fanOut, shape, WeightInit.UNIFORM, dist, params);
        // expected calculation
        Nd4j.getRandom().setSeed(123);
        double a = 1.0 / Math.sqrt(fanIn);
        INDArray weightsExpected = Nd4j.getDistributions().createUniform(-a, a).sample(Nd4j.create(shape, 'f'));
        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    @DisplayName("Test Xavier")
    void testXavier() {
        Nd4j.getRandom().setSeed(123);
        INDArray params = Nd4j.create(shape, 'f');
        INDArray weightsActual = WeightInitUtil.initWeights(fanIn, fanOut, shape, WeightInit.XAVIER, dist, params);
        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected = Nd4j.randn('f', shape);
        weightsExpected.muli(FastMath.sqrt(2.0 / (fanIn + fanOut)));
        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    @DisplayName("Test Xavier Fan In")
    void testXavierFanIn() {
        INDArray params = Nd4j.create(shape, 'f');
        INDArray weightsActual = WeightInitUtil.initWeights(fanIn, fanOut, shape, WeightInit.XAVIER_FAN_IN, dist, params);
        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected = Nd4j.randn('f', shape);
        weightsExpected.divi(FastMath.sqrt(fanIn));
        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    @DisplayName("Test Xavier Legacy")
    void testXavierLegacy() {
        INDArray params = Nd4j.create(shape, 'f');
        INDArray weightsActual = WeightInitUtil.initWeights(fanIn, fanOut, shape, WeightInit.XAVIER_LEGACY, dist, params);
        // expected calculation
        Nd4j.getRandom().setSeed(123);
        INDArray weightsExpected = Nd4j.randn('f', shape);
        weightsExpected.muli(FastMath.sqrt(1.0 / (fanIn + fanOut)));
        assertEquals(weightsExpected, weightsActual);
    }

    @Test
    @DisplayName("Test Zero")
    void testZero() {
        INDArray params = Nd4j.create(shape, 'f');
        INDArray weightsActual = WeightInitUtil.initWeights(fanIn, fanOut, shape, WeightInit.ZERO, dist, params);
        // expected calculation
        INDArray weightsExpected = Nd4j.create(shape, 'f');
        assertEquals(weightsExpected, weightsActual);
    }
}
