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

package org.eclipse.deeplearning4j.nd4j.linalg.rng;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@NativeTag
@Tag(TagNames.RNG)
@Execution(ExecutionMode.SAME_THREAD)
public class RandomOpExecutionIsolationTest {

    @Test
    public void testFloatingPointExtraArgsBufferUsesRequestedType() {
        UniformDistribution op = new UniformDistribution(Nd4j.createUninitialized(DataType.FLOAT, 4), 0.25, 0.75);
        DataBuffer extraArgs = op.extraArgsDataBuff(DataType.FLOAT);

        assertEquals(DataType.FLOAT, extraArgs.dataType(), "Random op extraArgs buffer must use the requested FLOAT dtype");
        assertEquals(0.25f, extraArgs.getFloat(0), 1.0e-6f, "Lower bound was not stored in the FLOAT extraArgs buffer");
        assertEquals(0.75f, extraArgs.getFloat(1), 1.0e-6f, "Upper bound was not stored in the FLOAT extraArgs buffer");
    }

    @Test
    public void testUniformDistributionExecWritesTargetArray() {
        Nd4j.getRandom().setSeed(12345);

        INDArray z = Nd4j.createUninitialized(DataType.FLOAT, 4096);
        z.assign(Float.NaN);

        Nd4j.getExecutioner().exec(new UniformDistribution(z, 1.0, 2.0), Nd4j.getRandom());

        double min = z.minNumber().doubleValue();
        double max = z.maxNumber().doubleValue();
        double std = z.stdNumber().doubleValue();

        assertFalse(Double.isNaN(min), "Uniform distribution left NaNs in the output array");
        assertTrue(min >= 1.0, "Uniform distribution wrote values below the requested lower bound");
        assertTrue(max <= 2.0, "Uniform distribution wrote values above the requested upper bound");
        assertTrue(std > 0.1, "Uniform distribution output collapsed to a near-constant array");
    }

    @Test
    public void testGaussianDistributionExecWritesTargetArray() {
        Nd4j.getRandom().setSeed(12345);

        INDArray z = Nd4j.createUninitialized(DataType.FLOAT, 4096);
        z.assign(Float.NaN);

        Nd4j.getExecutioner().exec(new GaussianDistribution(z, 0.0, 1.0), Nd4j.getRandom());

        double mean = z.meanNumber().doubleValue();
        double std = z.stdNumber().doubleValue();
        double norm1 = z.norm1Number().doubleValue();

        assertFalse(Double.isNaN(mean), "Gaussian distribution left NaNs in the output array");
        assertTrue(std > 0.3, "Gaussian distribution output collapsed to a near-constant array");
        assertTrue(norm1 > 100.0, "Gaussian distribution output was effectively zero");
    }

    @Test
    public void testNd4jRandAndRandnProduceNonDegenerateOutputs() {
        Nd4j.getRandom().setSeed(12345);
        INDArray uniform = Nd4j.rand(DataType.FLOAT, 4096);

        Nd4j.getRandom().setSeed(12345);
        INDArray gaussian = Nd4j.randn(DataType.FLOAT, 4096);

        assertTrue(uniform.minNumber().doubleValue() >= 0.0, "Nd4j.rand produced values below 0");
        assertTrue(uniform.maxNumber().doubleValue() <= 1.0, "Nd4j.rand produced values above 1");
        assertTrue(uniform.stdNumber().doubleValue() > 0.1, "Nd4j.rand output collapsed to a near-constant array");

        assertTrue(gaussian.stdNumber().doubleValue() > 0.3, "Nd4j.randn output collapsed to a near-constant array");
        assertTrue(gaussian.norm1Number().doubleValue() > 100.0, "Nd4j.randn output was effectively zero");
    }
}
