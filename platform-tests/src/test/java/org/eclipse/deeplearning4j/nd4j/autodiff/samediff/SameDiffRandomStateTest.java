/*
 *  ******************************************************************************
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
 *  *  SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.random.custom.DistributionUniform;
import org.nd4j.linalg.api.ops.random.custom.RandomExponential;
import org.nd4j.linalg.api.ops.random.custom.RandomGamma;
import org.nd4j.linalg.api.ops.random.custom.RandomNormal;
import org.nd4j.linalg.api.ops.random.custom.RandomPoisson;
import org.nd4j.linalg.api.ops.random.impl.DropOutBp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SameDiff's random ops draw from the thread's {@code Nd4j.getRandom()}: seeding it reproduces a
 * graph's samples and every execution advances it. Before, each op context brought its own
 * generator seeded from the clock, so {@code setSeed} had no effect, and unseeded dropout used one
 * fixed mask forever. Also covers the samplers behind the ops and the ops' seed arguments.
 */
@Slf4j
public class SameDiffRandomStateTest {

    /** Builds one random op on constant inputs. */
    interface RandomGraph {
        SDVariable build(SameDiff sd);
    }

    static Stream<Arguments> randomGraphs() {
        return Stream.of(
                Arguments.of("random_exponential",
                        (RandomGraph) sd -> sd.random().exponential(2.0, DataType.FLOAT, 64, 64)),
                Arguments.of("random_normal", (RandomGraph) sd -> new RandomNormal(sd,
                        sd.constant(Nd4j.createFromArray(64L, 64L)), 0.0, 1.0).outputVariable()),
                Arguments.of("randomuniform", (RandomGraph) sd -> new DistributionUniform(sd,
                        sd.constant(Nd4j.createFromArray(64L, 64L)), 0.0, 1.0).outputVariable()),
                Arguments.of("random_gamma", (RandomGraph) sd -> new RandomGamma(sd,
                        sd.constant(Nd4j.createFromArray(256L)),
                        sd.constant(Nd4j.createFromArray(0.5f, 2.0f, 9.0f)), null).outputVariable()),
                Arguments.of("random_poisson", (RandomGraph) sd -> new RandomPoisson(sd,
                        sd.constant(Nd4j.createFromArray(256L)),
                        sd.constant(Nd4j.createFromArray(0.5f, 4.0f, 30.0f, 200.0f))).outputVariable()),
                Arguments.of("dropout", (RandomGraph) sd ->
                        sd.nn().dropout(sd.constant(Nd4j.ones(DataType.FLOAT, 64, 64)), false, 0.5)),
                Arguments.of("random_crop", (RandomGraph) sd -> sd.image().randomCrop(
                        sd.constant(Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, 100 * 100).reshape(100, 100)),
                        sd.constant(Nd4j.createFromArray(8, 8)))));
    }

    /** Every random graph, executed through a DSP plan and through the standard session. */
    static Stream<Arguments> randomGraphsWithAndWithoutDsp() {
        return randomGraphs().flatMap(arguments -> Stream.of(true, false).map(dsp ->
                Arguments.of(arguments.get()[0], arguments.get()[1], dsp)));
    }

    @ParameterizedTest(name = "{0} dsp={2}")
    @MethodSource("randomGraphsWithAndWithoutDsp")
    void theThreadGeneratorReproducesAndAdvancesARandomOp(String op, RandomGraph graph, boolean useDsp) {
        boolean dsp = InferenceSession.isDynamicShapePlanEnabled();
        InferenceSession.setDynamicShapePlanEnabled(useDsp);
        try (SameDiff sd = SameDiff.create()) {
            SDVariable out = graph.build(sd);
            Nd4j.getRandom().setSeed(123);
            INDArray first = out.eval().dup();
            INDArray second = out.eval().dup();
            Nd4j.getRandom().setSeed(123);
            INDArray again = out.eval().dup();
            assertEquals(first, again, op + " dsp=" + useDsp + ": the same seed must reproduce the samples");
            assertNotEquals(first, second, op + " dsp=" + useDsp + ": an execution must advance the generator");
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(dsp);
        }
    }

    /**
     * A plan captures and replays its deterministic slots after a few executions; a random slot
     * stays live, or every replay would repeat the draws of the capture.
     */
    @Test
    void aReplayingPlanDrawsAnewEveryExecution() {
        boolean dsp = InferenceSession.isDynamicShapePlanEnabled();
        InferenceSession.setDynamicShapePlanEnabled(true);
        try (SameDiff sd = SameDiff.create()) {
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
            SDVariable noise = new DistributionUniform(sd, input.shape(), 0.0, 1.0, DataType.FLOAT).outputVariable();
            SDVariable out = input.mul(2.0).add(noise).add(1.0);
            INDArray in = Nd4j.ones(DataType.FLOAT, 4, 32);
            Nd4j.getRandom().setSeed(5);
            INDArray[] executions = new INDArray[8];
            for (int i = 0; i < executions.length; i++) {
                executions[i] = sd.output(Collections.singletonMap("input", in), out.name()).get(out.name()).dup();
            }
            for (int i = 1; i < executions.length; i++) {
                assertNotEquals(executions[i - 1], executions[i], "execution " + i + " repeated the previous draws");
            }
            Nd4j.getRandom().setSeed(5);
            assertEquals(executions[0], sd.output(Collections.singletonMap("input", in), out.name()).get(out.name()),
                    "reseeding must reproduce the first execution");
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(dsp);
        }
    }

    /** Constant folding would freeze a random op's output into a constant. */
    @Test
    void foldingKeepsRandomOpsOnConstants() {
        try (SameDiff sd = SameDiff.create()) {
            SDVariable dropped = sd.nn().dropout("dropped", sd.constant(Nd4j.ones(DataType.FLOAT, 32, 32)), false, 0.5);
            SDVariable cropped = sd.image().randomCrop("cropped",
                    sd.constant(Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, 50 * 50).reshape(50, 50)),
                    sd.constant(Nd4j.createFromArray(5, 5)));
            SameDiff optimized = GraphOptimizer.optimize(sd, dropped.name(), cropped.name());
            long randomOps = optimized.getOps().values().stream().map(SameDiffOp::getOp)
                    .filter(op -> op.opName().equals("dropout") || op.opName().equals("random_crop")).count();
            assertEquals(2, randomOps, "dropout and random_crop must survive constant folding");
        }
    }

    @Test
    void gammaSamplesHaveTheirDistributionsMoments() {
        Nd4j.getRandom().setSeed(7);
        float[] alphas = {0.5f, 1f, 2.5f, 10f};
        int n = 200_000;
        INDArray samples = Nd4j.exec(new RandomGamma(Nd4j.createFromArray((long) n), Nd4j.createFromArray(alphas), null))[0];
        assertEquals(DataType.FLOAT, samples.dataType());
        for (int c = 0; c < alphas.length; c++) {
            double alpha = alphas[c];
            INDArray column = samples.getColumn(c).castTo(DataType.DOUBLE);
            double mean = column.meanNumber().doubleValue();
            double variance = column.varNumber().doubleValue();
            // Gamma(alpha, 1): mean alpha, variance alpha; five standard errors.
            assertEquals(alpha, mean, 5 * Math.sqrt(alpha / n), "mean of Gamma(" + alpha + ")");
            assertEquals(alpha, variance, 5 * Math.sqrt((6 * alpha + 2 * alpha * alpha) / n), "variance of Gamma(" + alpha + ")");
        }
    }

    @Test
    void poissonSamplesHaveTheirDistributionsMoments() {
        Nd4j.getRandom().setSeed(7);
        // 200 used to hang the sequential search: exp(-200) underflows float.
        float[] lambdas = {0.5f, 4f, 30f, 200f};
        int n = 200_000;
        INDArray samples = Nd4j.exec(new RandomPoisson(Nd4j.createFromArray((long) n), Nd4j.createFromArray(lambdas)))[0];
        for (int c = 0; c < lambdas.length; c++) {
            double lambda = lambdas[c];
            INDArray column = samples.getColumn(c).castTo(DataType.DOUBLE);
            assertEquals(column, Transforms.floor(column, true), "Poisson samples are counts");
            double mean = column.meanNumber().doubleValue();
            double variance = column.varNumber().doubleValue();
            // Poisson(lambda): mean and variance lambda; five standard errors.
            assertEquals(lambda, mean, 5 * Math.sqrt(lambda / n), "mean of Poisson(" + lambda + ")");
            assertEquals(lambda, variance, 5 * Math.sqrt((lambda + 2 * lambda * lambda) / n), "variance of Poisson(" + lambda + ")");
        }
    }

    @Test
    void randomCropIsAWindowOfItsInput() {
        INDArray input = Nd4j.linspace(DataType.FLOAT, 0.0, 1.0, 100 * 100).reshape(100, 100);
        for (int seed = 1; seed <= 5; seed++) {
            Nd4j.getRandom().setSeed(seed);
            INDArray crop = Nd4j.exec(DynamicCustomOp.builder("random_crop")
                    .addInputs(input, Nd4j.createFromArray(10, 20))
                    .addOutputs(Nd4j.create(DataType.FLOAT, 10, 20)).build())[0];
            int corner = crop.getInt(0, 0);
            int row = corner / 100;
            int column = corner % 100;
            assertTrue(row <= 90 && column <= 80, "the crop must fit: corner " + row + "," + column);
            assertEquals(input.get(NDArrayIndex.interval(row, row + 10), NDArrayIndex.interval(column, column + 20)),
                    crop, "seed " + seed);
        }
    }

    @Test
    void dropoutBackpropPassesTheGradientWhereTheMaskKept() {
        Nd4j.getRandom().setSeed(11);
        INDArray input = Nd4j.rand(DataType.FLOAT, 16, 16);
        INDArray mask = Nd4j.rand(DataType.FLOAT, 16, 16).gt(0.5).castTo(DataType.FLOAT);
        INDArray gradOut = Nd4j.rand(DataType.FLOAT, 16, 16);
        INDArray expected = gradOut.mul(mask);
        INDArray maskBefore = mask.dup();
        INDArray gradIn = Nd4j.exec(new DropOutBp(new INDArray[]{input, mask, gradOut}, false, 0, 0.5))[0];
        assertEquals(expected, gradIn);
        assertEquals(maskBefore, mask, "the forward's mask is an input and stays as it was");
    }

    @Test
    void seededDropoutKeepsItsMaskAndUnseededDrawsANewOne() {
        boolean dsp = InferenceSession.isDynamicShapePlanEnabled();
        InferenceSession.setDynamicShapePlanEnabled(false);
        try (SameDiff sd = SameDiff.create()) {
            SDVariable ones = sd.constant(Nd4j.ones(DataType.FLOAT, 64, 64));
            SDVariable seeded = sd.nn().dropout("seeded", ones, false, 17, 0.5);
            SDVariable unseeded = sd.nn().dropout("unseeded", ones, false, 0.5);
            Map<String, INDArray> first = dupAll(sd.output(Collections.emptyMap(), seeded.name(), unseeded.name()));
            Map<String, INDArray> second = dupAll(sd.output(Collections.emptyMap(), seeded.name(), unseeded.name()));
            assertEquals(first.get(seeded.name()), second.get(seeded.name()), "a seed fixes the mask");
            assertNotEquals(first.get(unseeded.name()), second.get(unseeded.name()), "unseeded, each execution drops anew");
            double kept = first.get(unseeded.name()).sumNumber().doubleValue() / (64 * 64);
            assertEquals(0.5, kept, 0.05, "half the elements kept");
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(dsp);
        }
    }

    @Test
    void setSeedSeedsTheGenerator() {
        Nd4j.getRandom().setSeed(1);
        INDArray first = exponentialAfterSetSeed(7);
        Nd4j.getRandom().setSeed(2);
        INDArray second = exponentialAfterSetSeed(7);
        assertEquals(first, second, "set_seed must determine the draws that follow");
    }

    private static INDArray exponentialAfterSetSeed(long seed) {
        INDArray written = Nd4j.exec(DynamicCustomOp.builder("set_seed").addIntegerArguments(seed)
                .addOutputs(Nd4j.scalar(DataType.FLOAT, 0)).build())[0];
        assertEquals(seed, written.getLong(0), "set_seed writes the seed");
        return Nd4j.exec(new RandomExponential(2.0, DataType.FLOAT, 1000))[0];
    }

    private static Map<String, INDArray> dupAll(Map<String, INDArray> outputs) {
        outputs.replaceAll((name, array) -> array.dup());
        return outputs;
    }
}
