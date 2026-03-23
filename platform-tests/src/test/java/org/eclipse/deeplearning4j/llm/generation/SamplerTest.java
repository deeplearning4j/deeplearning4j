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

package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Sampler implementations.
 *
 * Adam Gibson
 */
class SamplerTest {

    @BeforeEach
    void setUp() {
        Nd4j.getRandom().setSeed(42);
    }

    // ==================== GreedySampler Tests ====================

    @Test
    void testGreedySampler_AlwaysSelectsMax() {
        GreedySampler sampler = new GreedySampler();
        INDArray logits = Nd4j.create(new double[]{1.0, 5.0, 3.0, 2.0});

        // Should always return index of max (1)
        for (int i = 0; i < 10; i++) {
            assertEquals(1, sampler.sample(logits));
        }
    }

    @Test
    void testGreedySampler_Batch() {
        GreedySampler sampler = new GreedySampler();
        INDArray logits = Nd4j.create(new double[][]{{1.0, 3.0, 2.0}, {5.0, 1.0, 2.0}});

        int[] results = sampler.sampleBatch(logits);
        assertEquals(2, results.length);
        assertEquals(1, results[0]);
        assertEquals(0, results[1]);
    }

    @Test
    void testGreedySampler_Name() {
        GreedySampler sampler = new GreedySampler();
        assertEquals("greedy", sampler.getName());
    }

    // ==================== CompositeSampler Tests ====================

    @Test
    void testCompositeSampler_TemperatureOnly() {
        CompositeSampler sampler = CompositeSampler.withTemperature(0.5);

        assertEquals(0.5, sampler.getTemperature(), 1e-6);
        assertEquals(0, sampler.getTopK());
        assertEquals(1.0, sampler.getTopP(), 1e-6);
    }

    @Test
    void testCompositeSampler_TopKOnly() {
        CompositeSampler sampler = CompositeSampler.withTopK(50);

        assertEquals(1.0, sampler.getTemperature(), 1e-6);
        assertEquals(50, sampler.getTopK());
        assertEquals(1.0, sampler.getTopP(), 1e-6);
    }

    @Test
    void testCompositeSampler_TopPOnly() {
        CompositeSampler sampler = CompositeSampler.withTopP(0.9);

        assertEquals(1.0, sampler.getTemperature(), 1e-6);
        assertEquals(0, sampler.getTopK());
        assertEquals(0.9, sampler.getTopP(), 1e-6);
    }

    @Test
    void testCompositeSampler_Combined() {
        CompositeSampler sampler = CompositeSampler.builder()
                .temperature(0.8)
                .topK(40)
                .topP(0.95)
                .seed(42L)
                .build();

        assertEquals(0.8, sampler.getTemperature(), 1e-6);
        assertEquals(40, sampler.getTopK());
        assertEquals(0.95, sampler.getTopP(), 1e-6);
    }

    @Test
    void testCompositeSampler_ProducesVariation() {
        CompositeSampler sampler = CompositeSampler.builder()
                .temperature(1.0)
                .seed(null) // Random seed
                .build();

        // Create logits where multiple tokens have similar probabilities
        INDArray logits = Nd4j.create(new double[]{2.0, 2.1, 2.0, 1.9, 2.0});

        // Sample multiple times and check for variation
        Set<Integer> samples = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            samples.add(sampler.sample(logits));
        }

        // Should have variation (more than just argmax)
        assertTrue(samples.size() > 1, "Expected variation in samples");
    }

    @Test
    void testCompositeSampler_DeterministicWithSeed() {
        CompositeSampler sampler1 = CompositeSampler.builder()
                .temperature(0.7)
                .seed(12345L)
                .build();

        CompositeSampler sampler2 = CompositeSampler.builder()
                .temperature(0.7)
                .seed(12345L)
                .build();

        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0, 2.5, 1.5});

        // Same seed should produce same sequence
        for (int i = 0; i < 10; i++) {
            assertEquals(sampler1.sample(logits), sampler2.sample(logits));
        }
    }

    @Test
    void testCompositeSampler_Name() {
        CompositeSampler sampler = CompositeSampler.builder()
                .temperature(0.8)
                .topK(40)
                .topP(0.95)
                .build();

        String name = sampler.getName();
        assertTrue(name.contains("temp="));
        assertTrue(name.contains("topK="));
        assertTrue(name.contains("topP="));
    }

    @Test
    void testCompositeSampler_GetDistribution() {
        CompositeSampler sampler = CompositeSampler.builder()
                .temperature(1.0)
                .build();

        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray probs = sampler.getDistribution(logits);

        // Should be valid probability distribution
        assertTrue(SamplerUtils.isValidDistribution(probs, 1e-6));
    }

    // ==================== Sampler Factory Tests ====================

    @Test
    void testSamplerFromConfig_Greedy() {
        SamplingConfig config = SamplingConfig.greedy();
        Sampler sampler = Sampler.fromConfig(config);

        assertTrue(sampler instanceof GreedySampler);
    }

    @Test
    void testSamplerFromConfig_Sampling() {
        SamplingConfig config = SamplingConfig.builder()
                .temperature(0.7)
                .topK(50)
                .topP(0.9)
                .doSample(true)
                .build();

        Sampler sampler = Sampler.fromConfig(config);
        assertTrue(sampler instanceof CompositeSampler);

        CompositeSampler cs = (CompositeSampler) sampler;
        assertEquals(0.7, cs.getTemperature(), 1e-6);
        assertEquals(50, cs.getTopK());
        assertEquals(0.9, cs.getTopP(), 1e-6);
    }

    @Test
    void testSamplerFromConfig_GreedyByTemperature() {
        // Even with doSample=true, temperature<=0 should be greedy
        SamplingConfig config = SamplingConfig.builder()
                .temperature(0.0)
                .doSample(true)
                .build();

        Sampler sampler = Sampler.fromConfig(config);
        assertTrue(sampler instanceof GreedySampler);
    }

    // ==================== SamplingConfig Tests ====================

    @Test
    void testSamplingConfig_Defaults() {
        SamplingConfig config = SamplingConfig.builder().build();

        assertEquals(1.0, config.getTemperature(), 1e-6);
        assertEquals(0, config.getTopK());
        assertEquals(1.0, config.getTopP(), 1e-6);
        assertTrue(config.isDoSample());
        assertEquals(1.0, config.getRepetitionPenalty(), 1e-6);
        assertEquals(512, config.getMaxNewTokens());
    }

    @Test
    void testSamplingConfig_Presets() {
        SamplingConfig greedy = SamplingConfig.greedy();
        assertTrue(greedy.isGreedy());

        SamplingConfig creative = SamplingConfig.creative();
        assertTrue(creative.getTemperature() > 0.8);
        assertTrue(creative.getTopK() > 0);

        SamplingConfig precise = SamplingConfig.precise();
        assertTrue(precise.getTemperature() < 0.5);
    }

    @Test
    void testSamplingConfig_HasChecks() {
        SamplingConfig config = SamplingConfig.builder()
                .topK(50)
                .topP(0.9)
                .repetitionPenalty(1.2)
                .build();

        assertTrue(config.hasTopK());
        assertTrue(config.hasTopP());
        assertTrue(config.hasRepetitionPenalty());
    }

    @Test
    void testSamplingConfig_NoHas() {
        SamplingConfig config = SamplingConfig.builder()
                .topK(0)
                .topP(1.0)
                .repetitionPenalty(1.0)
                .build();

        assertFalse(config.hasTopK());
        assertFalse(config.hasTopP());
        assertFalse(config.hasRepetitionPenalty());
    }

    // ==================== Integration Tests ====================

    @Test
    void testSamplerIntegration_TopKLimitsChoices() {
        CompositeSampler sampler = CompositeSampler.builder()
                .topK(2)
                .seed(42L)
                .build();

        // Logits with clear top 2: indices 1 and 3
        INDArray logits = Nd4j.create(new double[]{1.0, 10.0, 1.0, 9.0, 1.0});

        Set<Integer> samples = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            samples.add(sampler.sample(logits));
        }

        // Should only sample from top 2 (indices 1 and 3)
        assertTrue(samples.contains(1) || samples.contains(3));
        assertFalse(samples.contains(0));
        assertFalse(samples.contains(2));
        assertFalse(samples.contains(4));
    }

    @Test
    void testSamplerIntegration_HighTemperatureFlattens() {
        // Use more balanced logits where temperature effect is more visible
        // With these values, high temp will actually spread probability more evenly
        INDArray logits = Nd4j.create(new double[]{1.0, 2.0, 3.0, 2.5});

        // Low temperature: should strongly favor index 2 (highest logit)
        CompositeSampler lowTemp = CompositeSampler.builder()
                .temperature(0.1)
                .seed(42L)
                .build();

        int lowTempIdx2 = 0;
        for (int i = 0; i < 100; i++) {
            if (lowTemp.sample(logits) == 2) lowTempIdx2++;
        }

        // High temperature: more variation, less likely to always pick argmax
        CompositeSampler highTemp = CompositeSampler.builder()
                .temperature(2.0)
                .seed(42L)
                .build();

        int highTempIdx2 = 0;
        for (int i = 0; i < 100; i++) {
            if (highTemp.sample(logits) == 2) highTempIdx2++;
        }

        // Low temp should pick index 2 more often (or equal is also acceptable
        // since low temp makes it nearly deterministic)
        assertTrue(lowTempIdx2 >= highTempIdx2,
                "Low temp should favor argmax at least as much: " + lowTempIdx2 + " vs " + highTempIdx2);
    }
}
