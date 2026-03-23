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

package org.eclipse.deeplearning4j.nd4j.linalg;

import org.eclipse.deeplearning4j.llm.editing.AbliterationConfig;
import org.eclipse.deeplearning4j.llm.editing.AbliterationResult;
import org.eclipse.deeplearning4j.llm.editing.AbliterationWorkflow;
import org.eclipse.deeplearning4j.llm.editing.DefaultPromptSets;
import org.eclipse.deeplearning4j.llm.editing.RefusalDirection;
import org.eclipse.deeplearning4j.llm.editing.RefusalDirectionFinder;
import org.eclipse.deeplearning4j.llm.editing.WeightOrthogonalizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the abliteration (training-free refusal removal) workflow.
 * Uses synthetic data and small models to verify the core algorithms.
 */
public class AbliterationWorkflowTest {

    /**
     * Test that diff-in-means direction computation produces a unit vector
     * that separates harmful from harmless activations.
     */
    @Test
    public void testDiffInMeansProducesUnitVector() {
        int hiddenDim = 64;
        int numSamples = 20;

        // Create synthetic activations with a known separating direction
        INDArray refusalDir = Nd4j.randn(DataType.FLOAT, 1, hiddenDim);
        refusalDir.divi(refusalDir.norm2Number().doubleValue());

        // Harmful activations: random + bias along refusal direction
        INDArray harmful = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
        harmful.addiRowVector(refusalDir.mul(5.0));

        // Harmless activations: random + bias away from refusal direction
        INDArray harmless = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
        harmless.addiRowVector(refusalDir.mul(-5.0));

        Map<String, INDArray> harmfulActs = new LinkedHashMap<>();
        harmfulActs.put("model.layers.0.attn.o_proj.weight", harmful);

        Map<String, INDArray> harmlessActs = new LinkedHashMap<>();
        harmlessActs.put("model.layers.0.attn.o_proj.weight", harmless);

        AbliterationConfig config = AbliterationConfig.builder()
                .method(AbliterationConfig.Method.DIFF_IN_MEANS)
                .build();

        RefusalDirectionFinder finder = new RefusalDirectionFinder();
        List<RefusalDirection> directions = finder.findRefusalDirections(
                null, harmfulActs, harmlessActs, config);

        assertFalse(directions.isEmpty(), "Should find at least one direction");
        RefusalDirection dir = directions.get(0);

        // Direction should be a unit vector
        double norm = dir.getDirection().norm2Number().doubleValue();
        assertEquals(1.0, norm, 1e-5, "Direction should be unit length");

        // Score should be positive
        assertTrue(dir.getScore() > 0, "Score should be positive");

        // Direction should align with the planted refusal direction
        double alignment = Math.abs(Nd4j.getBlasWrapper().dot(
                dir.getDirection(), refusalDir.reshape(hiddenDim)));
        assertTrue(alignment > 0.5, "Found direction should align with planted direction, got " + alignment);
    }

    /**
     * Test all three methods: DIFF_IN_MEANS, PCA, PROJECTED.
     */
    @ParameterizedTest
    @EnumSource(AbliterationConfig.Method.class)
    public void testAllMethods(AbliterationConfig.Method method) {
        int hiddenDim = 32;
        int numSamples = 30;

        INDArray refusalDir = Nd4j.randn(DataType.FLOAT, 1, hiddenDim);
        refusalDir.divi(refusalDir.norm2Number().doubleValue());

        INDArray harmful = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
        harmful.addiRowVector(refusalDir.mul(3.0));

        INDArray harmless = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
        harmless.addiRowVector(refusalDir.mul(-3.0));

        Map<String, INDArray> harmfulActs = new LinkedHashMap<>();
        harmfulActs.put("model.layers.0.mlp.down_proj.weight", harmful);

        Map<String, INDArray> harmlessActs = new LinkedHashMap<>();
        harmlessActs.put("model.layers.0.mlp.down_proj.weight", harmless);

        AbliterationConfig config = AbliterationConfig.builder()
                .method(method)
                .build();

        RefusalDirectionFinder finder = new RefusalDirectionFinder();
        List<RefusalDirection> directions = finder.findRefusalDirections(
                null, harmfulActs, harmlessActs, config);

        assertFalse(directions.isEmpty(), "Should find directions for method " + method);
        RefusalDirection dir = directions.get(0);

        double norm = dir.getDirection().norm2Number().doubleValue();
        assertEquals(1.0, norm, 1e-4, "Direction should be unit length for " + method);
        assertTrue(dir.getScore() > 0, "Score should be positive for " + method);
    }

    /**
     * Test that orthogonalization actually removes the direction component from weights.
     */
    @Test
    public void testOrthogonalizationRemovesProjection() {
        int outDim = 16;
        int inDim = 32;

        SameDiff sd = SameDiff.create();
        INDArray weightData = Nd4j.randn(DataType.FLOAT, outDim, inDim);
        sd.constant("model.layers.0.attn.o_proj.weight", weightData);

        // Create a direction in the input dimension space
        INDArray direction = Nd4j.randn(DataType.FLOAT, inDim);
        direction.divi(direction.norm2Number().doubleValue());

        // Measure projection before
        INDArray dCol = direction.reshape(inDim, 1);
        INDArray projBefore = weightData.mmul(dCol); // [outDim, 1]
        double projNormBefore = projBefore.norm2Number().doubleValue();

        // Orthogonalize
        RefusalDirection refDir = new RefusalDirection(
                "model.layers.0.attn.o_proj.weight",
                AbliterationConfig.ResidualStreamPosition.POST_ATTENTION,
                direction, 1.0, 0);

        AbliterationConfig config = AbliterationConfig.builder()
                .ablationStrength(1.0)
                .build();

        WeightOrthogonalizer orthogonalizer = new WeightOrthogonalizer();
        List<String> modified = orthogonalizer.orthogonalizeWeights(
                sd, Collections.singletonList(refDir), config);

        assertFalse(modified.isEmpty(), "Should have modified at least one weight");

        // Measure projection after
        INDArray modifiedWeight = sd.getVariable("model.layers.0.attn.o_proj.weight").getArr();
        INDArray projAfter = modifiedWeight.mmul(dCol);
        double projNormAfter = projAfter.norm2Number().doubleValue();

        // Projection should be effectively zero after full orthogonalization
        assertTrue(projNormAfter < projNormBefore * 0.01,
                "Projection norm should be near zero after orthogonalization. " +
                        "Before: " + projNormBefore + ", After: " + projNormAfter);
    }

    /**
     * Test partial ablation strength.
     */
    @Test
    public void testPartialAblationStrength() {
        int outDim = 16;
        int inDim = 32;

        SameDiff sd = SameDiff.create();
        INDArray weightData = Nd4j.randn(DataType.FLOAT, outDim, inDim);
        sd.constant("model.layers.0.attn.o_proj.weight", weightData.dup());

        INDArray direction = Nd4j.randn(DataType.FLOAT, inDim);
        direction.divi(direction.norm2Number().doubleValue());

        INDArray dCol = direction.reshape(inDim, 1);
        double projNormBefore = weightData.mmul(dCol).norm2Number().doubleValue();

        double alpha = 0.5;
        RefusalDirection refDir = new RefusalDirection(
                "model.layers.0.attn.o_proj.weight",
                AbliterationConfig.ResidualStreamPosition.POST_ATTENTION,
                direction, 1.0, 0);

        AbliterationConfig config = AbliterationConfig.builder()
                .ablationStrength(alpha)
                .build();

        WeightOrthogonalizer orthogonalizer = new WeightOrthogonalizer();
        orthogonalizer.orthogonalizeWeights(sd, Collections.singletonList(refDir), config);

        INDArray modifiedWeight = sd.getVariable("model.layers.0.attn.o_proj.weight").getArr();
        double projNormAfter = modifiedWeight.mmul(dCol).norm2Number().doubleValue();

        // With alpha=0.5, projection should be roughly halved
        double ratio = projNormAfter / projNormBefore;
        assertTrue(ratio > 0.3 && ratio < 0.7,
                "Partial ablation should roughly halve projection. Ratio: " + ratio);
    }

    /**
     * Test Winsorization clips extreme values.
     */
    @Test
    public void testWinsorization() {
        int hiddenDim = 32;
        int numSamples = 100;

        // Create data with outliers
        INDArray harmful = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
        harmful.putScalar(0, 0, 1000.0f); // extreme outlier
        harmful.putScalar(1, 0, -1000.0f); // extreme outlier

        INDArray harmless = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);

        Map<String, INDArray> harmfulActs = new LinkedHashMap<>();
        harmfulActs.put("model.layers.0.attn.o_proj.weight", harmful);

        Map<String, INDArray> harmlessActs = new LinkedHashMap<>();
        harmlessActs.put("model.layers.0.attn.o_proj.weight", harmless);

        // Without winsorization
        AbliterationConfig configNoWinsorize = AbliterationConfig.builder()
                .method(AbliterationConfig.Method.DIFF_IN_MEANS)
                .winsorize(false)
                .build();

        RefusalDirectionFinder finder = new RefusalDirectionFinder();
        List<RefusalDirection> dirsNoWinsorize = finder.findRefusalDirections(
                null, harmfulActs, harmlessActs, configNoWinsorize);

        // With winsorization
        AbliterationConfig configWinsorize = AbliterationConfig.builder()
                .method(AbliterationConfig.Method.DIFF_IN_MEANS)
                .winsorize(true)
                .winsorizePercentile(0.99)
                .build();

        List<RefusalDirection> dirsWinsorize = finder.findRefusalDirections(
                null, harmfulActs, harmlessActs, configWinsorize);

        // Both should produce valid unit vectors
        assertFalse(dirsNoWinsorize.isEmpty());
        assertFalse(dirsWinsorize.isEmpty());

        assertEquals(1.0, dirsNoWinsorize.get(0).getDirection().norm2Number().doubleValue(), 1e-4);
        assertEquals(1.0, dirsWinsorize.get(0).getDirection().norm2Number().doubleValue(), 1e-4);

        // Winsorized direction should differ due to outlier clipping
        double alignment = Math.abs(Nd4j.getBlasWrapper().dot(
                dirsNoWinsorize.get(0).getDirection(),
                dirsWinsorize.get(0).getDirection()));
        // They should be somewhat different because the outliers skew the non-winsorized version
        assertTrue(alignment < 0.999, "Winsorization should change direction when outliers present");
    }

    /**
     * Test the full workflow with pre-computed activations.
     */
    @Test
    public void testFullWorkflowWithPrecomputedActivations() {
        int hiddenDim = 64;
        int numSamples = 20;
        int numLayers = 4;

        SameDiff sd = SameDiff.create();
        Map<String, INDArray> harmfulActs = new LinkedHashMap<>();
        Map<String, INDArray> harmlessActs = new LinkedHashMap<>();

        // Create a known refusal direction
        INDArray refusalDir = Nd4j.randn(DataType.FLOAT, 1, hiddenDim);
        refusalDir.divi(refusalDir.norm2Number().doubleValue());

        for (int layer = 0; layer < numLayers; layer++) {
            String name = "model.layers." + layer + ".attn.o_proj.weight";

            // Add weight to model
            INDArray weight = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim);
            sd.constant(name, weight);

            // Create activations — make the refusal signal stronger in middle layers
            double signalStrength = (layer == 1 || layer == 2) ? 5.0 : 1.0;

            INDArray harmful = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
            harmful.addiRowVector(refusalDir.mul(signalStrength));

            INDArray harmless = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
            harmless.addiRowVector(refusalDir.mul(-signalStrength));

            harmfulActs.put(name, harmful);
            harmlessActs.put(name, harmless);
        }

        AbliterationResult result = AbliterationWorkflow.builder()
                .model(sd)
                .config(AbliterationConfig.builder()
                        .method(AbliterationConfig.Method.DIFF_IN_MEANS)
                        .layerSelectionStrategy(AbliterationConfig.LayerSelection.BEST_SINGLE)
                        .build())
                .harmfulActivations(harmfulActs)
                .harmlessActivations(harmlessActs)
                .build()
                .execute();

        assertNotNull(result);
        assertEquals(numLayers, result.getAllCandidates().size());
        assertEquals(1, result.getAppliedDirections().size());
        assertTrue(result.getNumWeightsModified() > 0);

        // The best direction should come from layer 1 or 2 (strongest signal)
        RefusalDirection best = result.getAppliedDirections().get(0);
        assertTrue(best.getLayerIndex() == 1 || best.getLayerIndex() == 2,
                "Best direction should come from layer with strongest signal, got layer " + best.getLayerIndex());
    }

    /**
     * Test TOP_K layer selection strategy.
     */
    @Test
    public void testTopKSelection() {
        int hiddenDim = 32;
        int numSamples = 15;
        int numLayers = 6;

        SameDiff sd = SameDiff.create();
        Map<String, INDArray> harmfulActs = new LinkedHashMap<>();
        Map<String, INDArray> harmlessActs = new LinkedHashMap<>();

        INDArray refusalDir = Nd4j.randn(DataType.FLOAT, 1, hiddenDim);
        refusalDir.divi(refusalDir.norm2Number().doubleValue());

        for (int layer = 0; layer < numLayers; layer++) {
            String name = "model.layers." + layer + ".attn.o_proj.weight";
            sd.constant(name, Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim));

            INDArray harmful = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
            harmful.addiRowVector(refusalDir.mul(layer + 1));
            INDArray harmless = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
            harmless.addiRowVector(refusalDir.mul(-(layer + 1)));

            harmfulActs.put(name, harmful);
            harmlessActs.put(name, harmless);
        }

        AbliterationResult result = AbliterationWorkflow.builder()
                .model(sd)
                .config(AbliterationConfig.builder()
                        .method(AbliterationConfig.Method.DIFF_IN_MEANS)
                        .layerSelectionStrategy(AbliterationConfig.LayerSelection.TOP_K)
                        .topK(3)
                        .build())
                .harmfulActivations(harmfulActs)
                .harmlessActivations(harmlessActs)
                .build()
                .execute();

        assertEquals(3, result.getAppliedDirections().size(), "Should select top 3 directions");
    }

    /**
     * Test multiple probe points across layers.
     */
    @Test
    public void testMultipleProbePointsRanking() {
        int hiddenDim = 32;
        int numSamples = 20;

        INDArray refusalDir = Nd4j.randn(DataType.FLOAT, 1, hiddenDim);
        refusalDir.divi(refusalDir.norm2Number().doubleValue());

        Map<String, INDArray> harmfulActs = new LinkedHashMap<>();
        Map<String, INDArray> harmlessActs = new LinkedHashMap<>();

        // Layer 0: weak signal
        INDArray harmful0 = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
        harmful0.addiRowVector(refusalDir.mul(1.0));
        INDArray harmless0 = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
        harmless0.addiRowVector(refusalDir.mul(-1.0));
        harmfulActs.put("model.layers.0.mlp.down_proj.weight", harmful0);
        harmlessActs.put("model.layers.0.mlp.down_proj.weight", harmless0);

        // Layer 1: strong signal
        INDArray harmful1 = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
        harmful1.addiRowVector(refusalDir.mul(10.0));
        INDArray harmless1 = Nd4j.randn(DataType.FLOAT, numSamples, hiddenDim);
        harmless1.addiRowVector(refusalDir.mul(-10.0));
        harmfulActs.put("model.layers.1.mlp.down_proj.weight", harmful1);
        harmlessActs.put("model.layers.1.mlp.down_proj.weight", harmless1);

        AbliterationConfig config = AbliterationConfig.builder()
                .method(AbliterationConfig.Method.DIFF_IN_MEANS)
                .build();

        RefusalDirectionFinder finder = new RefusalDirectionFinder();
        List<RefusalDirection> directions = finder.findRefusalDirections(
                null, harmfulActs, harmlessActs, config);

        assertEquals(2, directions.size());
        // Layer 1 should rank first (stronger signal)
        assertEquals(1, directions.get(0).getLayerIndex(),
                "Layer with stronger signal should rank first");
        assertTrue(directions.get(0).getScore() > directions.get(1).getScore(),
                "First direction should have higher score");
    }

    /**
     * Test layer index extraction from variable names.
     */
    @Test
    public void testLayerIndexExtraction() {
        assertEquals(0, RefusalDirectionFinder.extractLayerIndex("model.layers.0.attn.o_proj.weight"));
        assertEquals(15, RefusalDirectionFinder.extractLayerIndex("model.layers.15.mlp.down_proj.weight"));
        assertEquals(3, RefusalDirectionFinder.extractLayerIndex("transformer.h.3.attn.c_proj.weight"));
        assertEquals(-1, RefusalDirectionFinder.extractLayerIndex("model.embed_tokens.weight"));
    }

    /**
     * Test position inference from variable names.
     */
    @Test
    public void testPositionInference() {
        assertEquals(AbliterationConfig.ResidualStreamPosition.POST_ATTENTION,
                RefusalDirectionFinder.inferPosition("model.layers.0.attn.o_proj.weight"));
        assertEquals(AbliterationConfig.ResidualStreamPosition.POST_MLP,
                RefusalDirectionFinder.inferPosition("model.layers.0.mlp.down_proj.weight"));
        assertEquals(AbliterationConfig.ResidualStreamPosition.PRE_ATTENTION,
                RefusalDirectionFinder.inferPosition("model.layers.0.input_layernorm.weight"));
    }

    /**
     * Test default prompt sets are non-empty and reasonable.
     */
    @Test
    public void testDefaultPromptSets() {
        List<String> harmful = DefaultPromptSets.getDefaultHarmfulPrompts();
        List<String> harmless = DefaultPromptSets.getDefaultHarmlessPrompts();

        assertFalse(harmful.isEmpty(), "Harmful prompts should not be empty");
        assertFalse(harmless.isEmpty(), "Harmless prompts should not be empty");
        assertTrue(harmful.size() >= 50, "Should have at least 50 harmful prompts");
        assertTrue(harmless.size() >= 50, "Should have at least 50 harmless prompts");

        // All prompts should be non-empty strings
        for (String p : harmful) {
            assertFalse(p.isEmpty(), "Harmful prompt should not be empty");
        }
        for (String p : harmless) {
            assertFalse(p.isEmpty(), "Harmless prompt should not be empty");
        }
    }

    /**
     * Test AbliterationConfig factory methods.
     */
    @Test
    public void testConfigFactoryMethods() {
        AbliterationConfig defaults = AbliterationConfig.defaults();
        assertEquals(AbliterationConfig.Method.DIFF_IN_MEANS, defaults.getMethod());
        assertEquals(1.0, defaults.getAblationStrength(), 1e-10);

        AbliterationConfig conservative = AbliterationConfig.conservative();
        assertEquals(AbliterationConfig.Method.PROJECTED, conservative.getMethod());
        assertEquals(0.8, conservative.getAblationStrength(), 1e-10);

        AbliterationConfig winsorized = AbliterationConfig.withWinsorization();
        assertTrue(winsorized.isWinsorize());
        assertEquals(0.995, winsorized.getWinsorizePercentile(), 1e-10);
    }

    /**
     * Test that orthogonalization with direction matching outDim works too.
     */
    @Test
    public void testOrthogonalizationOutDimDirection() {
        int outDim = 32;
        int inDim = 16;

        SameDiff sd = SameDiff.create();
        INDArray weightData = Nd4j.randn(DataType.FLOAT, outDim, inDim);
        sd.constant("model.layers.0.attn.o_proj.weight", weightData.dup());

        // Direction in the output dimension space
        INDArray direction = Nd4j.randn(DataType.FLOAT, outDim);
        direction.divi(direction.norm2Number().doubleValue());

        INDArray dRow = direction.reshape(1, outDim);
        double projNormBefore = dRow.mmul(weightData).norm2Number().doubleValue();

        RefusalDirection refDir = new RefusalDirection(
                "model.layers.0.attn.o_proj.weight",
                AbliterationConfig.ResidualStreamPosition.POST_ATTENTION,
                direction, 1.0, 0);

        AbliterationConfig config = AbliterationConfig.builder()
                .ablationStrength(1.0)
                .build();

        WeightOrthogonalizer orthogonalizer = new WeightOrthogonalizer();
        orthogonalizer.orthogonalizeWeights(sd, Collections.singletonList(refDir), config);

        INDArray modifiedWeight = sd.getVariable("model.layers.0.attn.o_proj.weight").getArr();
        double projNormAfter = dRow.mmul(modifiedWeight).norm2Number().doubleValue();

        assertTrue(projNormAfter < projNormBefore * 0.01,
                "Projection should be near zero. Before: " + projNormBefore + ", After: " + projNormAfter);
    }
}
