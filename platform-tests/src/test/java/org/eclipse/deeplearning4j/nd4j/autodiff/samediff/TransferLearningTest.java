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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.*;
import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.autodiff.samediff.peft.PeftModelFactory;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for SameDiff transfer learning API.
 */
@DisplayName("Transfer Learning Tests")
public class TransferLearningTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // ==================== VariableGroup Tests ====================

    @Test
    @DisplayName("VariableGroup - Basic pattern matching")
    public void testVariableGroupBasicPatternMatching() {
        VariableGroup group = new VariableGroup("encoder", 0.1, "encoder\\.layer\\.[0-5]\\..*");

        assertTrue(group.matches("encoder.layer.0.weight"));
        assertTrue(group.matches("encoder.layer.3.bias"));
        assertTrue(group.matches("encoder.layer.5.norm"));
        assertFalse(group.matches("encoder.layer.6.weight"));
        assertFalse(group.matches("decoder.layer.0.weight"));
        assertFalse(group.matches("classifier.weight"));
    }

    @Test
    @DisplayName("VariableGroup - Multiple patterns")
    public void testVariableGroupMultiplePatterns() {
        VariableGroup group = new VariableGroup("frozen", 0.0,
                Arrays.asList("encoder\\..*", "embedding\\..*"));

        assertTrue(group.matches("encoder.layer.0.weight"));
        assertTrue(group.matches("embedding.word"));
        assertFalse(group.matches("decoder.layer.0.weight"));
    }

    @Test
    @DisplayName("VariableGroup - Learning rate multiplier")
    public void testVariableGroupLearningRateMultiplier() {
        VariableGroup group = new VariableGroup("head", 2.0, "classifier\\..*");

        assertEquals(2.0, group.getLearningRateMultiplier(), 1e-6);
        assertTrue(group.matches("classifier.weight"));
        assertTrue(group.matches("classifier.bias"));
    }

    @Test
    @DisplayName("VariableGroup - With custom updater")
    public void testVariableGroupWithCustomUpdater() {
        VariableGroup group = VariableGroup.builder()
                .name("special")
                .updater(new Sgd(0.01))
                .learningRateMultiplier(1.0)
                .addVariablePatterns("special\\..*")
                .build();

        assertTrue(group.hasUpdater());
        assertTrue(group.getUpdater() instanceof Sgd);
        assertEquals(0.01, ((Sgd) group.getUpdater()).getLearningRate(), 1e-6);
    }

    @Test
    @DisplayName("VariableGroup - Builder pattern")
    public void testVariableGroupBuilder() {
        VariableGroup group = VariableGroup.builder()
                .name("test_group")
                .learningRateMultiplier(0.5)
                .addVariablePatterns("layer1\\..*", "layer2\\..*")
                .build();

        assertEquals("test_group", group.getName());
        assertEquals(0.5, group.getLearningRateMultiplier(), 1e-6);
        assertTrue(group.matches("layer1.weight"));
        assertTrue(group.matches("layer2.bias"));
        assertFalse(group.matches("layer3.weight"));
    }

    // ==================== FineTuneConfiguration Tests ====================

    @Test
    @DisplayName("FineTuneConfiguration - Basic builder")
    public void testFineTuneConfigurationBasicBuilder() {
        FineTuneConfiguration config = FineTuneConfiguration.builder()
                .updater(new Adam(1e-4))
                .l2(1e-5)
                .minimize(true)
                .build();

        assertNotNull(config);
        assertNotNull(config.getDefaultUpdater());
        assertTrue(config.getDefaultUpdater() instanceof Adam);
        assertTrue(config.getMinimize());
    }

    @Test
    @DisplayName("FineTuneConfiguration - Freeze by exact name")
    public void testFineTuneConfigurationFreezeByName() {
        FineTuneConfiguration config = FineTuneConfiguration.builder()
                .freeze("layer1.weight", "layer1.bias", "layer2.weight")
                .build();

        assertTrue(config.isFrozen("layer1.weight"));
        assertTrue(config.isFrozen("layer1.bias"));
        assertTrue(config.isFrozen("layer2.weight"));
        assertFalse(config.isFrozen("layer2.bias"));
        assertFalse(config.isFrozen("layer3.weight"));
    }

    @Test
    @DisplayName("FineTuneConfiguration - Freeze by prefix")
    public void testFineTuneConfigurationFreezeByPrefix() {
        FineTuneConfiguration config = FineTuneConfiguration.builder()
                .freezePrefix("encoder.layer.0")
                .freezePrefix("encoder.layer.1")
                .build();

        assertTrue(config.isFrozen("encoder.layer.0.weight"));
        assertTrue(config.isFrozen("encoder.layer.0.bias"));
        assertTrue(config.isFrozen("encoder.layer.1.weight"));
        assertFalse(config.isFrozen("encoder.layer.2.weight"));
        assertFalse(config.isFrozen("decoder.layer.0.weight"));
    }

    @Test
    @DisplayName("FineTuneConfiguration - Freeze by pattern")
    public void testFineTuneConfigurationFreezeByPattern() {
        FineTuneConfiguration config = FineTuneConfiguration.builder()
                .freezeMatching(".*embedding.*")
                .freezeMatching("encoder\\.layer\\.[0-3]\\..*")
                .build();

        assertTrue(config.isFrozen("token_embedding"));
        assertTrue(config.isFrozen("position_embedding_layer"));
        assertTrue(config.isFrozen("encoder.layer.0.weight"));
        assertTrue(config.isFrozen("encoder.layer.3.bias"));
        assertFalse(config.isFrozen("encoder.layer.4.weight"));
        assertFalse(config.isFrozen("classifier"));
    }

    @Test
    @DisplayName("FineTuneConfiguration - Variable groups")
    public void testFineTuneConfigurationVariableGroups() {
        FineTuneConfiguration config = FineTuneConfiguration.builder()
                .updater(new Adam(1e-4))
                .variableGroup("early", 0.1, "encoder\\.layer\\.[0-3]\\..*")
                .variableGroup("middle", 0.5, "encoder\\.layer\\.[4-7]\\..*")
                .variableGroup("late", 1.0, "encoder\\.layer\\.(8|9|10|11)\\..*")
                .variableGroup("head", 2.0, "classifier\\..*")
                .build();

        assertNotNull(config.getVariableGroups());
        assertEquals(4, config.getVariableGroups().size());

        IUpdater earlyUpdater = config.getUpdaterForVariable("encoder.layer.0.weight");
        IUpdater headUpdater = config.getUpdaterForVariable("classifier.weight");

        assertNotNull(earlyUpdater);
        assertNotNull(headUpdater);
    }

    @Test
    @DisplayName("FineTuneConfiguration - Per-variable updater")
    public void testFineTuneConfigurationPerVariableUpdater() {
        FineTuneConfiguration config = FineTuneConfiguration.builder()
                .updater(new Adam(1e-4))
                .updater("special.weight", new Sgd(0.01))
                .build();

        IUpdater defaultUpdater = config.getUpdaterForVariable("regular.weight");
        IUpdater specialUpdater = config.getUpdaterForVariable("special.weight");

        assertTrue(defaultUpdater instanceof Adam);
        assertTrue(specialUpdater instanceof Sgd);
    }

    // ==================== SameDiff Selective Freezing Tests ====================

    @Test
    @DisplayName("SameDiff - Freeze variables by name")
    public void testSameDiffFreezeVariablesByName() {
        SameDiff sd = SameDiff.create();

        sd.var("layer1.weight", Nd4j.randn(DataType.FLOAT, 10, 20));
        sd.var("layer1.bias", Nd4j.zeros(DataType.FLOAT, 20));
        sd.var("layer2.weight", Nd4j.randn(DataType.FLOAT, 20, 5));
        sd.var("layer2.bias", Nd4j.zeros(DataType.FLOAT, 5));

        assertEquals(VariableType.VARIABLE, sd.getVariable("layer1.weight").getVariableType());
        assertEquals(VariableType.VARIABLE, sd.getVariable("layer2.weight").getVariableType());

        sd.freezeVariables("layer1.weight", "layer1.bias");

        assertEquals(VariableType.CONSTANT, sd.getVariable("layer1.weight").getVariableType());
        assertEquals(VariableType.CONSTANT, sd.getVariable("layer1.bias").getVariableType());
        assertEquals(VariableType.VARIABLE, sd.getVariable("layer2.weight").getVariableType());
        assertEquals(VariableType.VARIABLE, sd.getVariable("layer2.bias").getVariableType());
    }

    @Test
    @DisplayName("SameDiff - Freeze variables by pattern")
    public void testSameDiffFreezeVariablesByPattern() {
        SameDiff sd = SameDiff.create();

        sd.var("encoder.layer.0.weight", Nd4j.randn(DataType.FLOAT, 10, 10));
        sd.var("encoder.layer.0.bias", Nd4j.zeros(DataType.FLOAT, 10));
        sd.var("encoder.layer.1.weight", Nd4j.randn(DataType.FLOAT, 10, 10));
        sd.var("encoder.layer.1.bias", Nd4j.zeros(DataType.FLOAT, 10));
        sd.var("decoder.layer.0.weight", Nd4j.randn(DataType.FLOAT, 10, 10));
        sd.var("classifier.weight", Nd4j.randn(DataType.FLOAT, 10, 5));

        sd.freezeMatching("encoder\\.layer\\.[0-9]+\\..*");

        assertEquals(VariableType.CONSTANT, sd.getVariable("encoder.layer.0.weight").getVariableType());
        assertEquals(VariableType.CONSTANT, sd.getVariable("encoder.layer.0.bias").getVariableType());
        assertEquals(VariableType.CONSTANT, sd.getVariable("encoder.layer.1.weight").getVariableType());
        assertEquals(VariableType.CONSTANT, sd.getVariable("encoder.layer.1.bias").getVariableType());

        assertEquals(VariableType.VARIABLE, sd.getVariable("decoder.layer.0.weight").getVariableType());
        assertEquals(VariableType.VARIABLE, sd.getVariable("classifier.weight").getVariableType());
    }

    @Test
    @DisplayName("SameDiff - Freeze by prefix")
    public void testSameDiffFreezeByPrefix() {
        SameDiff sd = SameDiff.create();

        sd.var("bert.encoder.layer.0.weight", Nd4j.randn(DataType.FLOAT, 10, 10));
        sd.var("bert.encoder.layer.1.weight", Nd4j.randn(DataType.FLOAT, 10, 10));
        sd.var("bert.pooler.weight", Nd4j.randn(DataType.FLOAT, 10, 10));
        sd.var("classifier.weight", Nd4j.randn(DataType.FLOAT, 10, 5));

        sd.freezePrefix("bert.");

        assertEquals(VariableType.CONSTANT, sd.getVariable("bert.encoder.layer.0.weight").getVariableType());
        assertEquals(VariableType.CONSTANT, sd.getVariable("bert.encoder.layer.1.weight").getVariableType());
        assertEquals(VariableType.CONSTANT, sd.getVariable("bert.pooler.weight").getVariableType());
        assertEquals(VariableType.VARIABLE, sd.getVariable("classifier.weight").getVariableType());
    }

    @Test
    @DisplayName("SameDiff - Unfreeze variables")
    public void testSameDiffUnfreezeVariables() {
        SameDiff sd = SameDiff.create();

        sd.var("weight1", Nd4j.randn(DataType.FLOAT, 10, 10));
        sd.var("weight2", Nd4j.randn(DataType.FLOAT, 10, 10));

        sd.freezeVariables("weight1", "weight2");
        assertEquals(VariableType.CONSTANT, sd.getVariable("weight1").getVariableType());
        assertEquals(VariableType.CONSTANT, sd.getVariable("weight2").getVariableType());

        sd.unfreezeVariables("weight1");
        assertEquals(VariableType.VARIABLE, sd.getVariable("weight1").getVariableType());
        assertEquals(VariableType.CONSTANT, sd.getVariable("weight2").getVariableType());
    }

    // ==================== TransferLearning.Builder Tests ====================

    @Test
    @DisplayName("TransferLearning.Builder - Basic usage")
    public void testTransferLearningBuilderBasic() {
        SameDiff sd = createSimpleNetwork();

        SameDiff fineTuned = new TransferLearning.Builder(sd)
                .freezeVariables("layer1.weight", "layer1.bias")
                .copyModel(true)
                .build();

        assertEquals(VariableType.VARIABLE, sd.getVariable("layer1.weight").getVariableType());

        assertEquals(VariableType.CONSTANT, fineTuned.getVariable("layer1.weight").getVariableType());
        assertEquals(VariableType.CONSTANT, fineTuned.getVariable("layer1.bias").getVariableType());
        assertEquals(VariableType.VARIABLE, fineTuned.getVariable("layer2.weight").getVariableType());
    }

    @Test
    @DisplayName("TransferLearning.Builder - In place modification")
    public void testTransferLearningBuilderInPlace() {
        SameDiff sd = createSimpleNetwork();

        SameDiff modified = new TransferLearning.Builder(sd)
                .freezeVariables("layer1.weight")
                .copyModel(false)
                .build();

        assertSame(sd, modified);
        assertEquals(VariableType.CONSTANT, sd.getVariable("layer1.weight").getVariableType());
    }

    @Test
    @DisplayName("TransferLearning.Builder - With FineTuneConfiguration")
    public void testTransferLearningBuilderWithFineTuneConfig() {
        SameDiff sd = createSimpleNetwork();

        FineTuneConfiguration ftConfig = FineTuneConfiguration.builder()
                .updater(new Adam(1e-5))
                .freezePrefix("layer1")
                .l2(1e-5)
                .build();

        SameDiff fineTuned = new TransferLearning.Builder(sd)
                .fineTuneConfiguration(ftConfig)
                .build();

        assertEquals(VariableType.CONSTANT, fineTuned.getVariable("layer1.weight").getVariableType());
        assertEquals(VariableType.CONSTANT, fineTuned.getVariable("layer1.bias").getVariableType());
        assertEquals(VariableType.VARIABLE, fineTuned.getVariable("layer2.weight").getVariableType());

        assertNotNull(fineTuned.getTrainingConfig());
        assertTrue(fineTuned.getTrainingConfig() instanceof TransferLearning.FineTuneTrainingConfig);
    }

    @Test
    @DisplayName("TransferLearning.Builder - Reinitialize weights")
    public void testTransferLearningBuilderReinitialize() {
        SameDiff sd = SameDiff.create();
        sd.var("weight", Nd4j.ones(DataType.FLOAT, 5, 5));

        INDArray newWeights = Nd4j.randn(DataType.FLOAT, 5, 5);

        SameDiff modified = new TransferLearning.Builder(sd)
                .reinitialize("weight", newWeights)
                .build();

        INDArray modifiedWeights = modified.getVariable("weight").getArr();
        assertEquals(newWeights, modifiedWeights);
    }

    @Test
    @DisplayName("TransferLearning.Builder - Find upstream variables")
    public void testFindUpstreamVariables() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 10);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 10, 20));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 20, 10));
        SDVariable w3 = sd.var("w3", Nd4j.randn(DataType.FLOAT, 10, 5));

        SDVariable h1 = input.mmul(w1);
        h1.rename("h1");
        SDVariable h2 = h1.mmul(w2);
        h2.rename("h2");
        SDVariable output = h2.mmul(w3);
        output.rename("output");

        Set<String> upstreamH2 = TransferLearning.Builder.findUpstreamVariables(sd, "h2");
        assertTrue(upstreamH2.contains("w1"));
        assertTrue(upstreamH2.contains("w2"));
        assertFalse(upstreamH2.contains("w3"));

        Set<String> upstreamOutput = TransferLearning.Builder.findUpstreamVariables(sd, "output");
        assertTrue(upstreamOutput.contains("w1"));
        assertTrue(upstreamOutput.contains("w2"));
        assertTrue(upstreamOutput.contains("w3"));
    }

    // ==================== FineTuneTrainingConfig Tests ====================

    @Test
    @DisplayName("FineTuneTrainingConfig - isTrainable")
    public void testFineTuneTrainingConfigIsTrainable() {
        FineTuneConfiguration ftConfig = FineTuneConfiguration.builder()
                .freeze("frozen_var")
                .freezePrefix("encoder.")
                .build();

        Set<String> frozenSet = new HashSet<>();
        frozenSet.add("frozen_var");
        frozenSet.add("encoder.layer1");

        TransferLearning.FineTuneTrainingConfig config =
                new TransferLearning.FineTuneTrainingConfig(ftConfig, frozenSet);

        assertFalse(config.isTrainable("frozen_var"));
        assertFalse(config.isTrainable("encoder.layer1"));
        assertTrue(config.isTrainable("decoder.layer1"));
        assertTrue(config.isTrainable("classifier"));
    }

    @Test
    @DisplayName("FineTuneTrainingConfig - hasPerVariableConfig")
    public void testFineTuneTrainingConfigHasPerVariableConfig() {
        FineTuneConfiguration ftConfig = FineTuneConfiguration.builder()
                .updater(new Adam(1e-4))
                .build();

        TransferLearning.FineTuneTrainingConfig config =
                new TransferLearning.FineTuneTrainingConfig(ftConfig, new HashSet<>());

        assertTrue(config.hasPerVariableConfig());
    }

    @Test
    @DisplayName("FineTuneTrainingConfig - getUpdaterForVariable")
    public void testFineTuneTrainingConfigGetUpdaterForVariable() {
        FineTuneConfiguration ftConfig = FineTuneConfiguration.builder()
                .updater(new Adam(1e-4))
                .updater("special.weight", new Sgd(0.01))
                .build();

        TransferLearning.FineTuneTrainingConfig config =
                new TransferLearning.FineTuneTrainingConfig(ftConfig, new HashSet<>());

        IUpdater defaultUpdater = config.getUpdaterForVariable("regular.weight");
        IUpdater specialUpdater = config.getUpdaterForVariable("special.weight");

        assertTrue(defaultUpdater instanceof Adam);
        assertTrue(specialUpdater instanceof Sgd);
    }

    // ==================== TransferLearningHelper Tests ====================

    @Test
    @DisplayName("TransferLearningHelper - Basic initialization")
    public void testTransferLearningHelperBasic() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 10);
        SDVariable w1 = sd.var("encoder.weight", Nd4j.randn(DataType.FLOAT, 10, 20));
        SDVariable w2 = sd.var("classifier.weight", Nd4j.randn(DataType.FLOAT, 20, 5));

        SDVariable hidden = input.mmul(w1);
        hidden.rename("hidden");
        SDVariable output = hidden.mmul(w2);
        output.rename("output");

        TransferLearningHelper helper = new TransferLearningHelper(sd, "hidden");

        assertNotNull(helper);
        assertTrue(helper.getFrozenVariables().contains("encoder.weight"));
        assertEquals(1, helper.getFeatureOutputNames().size());
        assertTrue(helper.getFeatureOutputNames().contains("hidden"));
    }

    @Test
    @DisplayName("TransferLearningHelper - Featurize setup and structure")
    public void testTransferLearningHelperFeaturize() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 10);
        SDVariable w1 = sd.var("encoder.weight", Nd4j.randn(DataType.FLOAT, 10, 20));

        SDVariable hidden = input.mmul(w1);
        hidden.rename("hidden");

        // Test that helper is properly configured
        TransferLearningHelper helper = new TransferLearningHelper(sd, "hidden");

        assertNotNull(helper);
        assertEquals(1, helper.getFeatureOutputNames().size());
        assertTrue(helper.getFeatureOutputNames().contains("hidden"));
        assertTrue(helper.getFrozenVariables().contains("encoder.weight"));
        assertEquals(VariableType.CONSTANT, sd.getVariable("encoder.weight").getVariableType());

        // Test that we can run the original SameDiff model (before freezing changed it)
        // by using a fresh model for execution test
        SameDiff execSd = SameDiff.create();
        SDVariable execInput = execSd.placeHolder("input", DataType.FLOAT, -1, 10);
        SDVariable execW1 = execSd.var("encoder.weight", Nd4j.randn(DataType.FLOAT, 10, 20));
        SDVariable execHidden = execInput.mmul(execW1);
        execHidden.rename("hidden");

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input", Nd4j.randn(DataType.FLOAT, 5, 10));

        Map<String, INDArray> result = execSd.output(placeholders, "hidden");
        assertNotNull(result);
        assertTrue(result.containsKey("hidden"));
        assertArrayEquals(new long[]{5, 20}, result.get("hidden").shape());
    }

    @Test
    @DisplayName("TransferLearningHelper - Variable counts")
    public void testTransferLearningHelperCounts() {
        SameDiff sd = SameDiff.create();

        sd.var("frozen1", Nd4j.randn(DataType.FLOAT, 10, 10));
        sd.var("frozen2", Nd4j.randn(DataType.FLOAT, 10, 10));
        sd.var("trainable1", Nd4j.randn(DataType.FLOAT, 10, 10));

        sd.freezeVariables("frozen1", "frozen2");

        TransferLearningHelper helper = new TransferLearningHelper(sd);

        assertEquals(2, helper.getFrozenVariableCount());
        assertEquals(1, helper.getTrainableVariableCount());
    }

    // ==================== Integration Tests ====================

    @Test
    @DisplayName("Integration - Fine-tuning configuration and freeze verification")
    public void testSimpleFineTuningWorkflow() {
        // Test that TransferLearning properly configures the model for fine-tuning
        // Note: Actual training with frozen variables requires additional SameDiff support
        // for handling CONSTANT type variables during gradient function creation

        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 10);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 2);

        SDVariable w1 = sd.var("encoder.weight", Nd4j.randn(DataType.FLOAT, 10, 20).mul(0.1));
        SDVariable b1 = sd.var("encoder.bias", Nd4j.zeros(DataType.FLOAT, 20));
        SDVariable w2 = sd.var("classifier.weight", Nd4j.randn(DataType.FLOAT, 20, 2).mul(0.1));
        SDVariable b2 = sd.var("classifier.bias", Nd4j.zeros(DataType.FLOAT, 2));

        SDVariable hidden = sd.nn.relu(input.mmul(w1).add(b1), 0);
        SDVariable output = hidden.mmul(w2).add(b2);
        SDVariable loss = sd.loss.softmaxCrossEntropy("loss", label, output, null);

        // Verify initial state: all weights are VARIABLE type
        assertEquals(VariableType.VARIABLE, sd.getVariable("encoder.weight").getVariableType());
        assertEquals(VariableType.VARIABLE, sd.getVariable("encoder.bias").getVariableType());
        assertEquals(VariableType.VARIABLE, sd.getVariable("classifier.weight").getVariableType());
        assertEquals(VariableType.VARIABLE, sd.getVariable("classifier.bias").getVariableType());

        FineTuneConfiguration ftConfig = FineTuneConfiguration.builder()
                .updater(new Adam(1e-3))
                .freezePrefix("encoder")
                .l2(1e-5)
                .build();

        // Apply transfer learning configuration in-place
        SameDiff fineTuned = new TransferLearning.Builder(sd)
                .fineTuneConfiguration(ftConfig)
                .copyModel(false)
                .build();

        // Verify frozen variables are now CONSTANT type
        assertEquals(VariableType.CONSTANT, fineTuned.getVariable("encoder.weight").getVariableType());
        assertEquals(VariableType.CONSTANT, fineTuned.getVariable("encoder.bias").getVariableType());

        // Verify trainable variables remain VARIABLE type
        assertEquals(VariableType.VARIABLE, fineTuned.getVariable("classifier.weight").getVariableType());
        assertEquals(VariableType.VARIABLE, fineTuned.getVariable("classifier.bias").getVariableType());

        // Verify training config was set
        assertNotNull(fineTuned.getTrainingConfig());
        assertTrue(fineTuned.getTrainingConfig() instanceof TransferLearning.FineTuneTrainingConfig);

        // Verify training config reports correct trainability
        TransferLearning.FineTuneTrainingConfig trainingConfig =
                (TransferLearning.FineTuneTrainingConfig) fineTuned.getTrainingConfig();
        assertFalse(trainingConfig.isTrainable("encoder.weight"));
        assertFalse(trainingConfig.isTrainable("encoder.bias"));
        assertTrue(trainingConfig.isTrainable("classifier.weight"));
        assertTrue(trainingConfig.isTrainable("classifier.bias"));

        // Verify per-variable config is enabled
        assertTrue(trainingConfig.hasPerVariableConfig());

        // Verify updater is available for trainable variables
        IUpdater classifierUpdater = trainingConfig.getUpdaterForVariable("classifier.weight");
        assertNotNull(classifierUpdater);
        assertTrue(classifierUpdater instanceof Adam);

        // Verify regularization is available
        List<Regularization> reg = trainingConfig.getRegularizationForVariable("classifier.weight");
        assertNotNull(reg);
        assertFalse(reg.isEmpty());
    }

    // ==================== PEFT (Parameter-Efficient Fine-Tuning) Tests ====================

    @Test
    @DisplayName("PEFT - LoraConfig builder and validation")
    public void testLoraConfigBuilderAndValidation() {
        LoraConfig config = LoraConfig.builder()
                .r(16)
                .loraAlpha(32)
                .loraDropout(0.1)
                .targetModules(Arrays.asList("query", "key", "value"))
                .taskType(TaskType.CAUSAL_LM)
                .build();

        assertEquals(16, config.getR());
        assertEquals(32, config.getLoraAlpha());
        assertEquals(0.1, config.getLoraDropout(), 1e-6);
        assertEquals(2.0, config.getScaling(), 1e-6); // alpha/r = 32/16 = 2
        assertEquals(PeftType.LORA, config.getPeftType());
        assertEquals(TaskType.CAUSAL_LM, config.getTaskType());

        // Test validation
        config.validate();
    }

    @Test
    @DisplayName("PEFT - LoraConfig rsLoRA scaling")
    public void testLoraConfigRsLoraScaling() {
        LoraConfig config = LoraConfig.builder()
                .r(16)
                .loraAlpha(16)
                .useRsLora(true)
                .targetModules(Arrays.asList("query", "key"))
                .build();

        // rsLoRA: scaling = alpha / sqrt(r) = 16 / 4 = 4
        assertEquals(4.0, config.getScaling(), 1e-6);
    }

    @Test
    @DisplayName("PEFT - LoraConfig default configurations")
    public void testLoraConfigDefaults() {
        LoraConfig defaultTransformer = LoraConfig.defaultTransformer();
        assertEquals(16, defaultTransformer.getR());
        assertEquals(32, defaultTransformer.getLoraAlpha());
        assertNotNull(defaultTransformer.getTargetModules());

        LoraConfig minimal = LoraConfig.minimal();
        assertEquals(4, minimal.getR());

        LoraConfig allLinear = LoraConfig.allLinear(32);
        assertEquals(32, allLinear.getR());
        assertTrue(allLinear.getTargetModules().size() > 4);
    }

    @Test
    @DisplayName("PEFT - PromptTuningConfig builder")
    public void testPromptTuningConfigBuilder() {
        PromptTuningConfig config = PromptTuningConfig.builder()
                .numVirtualTokens(20)
                .tokenEmbeddingDim(768)
                .promptTuningInit(PromptTuningConfig.PromptTuningInit.RANDOM)
                .taskType(TaskType.SEQ_CLS)
                .build();

        assertEquals(20, config.getNumVirtualTokens());
        assertEquals(768, config.getTokenEmbeddingDim());
        assertEquals(PeftType.PROMPT_TUNING, config.getPeftType());

        // Test parameter count
        long trainableParams = config.calculateTrainableParameters(0);
        assertEquals(20 * 768, trainableParams);
    }

    @Test
    @DisplayName("PEFT - PrefixTuningConfig builder")
    public void testPrefixTuningConfigBuilder() {
        PrefixTuningConfig config = PrefixTuningConfig.builder()
                .numVirtualTokens(30)
                .numLayers(12)
                .hiddenSize(768)
                .prefixProjection(true)
                .encoderHiddenSize(384)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        assertEquals(30, config.getNumVirtualTokens());
        assertEquals(12, config.getNumLayers());
        assertEquals(768, config.getHiddenSize());
        assertTrue(config.isPrefixProjection());
        assertEquals(PeftType.PREFIX_TUNING, config.getPeftType());
    }

    @Test
    @DisplayName("PEFT - AdapterConfig builder")
    public void testAdapterConfigBuilder() {
        AdapterConfig config = AdapterConfig.builder()
                .adapterSize(64)
                .hiddenSize(768)
                .numLayers(12)
                .adapterActivation("relu")
                .adapterAfterAttention(true)
                .adapterAfterFeedforward(true)
                .taskType(TaskType.GENERAL)
                .build();

        assertEquals(64, config.getAdapterSize());
        assertEquals(768, config.getHiddenSize());
        assertEquals("relu", config.getAdapterActivation());
        assertEquals(PeftType.ADAPTERS, config.getPeftType());
    }

    @Test
    @DisplayName("PEFT - IA3Config builder")
    public void testIA3ConfigBuilder() {
        IA3Config config = IA3Config.builder()
                .targetModules(Arrays.asList("k_proj", "v_proj"))
                .feedforwardModules(Arrays.asList("up_proj"))
                .attentionHiddenSize(768)
                .feedforwardHiddenSize(3072)
                .initToOne(true)
                .taskType(TaskType.CAUSAL_LM)
                .build();

        assertEquals(PeftType.IA3, config.getPeftType());
        assertTrue(config.isInitToOne());
    }

    @Test
    @DisplayName("PEFT - PeftModel creation with LoRA")
    public void testPeftModelCreationWithLora() {
        SameDiff baseModel = createTransformerLikeNetwork();

        LoraConfig config = LoraConfig.builder()
                .r(8)
                .loraAlpha(16)
                .targetModules(Arrays.asList("query", "key", "value"))
                .taskType(TaskType.GENERAL)
                .build();

        PeftModel peftModel = PeftModel.fromPretrained(baseModel, config);

        assertNotNull(peftModel);
        assertNotNull(peftModel.getModel());
        assertEquals(PeftType.LORA, peftModel.getPeftConfig().getPeftType());

        // Trainable parameters should be less than total parameters
        long trainable = peftModel.getTrainableParameterCount();
        long total = peftModel.getTotalParameterCount();
        assertTrue(trainable < total, "Trainable params should be less than total");

        // Check summary output
        String summary = peftModel.getSummary();
        assertNotNull(summary);
        assertTrue(summary.contains("LoRA"));
    }

    @Test
    @DisplayName("PEFT - PeftModel creation with Prompt Tuning")
    public void testPeftModelCreationWithPromptTuning() {
        SameDiff baseModel = createTransformerLikeNetwork();

        PromptTuningConfig config = PromptTuningConfig.builder()
                .numVirtualTokens(20)
                .tokenEmbeddingDim(64)
                .promptTuningInit(PromptTuningConfig.PromptTuningInit.RANDOM)
                .taskType(TaskType.SEQ_CLS)
                .build();

        PeftModel peftModel = PeftModel.fromPretrained(baseModel, config);

        assertNotNull(peftModel);
        assertEquals(PeftType.PROMPT_TUNING, peftModel.getPeftConfig().getPeftType());

        // Check trainable percentage
        double trainablePercentage = peftModel.getTrainablePercentage();
        assertTrue(trainablePercentage > 0 && trainablePercentage < 100);
    }

    @Test
    @DisplayName("PEFT - TransferLearning.Builder with LoRA")
    public void testTransferLearningBuilderWithLora() {
        SameDiff baseModel = createTransformerLikeNetwork();

        PeftModel peftModel = new TransferLearning.Builder(baseModel)
                .lora(16, 32, Arrays.asList("query", "key", "value"))
                .buildPeft();

        assertNotNull(peftModel);
        assertTrue(peftModel.getPeftConfig() instanceof LoraConfig);

        LoraConfig loraConfig = (LoraConfig) peftModel.getPeftConfig();
        assertEquals(16, loraConfig.getR());
        assertEquals(32, loraConfig.getLoraAlpha());
    }

    @Test
    @DisplayName("PEFT - TransferLearning.Builder with Prompt Tuning")
    public void testTransferLearningBuilderWithPromptTuning() {
        SameDiff baseModel = createTransformerLikeNetwork();

        PeftModel peftModel = new TransferLearning.Builder(baseModel)
                .promptTuning(20, 64)
                .buildPeft();

        assertNotNull(peftModel);
        assertTrue(peftModel.getPeftConfig() instanceof PromptTuningConfig);
    }

    @Test
    @DisplayName("PEFT - TransferLearning.Builder with Prefix Tuning")
    public void testTransferLearningBuilderWithPrefixTuning() {
        SameDiff baseModel = createTransformerLikeNetwork();

        PeftModel peftModel = new TransferLearning.Builder(baseModel)
                .prefixTuning(30, 2, 64)
                .buildPeft();

        assertNotNull(peftModel);
        assertTrue(peftModel.getPeftConfig() instanceof PrefixTuningConfig);
    }

    @Test
    @DisplayName("PEFT - TransferLearning.Builder with Adapters")
    public void testTransferLearningBuilderWithAdapters() {
        SameDiff baseModel = createTransformerLikeNetwork();

        PeftModel peftModel = new TransferLearning.Builder(baseModel)
                .adapters(32, 64, 2)
                .buildPeft();

        assertNotNull(peftModel);
        assertTrue(peftModel.getPeftConfig() instanceof AdapterConfig);
    }

    @Test
    @DisplayName("PEFT - Static factory methods")
    public void testPeftStaticFactoryMethods() {
        SameDiff baseModel = createTransformerLikeNetwork();

        // Test static LoRA factory
        PeftModel loraModel = TransferLearning.loraModel(baseModel, 16,
                Arrays.asList("query", "key"));
        assertNotNull(loraModel);

        // Test prompt tuning factory
        PeftModel promptModel = TransferLearning.promptTuningModel(baseModel, 20, 64);
        assertNotNull(promptModel);

        // Test prefix tuning factory
        PeftModel prefixModel = TransferLearning.prefixTuningModel(baseModel, 30, 2, 64);
        assertNotNull(prefixModel);
    }

    @Test
    @DisplayName("PEFT - PeftModelFactory convenience methods")
    public void testPeftModelFactoryMethods() {
        SameDiff baseModel = createTransformerLikeNetwork();

        // Test factory methods
        PeftModel model1 = PeftModelFactory.createLoraModel(baseModel);
        assertNotNull(model1);

        PeftModel model2 = PeftModelFactory.createLoraModel(baseModel, 8);
        assertNotNull(model2);

        PeftModel model3 = PeftModelFactory.createMinimalLora(baseModel);
        assertNotNull(model3);
        LoraConfig minimalConfig = (LoraConfig) model3.getPeftConfig();
        assertEquals(4, minimalConfig.getR());
    }

    @Test
    @DisplayName("PEFT - printTrainableParameters")
    public void testPeftPrintTrainableParameters() {
        SameDiff baseModel = createTransformerLikeNetwork();

        PeftModel peftModel = PeftModelFactory.createLoraModel(baseModel, 8, 16,
                Arrays.asList("query"));

        // Just verify it doesn't throw
        peftModel.printTrainableParameters();

        // Check the values are sane
        assertTrue(peftModel.getTrainableParameterCount() > 0);
        assertTrue(peftModel.getTotalParameterCount() > 0);
        assertTrue(peftModel.getTrainablePercentage() > 0);
    }

    @Test
    @DisplayName("PEFT - TaskType enum coverage")
    public void testTaskTypeEnumCoverage() {
        // Ensure all task types are accessible
        assertEquals(14, TaskType.values().length);

        // Test a few specific ones
        assertNotNull(TaskType.CAUSAL_LM);
        assertNotNull(TaskType.SEQ_CLS);
        assertNotNull(TaskType.TOKEN_CLS);
        assertNotNull(TaskType.SEQ_2_SEQ_LM);
        assertNotNull(TaskType.QUESTION_ANS);
    }

    @Test
    @DisplayName("PEFT - PeftType enum coverage")
    public void testPeftTypeEnumCoverage() {
        // Ensure all PEFT types are accessible
        assertEquals(15, PeftType.values().length);

        assertNotNull(PeftType.LORA);
        assertNotNull(PeftType.PREFIX_TUNING);
        assertNotNull(PeftType.PROMPT_TUNING);
        assertNotNull(PeftType.P_TUNING);
        assertNotNull(PeftType.IA3);
        assertNotNull(PeftType.ADAPTERS);
        assertNotNull(PeftType.BITFIT);
        assertNotNull(PeftType.QLORA);
    }

    @Test
    @DisplayName("PEFT - LoraConfig validation failures")
    public void testLoraConfigValidationFailures() {
        // Invalid rank
        assertThrows(IllegalStateException.class, () -> {
            LoraConfig.builder()
                    .r(0)
                    .loraAlpha(16)
                    .targetModules(Arrays.asList("query"))
                    .build()
                    .validate();
        });

        // Invalid dropout
        assertThrows(IllegalStateException.class, () -> {
            LoraConfig.builder()
                    .r(8)
                    .loraAlpha(16)
                    .loraDropout(1.5)
                    .targetModules(Arrays.asList("query"))
                    .build()
                    .validate();
        });

        // No target modules
        assertThrows(IllegalStateException.class, () -> {
            LoraConfig.builder()
                    .r(8)
                    .loraAlpha(16)
                    .build()
                    .validate();
        });
    }

    // ==================== Helper Methods ====================

    private SameDiff createSimpleNetwork() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 10);
        SDVariable w1 = sd.var("layer1.weight", Nd4j.randn(DataType.FLOAT, 10, 20));
        SDVariable b1 = sd.var("layer1.bias", Nd4j.zeros(DataType.FLOAT, 20));
        SDVariable w2 = sd.var("layer2.weight", Nd4j.randn(DataType.FLOAT, 20, 5));
        SDVariable b2 = sd.var("layer2.bias", Nd4j.zeros(DataType.FLOAT, 5));

        SDVariable hidden = sd.nn.relu(input.mmul(w1).add(b1), 0);
        SDVariable output = hidden.mmul(w2).add(b2);
        output.rename("output");

        return sd;
    }

    private SameDiff createTransformerLikeNetwork() {
        SameDiff sd = SameDiff.create();

        // Create a simple network with transformer-like naming
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);

        // Attention weights (targets for LoRA)
        SDVariable queryWeight = sd.var("attention.query", Nd4j.randn(DataType.FLOAT, 64, 64));
        SDVariable keyWeight = sd.var("attention.key", Nd4j.randn(DataType.FLOAT, 64, 64));
        SDVariable valueWeight = sd.var("attention.value", Nd4j.randn(DataType.FLOAT, 64, 64));
        SDVariable outputWeight = sd.var("attention.output", Nd4j.randn(DataType.FLOAT, 64, 64));

        // Feed-forward weights
        SDVariable ff1 = sd.var("ff.up_proj", Nd4j.randn(DataType.FLOAT, 64, 256));
        SDVariable ff2 = sd.var("ff.down_proj", Nd4j.randn(DataType.FLOAT, 256, 64));

        // Classifier head
        SDVariable classifierWeight = sd.var("classifier.weight", Nd4j.randn(DataType.FLOAT, 64, 10));

        // Build the graph
        SDVariable q = input.mmul(queryWeight);
        SDVariable k = input.mmul(keyWeight);
        SDVariable v = input.mmul(valueWeight);

        // Simplified attention: softmax(Q @ K.T) @ V
        SDVariable attnScores = q.mmul(sd.transpose(k));
        SDVariable attnProbs = sd.nn.softmax(attnScores);
        SDVariable attnOutput = attnProbs.mmul(v);
        SDVariable projOutput = attnOutput.mmul(outputWeight);

        // Feed-forward
        SDVariable ffHidden = sd.nn.relu(projOutput.mmul(ff1), 0);
        SDVariable ffOutput = ffHidden.mmul(ff2);

        // Classifier
        SDVariable logits = ffOutput.mmul(classifierWeight);
        logits.rename("logits");

        return sd;
    }
}
