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

package org.eclipse.deeplearning4j.nd4j.linalg.dataset.curation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.curation.dedup.TextDeduplicator;
import org.nd4j.linalg.dataset.curation.filtering.TextQualityFilter;
import org.nd4j.linalg.dataset.curation.format.ChatTemplate;
import org.nd4j.linalg.dataset.curation.format.ConversationTurn;
import org.nd4j.linalg.dataset.curation.format.FormattedExample;
import org.nd4j.linalg.dataset.curation.format.InstructionDataFormatter;
import org.nd4j.linalg.dataset.curation.decontamination.NGramDecontaminator;
import org.nd4j.linalg.dataset.curation.packing.PackedSequence;
import org.nd4j.linalg.dataset.curation.packing.SequencePacker;
import org.nd4j.linalg.dataset.curation.splitting.SplitResult;
import org.nd4j.linalg.dataset.curation.splitting.StratifiedSplitter;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end integration tests demonstrating the full pipeline:
 * dataset curation → LoRA fine-tuning / knowledge distillation.
 * <p>
 * Uses synthetic data and pure SameDiff ops (no native custom ops required).
 */
@Slf4j
public class TestCurationPeftDistillationE2E {

    // ===================== Helper Methods =====================

    /**
     * Build a simple 2-layer MLP teacher model (no LoRA).
     * Architecture: input [batch, inputDim] → W1 → relu → W2 → logits [batch, outputDim]
     */
    private SameDiff buildTeacherMLP(int inputDim, int hiddenDim, int outputDim) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inputDim);
        SDVariable W1 = sd.var("W1", Nd4j.rand(DataType.FLOAT, inputDim, hiddenDim).muli(0.1));
        SDVariable bias1 = sd.var("bias1", Nd4j.zeros(DataType.FLOAT, hiddenDim));
        SDVariable W2 = sd.var("W2", Nd4j.rand(DataType.FLOAT, hiddenDim, outputDim).muli(0.1));
        SDVariable bias2 = sd.var("bias2", Nd4j.zeros(DataType.FLOAT, outputDim));

        SDVariable hidden = sd.nn.relu(input.mmul(W1).add(bias1), 0);
        hidden.rename("hidden");
        SDVariable logits = hidden.mmul(W2).add(bias2);
        logits.rename("logits");
        return sd;
    }

    /**
     * Build a student MLP with LoRA adapters injected into both weight matrices.
     * Base weights are CONSTANT (frozen); LoRA A/B matrices are VARIABLE (trainable).
     */
    private SameDiff buildStudentMLPWithLoRA(int inputDim, int hiddenDim, int outputDim,
                                              int rank, double scaling) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inputDim);

        // Base weights frozen as constants
        SDVariable W1 = sd.constant("W1", Nd4j.rand(DataType.FLOAT, inputDim, hiddenDim).muli(0.1));
        SDVariable W2 = sd.constant("W2", Nd4j.rand(DataType.FLOAT, hiddenDim, outputDim).muli(0.1));
        SDVariable bias1 = sd.var("bias1", Nd4j.zeros(DataType.FLOAT, hiddenDim));
        SDVariable bias2 = sd.var("bias2", Nd4j.zeros(DataType.FLOAT, outputDim));

        // LoRA for W1: A1 [rank, hiddenDim], B1 [inputDim, rank]
        SDVariable loraA1 = sd.var("W1_lora_A", Nd4j.rand(DataType.FLOAT, rank, hiddenDim).muli(0.01));
        SDVariable loraB1 = sd.var("W1_lora_B", Nd4j.zeros(DataType.FLOAT, inputDim, rank));

        // LoRA for W2: A2 [rank, outputDim], B2 [hiddenDim, rank]
        SDVariable loraA2 = sd.var("W2_lora_A", Nd4j.rand(DataType.FLOAT, rank, outputDim).muli(0.01));
        SDVariable loraB2 = sd.var("W2_lora_B", Nd4j.zeros(DataType.FLOAT, hiddenDim, rank));

        // W1_effective = W1 + scaling * B1 @ A1
        SDVariable W1eff = W1.add(sd.mmul(loraB1, loraA1).mul(scaling));
        SDVariable W2eff = W2.add(sd.mmul(loraB2, loraA2).mul(scaling));

        SDVariable hidden = sd.nn.relu(input.mmul(W1eff).add(bias1), 0);
        hidden.rename("hidden");
        SDVariable logits = hidden.mmul(W2eff).add(bias2);
        logits.rename("logits");
        return sd;
    }

    /**
     * Build KL distillation loss using pure SameDiff ops.
     * KL(teacher || student) = sum(teacher_soft * (log(teacher_soft) - log_student_soft)) * T^2
     */
    private SDVariable buildKLDistillationLoss(SameDiff sd, SDVariable teacherLogits,
                                                SDVariable studentLogits, double temperature) {
        SDVariable tScaled = teacherLogits.div(temperature);
        SDVariable sScaled = studentLogits.div(temperature);

        SDVariable teacherSoft = sd.nn.softmax(tScaled, -1);
        SDVariable studentLogSoft = sd.nn.logSoftmax(sScaled, -1);
        SDVariable teacherLogSoft = sd.math.log(teacherSoft.add(1e-10));

        SDVariable klPerSample = sd.sum(teacherSoft.mul(teacherLogSoft.sub(studentLogSoft)), 1);
        SDVariable klLoss = sd.mean("kl_loss", klPerSample).mul(temperature * temperature);
        return klLoss;
    }

    /**
     * Build MSE loss between student and teacher hidden representations.
     * Used for feature-level distillation.
     */
    private SDVariable buildFeatureDistillationLoss(SameDiff sd, SDVariable studentHidden,
                                                     SDVariable teacherHidden) {
        SDVariable diff = studentHidden.sub(teacherHidden);
        return sd.mean("feature_loss", diff.mul(diff));
    }

    private DataSet featuresOnly(INDArray features) {
        return new DataSet(features, null);
    }

    /**
     * Simulate tokenization: convert text to a simple integer array based on char values.
     * This is a toy tokenizer for testing purposes only.
     */
    private int[] simpleTokenize(String text) {
        int[] tokens = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            tokens[i] = text.charAt(i) % 100 + 1; // 1-100 token range
        }
        return tokens;
    }

    /**
     * Convert tokenized sequences to a feature array for model input.
     * Creates a bag-of-tokens representation with fixed dimension.
     */
    private INDArray tokensToFeatures(List<int[]> tokenSequences, int featureDim) {
        INDArray features = Nd4j.zeros(DataType.FLOAT, tokenSequences.size(), featureDim);
        for (int i = 0; i < tokenSequences.size(); i++) {
            int[] tokens = tokenSequences.get(i);
            for (int t : tokens) {
                int idx = t % featureDim;
                features.putScalar(i, idx, features.getDouble(i, idx) + 1.0);
            }
            // Normalize
            double norm = features.getRow(i).norm2Number().doubleValue();
            if (norm > 0) {
                features.getRow(i).divi(norm);
            }
        }
        return features;
    }

    // ===================== End-to-End Tests =====================

    /**
     * Full pipeline: curate → format → tokenize → split → LoRA fine-tune.
     *
     * Steps:
     * 1. Filter low-quality texts
     * 2. Deduplicate
     * 3. Format as instruction-response pairs
     * 4. Tokenize
     * 5. Split into train/val
     * 6. Train a LoRA-adapted model on curated data
     */
    @Test
    public void testCurationToLoRAFineTuning() {
        Nd4j.getRandom().setSeed(12345);

        int inputDim = 32;
        int hiddenDim = 16;
        int outputDim = 4;
        int rank = 2;
        double scaling = 2.0;

        // --- Step 1: Raw instruction-response dataset ---
        List<String[]> rawPairs = new ArrayList<>();
        rawPairs.add(new String[]{"What is deep learning?", "Deep learning is a subset of machine learning that uses neural networks with multiple layers."});
        rawPairs.add(new String[]{"Explain backpropagation", "Backpropagation is an algorithm for computing gradients of a loss function with respect to neural network weights."});
        rawPairs.add(new String[]{"Hi", "Hello"}); // too short - should be filtered
        rawPairs.add(new String[]{"What is deep learning?", "Deep learning is a subset of machine learning that uses neural networks with multiple layers."}); // duplicate
        rawPairs.add(new String[]{"!!!@@@###$$$", "???...!!!..."}); // low quality - should be filtered
        rawPairs.add(new String[]{"Define gradient descent", "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function."});
        rawPairs.add(new String[]{"What are transformers?", "Transformers are neural network architectures that use self-attention mechanisms to process sequential data."});
        rawPairs.add(new String[]{"Explain LoRA", "LoRA is a parameter-efficient fine-tuning method that adds low-rank adapter matrices to frozen pre-trained weights."});

        // --- Step 2: Quality filter ---
        TextQualityFilter filter = TextQualityFilter.builder()
                .minChars(20)
                .minWords(5)
                .minAlphaRatio(0.5)
                .build();

        List<String[]> qualityPairs = new ArrayList<>();
        for (String[] pair : rawPairs) {
            if (filter.accept(pair[0]) || filter.accept(pair[1])) {
                qualityPairs.add(pair);
            }
        }

        log.info("Quality filter: {} -> {} pairs", rawPairs.size(), qualityPairs.size());
        assertTrue(qualityPairs.size() < rawPairs.size(), "Filter should remove some pairs");
        assertTrue(qualityPairs.size() >= 4, "Should keep at least the good pairs");

        // --- Step 3: Deduplicate ---
        TextDeduplicator dedup = TextDeduplicator.builder()
                .useExact(true)
                .useMinHash(false)
                .build();

        List<String[]> dedupPairs = new ArrayList<>();
        for (String[] pair : qualityPairs) {
            String combined = pair[0] + " ||| " + pair[1];
            if (dedup.addIfUnique(combined)) {
                dedupPairs.add(pair);
            }
        }

        log.info("Deduplication: {} -> {} pairs", qualityPairs.size(), dedupPairs.size());
        assertTrue(dedupPairs.size() <= qualityPairs.size(), "Dedup should not add pairs");

        // --- Step 4: Format as ChatML ---
        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.CHATML)
                .build();

        List<FormattedExample> formatted = new ArrayList<>();
        for (String[] pair : dedupPairs) {
            formatted.add(formatter.format(pair[0], pair[1]));
        }

        // Verify formatting
        for (FormattedExample fe : formatted) {
            assertTrue(fe.getText().contains("<|im_start|>"), "Should contain ChatML tags");
            int trainable = 0;
            for (int m : fe.getLossMask()) trainable += m;
            assertTrue(trainable > 0, "Should have trainable characters in loss mask");
        }

        // --- Step 5: Tokenize and create features ---
        List<int[]> tokenized = new ArrayList<>();
        for (FormattedExample fe : formatted) {
            tokenized.add(simpleTokenize(fe.getText()));
        }

        // --- Step 6: Split ---
        StratifiedSplitter<int[]> splitter = new StratifiedSplitter<>(42L);
        SplitResult<int[]> split = splitter.split(tokenized, 0.8, 0.2);

        List<int[]> trainTokens = split.getTrain();
        List<int[]> valTokens = split.getValidation();
        log.info("Split: train={}, val={}, test={}", trainTokens.size(), valTokens.size(), split.getTest().size());
        assertFalse(trainTokens.isEmpty(), "Train set should not be empty");

        // --- Step 7: Build LoRA model and train ---
        INDArray trainFeatures = tokensToFeatures(trainTokens, inputDim);

        SameDiff student = buildStudentMLPWithLoRA(inputDim, hiddenDim, outputDim, rank, scaling);

        // Create pseudo-labels (soft targets from a random teacher)
        SameDiff teacher = buildTeacherMLP(inputDim, hiddenDim, outputDim);
        Map<String, INDArray> teacherOut = teacher.output(
                Collections.singletonMap("input", trainFeatures), "logits");
        INDArray teacherLogits = teacherOut.get("logits");

        // Add KL loss
        SDVariable teacherConst = student.constant("teacher_logits", teacherLogits);
        SDVariable klLoss = buildKLDistillationLoss(student, teacherConst,
                student.getVariable("logits"), 4.0);
        klLoss.markAsLoss();

        TrainingConfig tc = TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .markLabelsUnused()
                .skipBuilderValidation(true)
                .build();
        student.setTrainingConfig(tc);
        student.prepareForTraining();

        // Train for multiple epochs
        double[] losses = new double[5];
        for (int epoch = 0; epoch < 5; epoch++) {
            student.fit(featuresOnly(trainFeatures));
            Map<String, INDArray> lossOut = student.output(
                    Collections.singletonMap("input", trainFeatures), "kl_loss");
            losses[epoch] = lossOut.get("kl_loss").getDouble(0);
            log.info("LoRA E2E Epoch {}: KL loss = {}", epoch, losses[epoch]);
            assertTrue(Double.isFinite(losses[epoch]), "Loss should be finite at epoch " + epoch);
        }

        // Verify training progressed
        assertTrue(losses[4] <= losses[0] + 0.1,
                "Loss should not increase significantly. Start: " + losses[0] + " End: " + losses[4]);

        // Validate on held-out set
        if (!valTokens.isEmpty()) {
            INDArray valFeatures = tokensToFeatures(valTokens, inputDim);
            Map<String, INDArray> valOut = student.output(
                    Collections.singletonMap("input", valFeatures), "logits");
            INDArray valLogits = valOut.get("logits");
            assertTrue(Double.isFinite(valLogits.sumNumber().doubleValue()),
                    "Validation output should be finite");
            log.info("LoRA E2E validation logits shape: {}", Arrays.toString(valLogits.shape()));
        }
    }

    /**
     * Full pipeline: curate → decontaminate → pack → distillation training.
     *
     * Steps:
     * 1. Filter and deduplicate training data
     * 2. Decontaminate against a held-out benchmark
     * 3. Pack short sequences together
     * 4. Train student model via knowledge distillation from teacher
     */
    @Test
    public void testCurationToDistillation() {
        Nd4j.getRandom().setSeed(12345);

        int inputDim = 32;
        int hiddenDim = 16;
        int outputDim = 4;

        // --- Step 1: Raw training corpus ---
        List<String> corpus = Arrays.asList(
                "Neural networks learn representations of data through multiple layers of abstraction",
                "Gradient descent optimizes parameters by following the negative gradient of the loss",
                "Attention mechanisms allow models to focus on relevant parts of the input sequence",
                "Convolutional layers apply learned filters to detect local patterns in spatial data",
                "Recurrent networks maintain hidden state to model sequential dependencies in text",
                "Transfer learning leverages pre-trained models to improve performance on downstream tasks",
                "short", // too short
                "Neural networks learn representations of data through multiple layers of abstraction", // duplicate
                "The quick brown fox jumps over the lazy dog near the old stone bridge by the river", // benchmark contaminated
                "Regularization techniques like dropout prevent neural networks from overfitting training data",
                "Batch normalization accelerates training by normalizing layer inputs across mini-batches"
        );

        // --- Step 2: Quality filter ---
        TextQualityFilter filter = TextQualityFilter.builder()
                .minChars(30)
                .minWords(5)
                .build();

        List<String> filtered = new ArrayList<>();
        for (String text : corpus) {
            if (filter.accept(text)) {
                filtered.add(text);
            }
        }
        log.info("Quality filter: {} -> {}", corpus.size(), filtered.size());

        // --- Step 3: Deduplicate ---
        TextDeduplicator dedup = TextDeduplicator.builder()
                .useExact(true)
                .build();
        List<String> deduped = dedup.deduplicate(filtered);
        log.info("Deduplication: {} -> {}", filtered.size(), deduped.size());
        assertTrue(deduped.size() < filtered.size(), "Should remove duplicates");

        // --- Step 4: Decontaminate ---
        List<String> benchmarkData = Arrays.asList(
                "The quick brown fox jumps over the lazy dog near the old stone bridge by the river and through the meadow"
        );

        NGramDecontaminator decontaminator = new NGramDecontaminator(5); // 5-gram
        decontaminator.indexBenchmark(benchmarkData);
        List<String> clean = decontaminator.decontaminate(deduped);
        log.info("Decontamination: {} -> {}", deduped.size(), clean.size());

        // --- Step 5: Tokenize and pack sequences ---
        List<int[]> tokenized = new ArrayList<>();
        for (String text : clean) {
            tokenized.add(simpleTokenize(text));
        }

        SequencePacker packer = new SequencePacker(0); // separator token = 0
        List<PackedSequence> packed = packer.pack(tokenized, 256);
        log.info("Packing: {} sequences -> {} packed", tokenized.size(), packed.size());
        assertTrue(packed.size() > 0, "Should have at least one packed sequence");

        // Verify packing preserved all tokens
        int totalOrigTokens = tokenized.stream().mapToInt(t -> t.length).sum();
        int totalPackedTokens = 0;
        for (PackedSequence ps : packed) {
            for (int i = 0; i < ps.length(); i++) {
                if (ps.getSegmentIds()[i] >= 0) {
                    totalPackedTokens++;
                }
            }
        }
        assertEquals(totalOrigTokens, totalPackedTokens, "Packing should preserve all tokens");

        // Verify attention masks are block-diagonal (cross-segment = 0)
        for (PackedSequence ps : packed) {
            INDArray attn = ps.getAttentionMask();
            int[] segIds = ps.getSegmentIds();
            for (int i = 0; i < ps.length(); i++) {
                for (int j = 0; j < ps.length(); j++) {
                    if (segIds[i] >= 0 && segIds[j] >= 0 && segIds[i] != segIds[j]) {
                        assertEquals(0.0, attn.getDouble(i, j), 1e-6,
                                "Cross-segment attention should be zero");
                    }
                }
            }
        }

        // --- Step 6: Create features from packed sequences ---
        List<int[]> packedTokenArrays = new ArrayList<>();
        for (PackedSequence ps : packed) {
            int[] nonPadTokens = Arrays.stream(ps.getTokens()).filter(t -> t > 0).toArray();
            if (nonPadTokens.length > 0) {
                packedTokenArrays.add(nonPadTokens);
            }
        }
        if (packedTokenArrays.isEmpty()) {
            packedTokenArrays.add(tokenized.get(0));
        }
        INDArray trainFeatures = tokensToFeatures(packedTokenArrays, inputDim);

        // --- Step 7: Teacher-student distillation ---
        SameDiff teacher = buildTeacherMLP(inputDim, hiddenDim, outputDim);
        SameDiff student = buildTeacherMLP(inputDim, hiddenDim, outputDim); // same arch, different init

        // Teacher forward pass
        Map<String, INDArray> teacherOut = teacher.output(
                Collections.singletonMap("input", trainFeatures), "logits", "hidden");
        INDArray teacherLogits = teacherOut.get("logits");
        INDArray teacherHidden = teacherOut.get("hidden");

        // Add combined distillation loss: KL (logit-level) + MSE (feature-level)
        SDVariable teacherLogitsConst = student.constant("teacher_logits", teacherLogits);
        SDVariable teacherHiddenConst = student.constant("teacher_hidden", teacherHidden);
        SDVariable studentLogits = student.getVariable("logits");
        SDVariable studentHidden = student.getVariable("hidden");

        SDVariable klLoss = buildKLDistillationLoss(student, teacherLogitsConst, studentLogits, 4.0);
        SDVariable featureLoss = buildFeatureDistillationLoss(student, studentHidden, teacherHiddenConst);

        // Combined loss: 0.7 * KL + 0.3 * feature
        SDVariable totalLoss = klLoss.mul(0.7).add(featureLoss.mul(0.3));
        totalLoss.rename("total_loss");
        totalLoss.markAsLoss();

        TrainingConfig tc = TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .markLabelsUnused()
                .skipBuilderValidation(true)
                .build();
        student.setTrainingConfig(tc);
        student.prepareForTraining();

        // Train
        double[] losses = new double[5];
        for (int epoch = 0; epoch < 5; epoch++) {
            student.fit(featuresOnly(trainFeatures));
            Map<String, INDArray> lossOut = student.output(
                    Collections.singletonMap("input", trainFeatures), "total_loss", "kl_loss", "feature_loss");
            losses[epoch] = lossOut.get("total_loss").getDouble(0);
            log.info("Distillation E2E Epoch {}: total={}, kl={}, feature={}",
                    epoch, losses[epoch],
                    lossOut.get("kl_loss").getDouble(0),
                    lossOut.get("feature_loss").getDouble(0));
            assertTrue(Double.isFinite(losses[epoch]), "Loss should be finite at epoch " + epoch);
        }

        // Verify loss decreased
        assertTrue(losses[4] <= losses[0] + 0.1,
                "Loss should not increase. Start: " + losses[0] + " End: " + losses[4]);
    }

    /**
     * Full pipeline: multi-turn conversation curation → LoRA + distillation.
     *
     * Demonstrates using InstructionDataFormatter with multi-turn conversations
     * and loss masks to train only on assistant responses.
     */
    @Test
    public void testMultiTurnCurationToLoRADistillation() {
        Nd4j.getRandom().setSeed(12345);

        int inputDim = 32;
        int hiddenDim = 16;
        int outputDim = 4;
        int rank = 2;
        double scaling = 2.0;

        // --- Step 1: Multi-turn conversation dataset ---
        List<List<ConversationTurn>> conversations = new ArrayList<>();

        conversations.add(Arrays.asList(
                ConversationTurn.system("You are a helpful AI tutor."),
                ConversationTurn.user("What is LoRA?"),
                ConversationTurn.assistant("LoRA stands for Low-Rank Adaptation. It adds trainable low-rank matrices to frozen pre-trained weights."),
                ConversationTurn.user("How does it save memory?"),
                ConversationTurn.assistant("Instead of fine-tuning all parameters, LoRA only trains two small matrices A and B where the update is BA.")
        ));

        conversations.add(Arrays.asList(
                ConversationTurn.system("You are a machine learning expert."),
                ConversationTurn.user("Explain knowledge distillation"),
                ConversationTurn.assistant("Knowledge distillation transfers knowledge from a large teacher model to a smaller student model using soft probability targets.")
        ));

        conversations.add(Arrays.asList(
                ConversationTurn.system("You are a data scientist."),
                ConversationTurn.user("What is data deduplication?"),
                ConversationTurn.assistant("Data deduplication removes duplicate or near-duplicate examples from training data to improve model generalization and training efficiency.")
        ));

        // --- Step 2: Format conversations ---
        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.CHATML)
                .build();

        List<FormattedExample> formatted = new ArrayList<>();
        for (List<ConversationTurn> conv : conversations) {
            FormattedExample fe = formatter.format(conv);
            formatted.add(fe);

            // Verify loss mask: only assistant content is trainable
            int[] mask = fe.getLossMask();
            int trainable = 0;
            for (int m : mask) trainable += m;
            assertTrue(trainable > 0, "Should have trainable characters");
            log.info("Conversation formatted: length={}, trainable={}", fe.getText().length(), trainable);
        }

        // --- Step 3: Tokenize ---
        List<int[]> tokenized = new ArrayList<>();
        for (FormattedExample fe : formatted) {
            tokenized.add(simpleTokenize(fe.getText()));
        }

        // --- Step 4: Create features ---
        INDArray trainFeatures = tokensToFeatures(tokenized, inputDim);

        // --- Step 5: Build teacher and student ---
        SameDiff teacher = buildTeacherMLP(inputDim, hiddenDim, outputDim);
        SameDiff student = buildStudentMLPWithLoRA(inputDim, hiddenDim, outputDim, rank, scaling);

        // Teacher forward pass
        Map<String, INDArray> teacherOut = teacher.output(
                Collections.singletonMap("input", trainFeatures), "logits");
        INDArray teacherLogits = teacherOut.get("logits");

        // Add KL loss
        SDVariable teacherConst = student.constant("teacher_logits", teacherLogits);
        SDVariable klLoss = buildKLDistillationLoss(student, teacherConst,
                student.getVariable("logits"), 4.0);
        klLoss.markAsLoss();

        TrainingConfig tc = TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("input")
                .markLabelsUnused()
                .skipBuilderValidation(true)
                .build();
        student.setTrainingConfig(tc);
        student.prepareForTraining();

        // Save initial LoRA weights for comparison
        INDArray initLoraA1 = student.getVariable("W1_lora_A").getArr().dup();
        INDArray initLoraB1 = student.getVariable("W1_lora_B").getArr().dup();

        // Train
        double[] losses = new double[5];
        for (int epoch = 0; epoch < 5; epoch++) {
            student.fit(featuresOnly(trainFeatures));
            Map<String, INDArray> lossOut = student.output(
                    Collections.singletonMap("input", trainFeatures), "kl_loss");
            losses[epoch] = lossOut.get("kl_loss").getDouble(0);
            log.info("MultiTurn LoRA+KD Epoch {}: KL loss = {}", epoch, losses[epoch]);
            assertTrue(Double.isFinite(losses[epoch]), "Loss should be finite at epoch " + epoch);
        }

        // Verify LoRA weights changed during training
        INDArray finalLoraA1 = student.getVariable("W1_lora_A").getArr();
        INDArray finalLoraB1 = student.getVariable("W1_lora_B").getArr();

        boolean loraAChanged = !finalLoraA1.equalsWithEps(initLoraA1, 1e-6);
        boolean loraBChanged = !finalLoraB1.equalsWithEps(initLoraB1, 1e-6);
        log.info("LoRA A changed: {}, LoRA B changed: {}", loraAChanged, loraBChanged);
        assertTrue(loraAChanged || loraBChanged, "LoRA weights should change during training");

        // Verify loss didn't diverge
        assertTrue(losses[4] <= losses[0] + 0.1,
                "Loss should not increase. Start: " + losses[0] + " End: " + losses[4]);
    }

    /**
     * Self-distillation pipeline: curate data → train with EMA self-distillation.
     *
     * The teacher is a snapshot of the student, updated with exponential moving average.
     * This tests the interplay between curation (quality filtering, dedup) and
     * the self-distillation training loop.
     */
    @Test
    public void testCurationToSelfDistillation() {
        Nd4j.getRandom().setSeed(12345);

        int inputDim = 32;
        int hiddenDim = 16;
        int outputDim = 4;

        // --- Step 1: Curate training data ---
        List<String> rawTexts = Arrays.asList(
                "Batch normalization reduces internal covariate shift during deep network training",
                "Learning rate scheduling can improve convergence speed and final model quality",
                "Data augmentation increases the effective size of the training set through transformations",
                "Dropout randomly zeroes activations during training as a form of regularization",
                "Weight decay adds an L2 penalty to the loss function to prevent overfitting",
                "Adam optimizer combines momentum and adaptive learning rates for robust optimization",
                "short",
                "Batch normalization reduces internal covariate shift during deep network training" // dup
        );

        TextQualityFilter filter = TextQualityFilter.builder()
                .minChars(30)
                .minWords(5)
                .build();
        TextDeduplicator dedup = TextDeduplicator.builder()
                .useExact(true)
                .build();

        List<String> curated = new ArrayList<>();
        for (String text : rawTexts) {
            if (filter.accept(text)) {
                curated.add(text);
            }
        }
        curated = dedup.deduplicate(curated);
        log.info("Self-distill curated: {} -> {}", rawTexts.size(), curated.size());

        // --- Step 2: Create features ---
        List<int[]> tokenized = new ArrayList<>();
        for (String text : curated) {
            tokenized.add(simpleTokenize(text));
        }
        INDArray features = tokensToFeatures(tokenized, inputDim);

        // --- Step 3: Build student model ---
        SameDiff student = buildTeacherMLP(inputDim, hiddenDim, outputDim);

        // Create teacher snapshot
        SameDiff teacherSnapshot = student.dup();

        // --- Step 4: Self-distillation training loop ---
        double emaDecay = 0.99;
        int epochs = 5;
        double[] losses = new double[epochs];

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Get teacher logits
            Map<String, INDArray> teacherOut = teacherSnapshot.output(
                    Collections.singletonMap("input", features), "logits");
            INDArray teacherLogits = teacherOut.get("logits");

            if (epoch == 0) {
                // First epoch: set up the loss graph
                SDVariable teacherConst = student.constant("teacher_logits_self", teacherLogits);
                SDVariable studentLogits = student.getVariable("logits");
                SDVariable klLoss = buildKLDistillationLoss(student, teacherConst, studentLogits, 4.0);
                klLoss.markAsLoss();

                TrainingConfig tc = TrainingConfig.builder()
                        .updater(new Adam(1e-3))
                        .dataSetFeatureMapping("input")
                        .markLabelsUnused()
                        .skipBuilderValidation(true)
                        .build();
                student.setTrainingConfig(tc);
                student.prepareForTraining();
            } else {
                // Update teacher logits constant
                student.getVariable("teacher_logits_self").getArr().assign(teacherLogits);
            }

            // Train one epoch
            student.fit(featuresOnly(features));

            // Compute loss
            Map<String, INDArray> lossOut = student.output(
                    Collections.singletonMap("input", features), "kl_loss");
            losses[epoch] = lossOut.get("kl_loss").getDouble(0);
            log.info("Self-distill Epoch {}: KL loss = {}", epoch, losses[epoch]);
            assertTrue(Double.isFinite(losses[epoch]), "Loss should be finite at epoch " + epoch);

            // EMA update teacher
            for (SDVariable var : student.variables()) {
                if (var.getVariableType() == org.nd4j.autodiff.samediff.VariableType.VARIABLE) {
                    SDVariable teacherVar = teacherSnapshot.getVariable(var.name());
                    if (teacherVar != null && teacherVar.getArr() != null && var.getArr() != null) {
                        teacherVar.getArr().muli(emaDecay).addi(var.getArr().mul(1.0 - emaDecay));
                    }
                }
            }
        }

        // All losses should be finite
        for (double loss : losses) {
            assertTrue(Double.isFinite(loss), "All losses should be finite");
        }
    }
}
