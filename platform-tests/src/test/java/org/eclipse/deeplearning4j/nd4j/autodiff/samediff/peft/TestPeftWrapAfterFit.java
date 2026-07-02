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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.peft;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for wrapping an ALREADY-TRAINED model with {@link PeftModel}.
 *
 * <p>The canonical LoRA workflow is: load/train a base model, fit it, THEN wrap it with
 * adapters and fit again. Historically only never-trained models were tested, and the
 * trained-then-wrapped path failed through a chain of stale-training-state bugs
 * (all fixed 2026-07-02):</p>
 * <ol>
 *   <li>{@code dup()} carried a stale "grad" sub-function that predates adapter
 *       injection — training the wrapped model referenced adapter variables missing
 *       from the stale backward graph. Fixed by {@code SameDiff.invalidateGradFunction()}
 *       called from {@code PeftModel.applyPeft()}.</li>
 *   <li>{@code createGradFunction} NPE'd on a null {@code prereqs} list for
 *       single-consumer variables.</li>
 *   <li>{@code PeftModel.replaceVariableUsages} rewired the op producing the effective
 *       weight onto itself ({@code W_eff = W_eff + delta}) — the execution DAG correctly
 *       rejected the cycle — and never maintained the {@code Variable.inputsForOp}
 *       reverse index that gradient construction traverses.</li>
 *   <li>{@code TrainingSession.tryDspTrainingIteration} threw on DAG rejection instead
 *       of honoring its return-false-to-fall-back contract.</li>
 *   <li>{@code dup()} carried {@code initializedTraining=true} + the old updater map, so
 *       adapter variables had no updater state ("No updater found for variable ...").
 *       Fixed by the updater top-up in {@code initializeTraining()} (and the fitHelper
 *       call gate removal that made the top-up reachable).</li>
 * </ol>
 *
 * <p>Each test builds a tiny MLP with HuggingFace-style parameter names so LoRA
 * target-module matching behaves as it does on imported models.</p>
 *
 * <p>Run:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dbackend.artifactId=nd4j-native \
 *       -Dtest=TestPeftWrapAfterFit 2&gt;&amp;1 | tee /tmp/peft-wrap-after-fit.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.TRAINING)
@Tag(TagNames.SAMEDIFF)
@DisplayName("PeftModel wrap-after-fit regression")
public class TestPeftWrapAfterFit {

    private static final int N_IN = 8;
    private static final int HIDDEN = 16;
    private static final int N_OUT = 4;
    private static final int BATCH = 4;

    /**
     * Tiny MLP with HF-style names. q_proj/v_proj are the LoRA targets; the output
     * head stays un-adapted so the frozen-base assertions cover both matched and
     * unmatched variables.
     */
    private static SameDiff buildBaseModel(long seed) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, N_IN);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, -1, N_OUT);

        SDVariable wq = sd.var("model.layers.0.self_attn.q_proj.weight",
                new XavierInitScheme('c', N_IN, HIDDEN), DataType.FLOAT, N_IN, HIDDEN);
        SDVariable wv = sd.var("model.layers.0.self_attn.v_proj.weight",
                new XavierInitScheme('c', N_IN, HIDDEN), DataType.FLOAT, N_IN, HIDDEN);
        SDVariable wOut = sd.var("head.weight",
                new XavierInitScheme('c', HIDDEN, N_OUT), DataType.FLOAT, HIDDEN, N_OUT);

        SDVariable h = sd.nn.tanh(input.mmul(wq).add(input.mmul(wv)));
        SDVariable pred = h.mmul("pred", wOut);
        SDVariable loss = sd.loss.meanSquaredError("loss", labels, pred, null);
        loss.markAsLoss();
        return sd;
    }

    private static DataSet regressionData(long seed) {
        Nd4j.getRandom().setSeed(seed);
        INDArray features = Nd4j.randn(DataType.FLOAT, BATCH, N_IN);
        INDArray labels = Nd4j.randn(DataType.FLOAT, BATCH, N_OUT).muli(0.1);
        return new DataSet(features, labels);
    }

    private static TrainingConfig trainingConfig(double lr) {
        return new TrainingConfig.Builder()
                .updater(new Adam(lr))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build();
    }

    private static LoraConfig loraConfig() {
        return LoraConfig.builder()
                .r(2)
                .loraAlpha(4)
                .loraDropout(0.0)
                .targetModules(Arrays.asList("q_proj", "v_proj"))
                .build();
    }

    @Test
    @DisplayName("fit -> wrap with LoRA -> fit trains adapters, freezes base, merges")
    public void testWrapAfterFitTrainsEndToEnd() {
        SameDiff base = buildBaseModel(12345);

        // Phase 1: train the BASE model first. This creates the grad function,
        // updater state and sets initializedTraining — the exact state dup() carries
        // into the PEFT wrap and that the historical bug chain choked on.
        base.setTrainingConfig(trainingConfig(1e-3));
        base.fit(new SingletonDataSetIterator(regressionData(777)), 2);

        // Phase 2: wrap the trained model.
        PeftModel peft = PeftModel.fromPretrained(base, loraConfig());
        assertTrue(peft.getTrainableParameterCount() > 0, "LoRA must inject trainable params");
        assertTrue(peft.getTrainableParameterCount() < peft.getTotalParameterCount(),
                "Adapter params must be a strict subset");

        String frozenProbe = "model.layers.0.self_attn.q_proj.weight";
        INDArray baseWeightBefore = peft.getModel().getVariable(frozenProbe).getArr().dup();
        String loraAName = frozenProbe + "_lora_A";
        String loraBName = frozenProbe + "_lora_B";
        assertTrue(peft.getModel().hasVariable(loraAName), "lora_A variable must exist after wrap");
        INDArray loraBBefore = peft.getModel().getVariable(loraBName).getArr().dup();
        assertEquals(0.0, loraBBefore.norm2Number().doubleValue(), 0.0,
                "lora_B initializes to zeros (identity adapter)");

        // Phase 3: fit the WRAPPED model. Pre-fix this threw, in order of the bug chain:
        // missing "<w>_lora_A" variable / prereqs NPE / "DAG validation failed: Cycle
        // detected" / "No updater found for variable "<w>_lora_A"".
        peft.setTrainingConfig(trainingConfig(1e-2));
        assertDoesNotThrow(() -> peft.fit(new SingletonDataSetIterator(regressionData(888)), 2),
                "fit() on a wrapped ALREADY-TRAINED model must train");

        // Adapters moved...
        INDArray loraBAfter = peft.getModel().getVariable(loraBName).getArr();
        assertTrue(loraBAfter.norm2Number().doubleValue() > 0.0,
                "lora_B must receive gradient updates");
        // ...the frozen base did not.
        INDArray baseWeightAfter = peft.getModel().getVariable(frozenProbe).getArr();
        assertEquals(0.0, baseWeightBefore.sub(baseWeightAfter).norm2Number().doubleValue(), 0.0,
                "Frozen base weight must not drift during adapter training");
        assertEquals(VariableType.CONSTANT,
                peft.getModel().getVariable(frozenProbe).getVariableType(),
                "Base weight must be frozen to CONSTANT");

        // Phase 4: merge produces a standalone model whose forward includes the delta.
        SameDiff merged = peft.mergeAndUnload();
        INDArray mergedWeight = merged.getVariable(frozenProbe).getArr();
        assertTrue(mergedWeight.sub(baseWeightBefore).norm2Number().doubleValue() > 0.0,
                "Merged weight must include the trained adapter delta");

        Map<String, INDArray> feed = Collections.singletonMap("input",
                Nd4j.randn(DataType.FLOAT, 1, N_IN));
        INDArray mergedOut = merged.output(feed, "pred").get("pred");
        assertFalse(mergedOut.isNaN().any(), "Merged model forward must be finite");
    }

    @Test
    @DisplayName("wrap-after-fit forward is identity before adapter training (B=0)")
    public void testWrapAfterFitIdentityBeforeTraining() {
        SameDiff base = buildBaseModel(4242);
        base.setTrainingConfig(trainingConfig(1e-3));
        base.fit(new SingletonDataSetIterator(regressionData(999)), 1);

        Map<String, INDArray> feed = Collections.singletonMap("input",
                Nd4j.randn(Nd4j.getRandom().getSeed(), new long[]{2, N_IN}).castTo(DataType.FLOAT));
        INDArray baseOut = base.output(feed, "pred").get("pred").dup();

        PeftModel peft = PeftModel.fromPretrained(base, loraConfig());
        INDArray wrappedOut = peft.output(feed, "pred").get("pred");

        // W_eff = W + B@A with B=0 must be exactly the base forward — this pins the
        // replaceVariableUsages rewiring (no self-cycle, consumers correctly moved).
        assertTrue(baseOut.equalsWithEps(wrappedOut, 1e-6),
                "Zero-initialized adapter must not change the forward pass");
    }

    @Test
    @DisplayName("fit -> wrap -> fit -> wrap AGAIN on the merged model (repeatability)")
    public void testMergedModelCanBeWrappedAndTrainedAgain() {
        SameDiff base = buildBaseModel(31337);
        base.setTrainingConfig(trainingConfig(1e-3));
        base.fit(new SingletonDataSetIterator(regressionData(111)), 1);

        PeftModel first = PeftModel.fromPretrained(base, loraConfig());
        first.setTrainingConfig(trainingConfig(1e-2));
        first.fit(new SingletonDataSetIterator(regressionData(222)), 1);
        SameDiff merged = first.mergeAndUnload();

        // The merged model has CONSTANT base weights; wrapping it again must promote
        // nothing unexpected and training must again only move the fresh adapters.
        PeftModel second = PeftModel.fromPretrained(merged, loraConfig());
        second.setTrainingConfig(trainingConfig(1e-2));
        assertDoesNotThrow(() -> second.fit(new SingletonDataSetIterator(regressionData(333)), 1),
                "Wrapping and training a previously-merged model must work");
    }
}
