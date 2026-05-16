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

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproduces the SIGSEGV crash found in VLM benchmark:
 *
 * <pre>
 *   SIGSEGV in NDArray::e<float>(long) during reshape_no_copy
 *   called from GenerationPipeline.generateNative line 1779:
 *     embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup()
 * </pre>
 *
 * The crash happens AFTER a DSP prefill step when Java-side code does
 * .getRow().reshape() on a weight variable. The DSP execution may corrupt
 * the weight's DataBuffer state (constant marking, buffer validity).
 *
 * This test builds a standalone graph that mimics the VLM decode pipeline:
 * <ol>
 *   <li>Build a graph with an embedding lookup, attention-like compute, and
 *       KV-cache-like outputs.</li>
 *   <li>Run a prefill step via DSP (multiple outputs including KV caches).</li>
 *   <li>Close some outputs (logits), keep others (KV caches).</li>
 *   <li>Do a Java-side .getRow().reshape().dup() on the embedding weight.</li>
 *   <li>Run a second decode step with different shapes.</li>
 * </ol>
 *
 * If the DataBuffer is corrupted after DSP execution, step 4 crashes with
 * SIGSEGV in e<float>() during verbose logging.
 */
@Slf4j
@Tag("dsp")
@DisplayName("DSP post-prefill reshape crash — getRow().reshape() after DSP execution")
public class DspPostPrefillReshapeCrashTest {

    private SameDiff sd;

    @BeforeEach
    void setUp() {
        Nd4j.getMemoryManager().togglePeriodicGc(false);
    }

    @AfterEach
    void tearDown() {
        if (sd != null) {
            try { sd.close(); } catch (Throwable ignored) {}
            sd = null;
        }
    }

    /**
     * Minimal reproduction: after DSP execution with large graph, do
     * getRow().reshape() on a constant/variable weight. This is the
     * exact pattern that crashes in GenerationPipeline.
     */
    @ParameterizedTest(name = "postPrefillReshape[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON", "EMULATED_REPLAY"}, mode = org.junit.jupiter.params.provider.EnumSource.Mode.INCLUDE)
    @DisplayName("getRow().reshape().dup() on weight after DSP prefill must not crash")
    public void testPostPrefillReshapeCrash(GraphExecutionMode mode) {
        int vocabSize = 128;
        int hiddenSize = 64;
        int seqLen = 16;
        int numHeads = 4;
        int headDim = hiddenSize / numHeads;

        // Build a graph with embedding table as a sd.var (weight)
        sd = SameDiff.create();
        SDVariable embedTable = sd.var("embedTable",
                Nd4j.linspace(DataType.FLOAT, -0.1, 0.001, vocabSize * hiddenSize)
                        .reshape(vocabSize, hiddenSize));
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, seqLen, hiddenSize);
        SDVariable wq = sd.var("wq", Nd4j.linspace(DataType.FLOAT, -0.01, 0.0001, hiddenSize * hiddenSize)
                .reshape(hiddenSize, hiddenSize));
        SDVariable wk = sd.var("wk", Nd4j.linspace(DataType.FLOAT, 0.01, 0.0001, hiddenSize * hiddenSize)
                .reshape(hiddenSize, hiddenSize));

        // Flatten for projection
        SDVariable xFlat = sd.reshape("xFlat", x, seqLen, hiddenSize);
        SDVariable q = sd.mmul("q", xFlat, wq);
        SDVariable k = sd.mmul("k", xFlat, wk);

        // Reshape into heads: [seqLen, hiddenSize] -> [1, seqLen, numHeads, headDim]
        SDVariable qR = sd.reshape("qR", q, 1, seqLen, numHeads, headDim);
        SDVariable kR = sd.reshape("kR", k, 1, seqLen, numHeads, headDim);

        // Permute to [1, numHeads, seqLen, headDim]
        SDVariable qP = sd.permute("qP", qR, 0, 2, 1, 3);
        SDVariable kP = sd.permute("kP", kR, 0, 2, 1, 3);

        // Reshape for batched matmul
        SDVariable qBatch = sd.reshape("qBatch", qP, numHeads, seqLen, headDim);
        SDVariable kBatch = sd.reshape("kBatch", kP, numHeads, seqLen, headDim);

        // K transpose
        SDVariable kT = sd.permute("kT", kBatch, 0, 2, 1);

        // Scores
        SDVariable scores = sd.mmul("scores", qBatch, kT);
        SDVariable attn = sd.nn.softmax("attn", scores, -1);

        // Output — simulate logits and KV cache outputs
        SDVariable logits = sd.reshape("logits", attn, 1, numHeads * seqLen, seqLen);
        SDVariable kvKey = sd.reshape("kvKey", kBatch, 1, numHeads, seqLen, headDim);

        sd.setOutputs("logits", "kvKey");

        // Configure DSP
        if (mode == GraphExecutionMode.SLOT_BY_SLOT) {
            sd.setDspAutoCompileEnabled(false);
        }
        sd.setGraphExecutionMode(mode);

        // STEP 1: Run prefill — multiple iterations to advance DSP phases
        INDArray input = Nd4j.linspace(DataType.FLOAT, -0.05, 0.001, 1 * seqLen * hiddenSize)
                .reshape(1, seqLen, hiddenSize);

        Map<String, INDArray> outputs = null;
        for (int i = 0; i < 6; i++) {
            outputs = sd.output(
                    Map.of("x", input),
                    "logits", "kvKey");
        }

        // STEP 2: Close logits output (like GenerationPipeline line 1767)
        assertNotNull(outputs);
        INDArray logitsOut = outputs.get("logits");
        assertNotNull(logitsOut, "logits output should not be null");
        // Read a value first to verify it's valid
        float firstLogit = logitsOut.getFloat(0);
        log.info("First logit value: {}", firstLogit);
        logitsOut.close();

        // STEP 3: The critical test — do getRow().reshape().dup() on the
        // embedding table weight, exactly like GenerationPipeline line 1779:
        //   embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup()
        INDArray embedArr = sd.getVariable("embedTable").getArr();
        assertNotNull(embedArr, "embedTable array should not be null");
        assertFalse(embedArr.isEmpty(), "embedTable should not be empty");
        assertEquals(DataType.FLOAT, embedArr.dataType(), "embedTable should be FLOAT");

        // This is where the VLM benchmark crashes with SIGSEGV
        int tokenId = 42;  // arbitrary valid row
        INDArray row = embedArr.getRow(tokenId);
        assertNotNull(row, "getRow should return non-null");
        assertEquals(hiddenSize, row.length(), "row should have hiddenSize elements");

        // This is the exact crash line: .reshape(1, 1, hiddenSize).dup()
        INDArray reshaped = row.reshape(1, 1, hiddenSize);
        assertNotNull(reshaped, "reshape should return non-null");
        assertArrayEquals(new long[]{1, 1, hiddenSize}, reshaped.shape(),
                "reshaped shape should be [1, 1, hiddenSize]");

        INDArray duped = reshaped.dup();
        assertNotNull(duped, "dup should return non-null");

        // Verify the values are correct (not garbage/NaN/zero)
        float[] values = duped.data().asFloat();
        boolean allZero = true;
        boolean anyNaN = false;
        for (float v : values) {
            if (v != 0.0f) allZero = false;
            if (Float.isNaN(v)) anyNaN = true;
        }
        assertFalse(allZero, "duped embedding should not be all zeros");
        assertFalse(anyNaN, "duped embedding should not contain NaN");

        // Verify the values match what we expect from linspace initialization
        float expectedFirst = (float)(-0.1 + 0.001 * tokenId * hiddenSize);
        assertEquals(expectedFirst, values[0], 0.01,
                "First value should match linspace initialization for row " + tokenId);

        duped.close();
        log.info("Post-prefill reshape succeeded without crash");

        // STEP 4: Run a second decode step to verify nothing is corrupted
        Map<String, INDArray> decodeOutputs = sd.output(
                Map.of("x", input),
                "logits", "kvKey");
        assertNotNull(decodeOutputs.get("logits"), "decode logits should not be null");
        assertNotNull(decodeOutputs.get("kvKey"), "decode kvKey should not be null");
        log.info("Second decode step succeeded");
    }

    /**
     * Tests the padKvToStaticSize pattern: after DSP output, create a larger
     * buffer, copy KV into it, close the original. This mimics
     * GenerationPipeline lines 1726-1745.
     */
    @ParameterizedTest(name = "kvPadAndReshape[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON", "EMULATED_REPLAY"}, mode = org.junit.jupiter.params.provider.EnumSource.Mode.INCLUDE)
    @DisplayName("KV cache padding + getRow().reshape() must not corrupt buffers")
    public void testKvPadThenReshape(GraphExecutionMode mode) {
        int hiddenSize = 32;
        int seqLen = 8;
        int maxKvLen = seqLen + 10;

        sd = SameDiff.create();
        SDVariable embedTable = sd.var("embedTable",
                Nd4j.linspace(DataType.FLOAT, -0.05, 0.001, 64 * hiddenSize)
                        .reshape(64, hiddenSize));
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, seqLen, hiddenSize);
        SDVariable w = sd.var("w", Nd4j.linspace(DataType.FLOAT, -0.01, 0.001, hiddenSize * hiddenSize)
                .reshape(hiddenSize, hiddenSize));

        SDVariable xFlat = sd.reshape("xFlat", x, seqLen, hiddenSize);
        SDVariable h = sd.mmul("h", xFlat, w);
        SDVariable kvOut = sd.reshape("kvOut", h, 1, 1, seqLen, hiddenSize);
        SDVariable logits = sd.math.add("logits", h.sum("hSum"), 0.0);

        sd.setOutputs("logits", "kvOut");
        if (mode == GraphExecutionMode.SLOT_BY_SLOT) {
            sd.setDspAutoCompileEnabled(false);
        }
        sd.setGraphExecutionMode(mode);

        // Run prefill
        INDArray input = Nd4j.linspace(DataType.FLOAT, -0.02, 0.001, 1 * seqLen * hiddenSize)
                .reshape(1, seqLen, hiddenSize);

        Map<String, INDArray> outputs = null;
        for (int i = 0; i < 6; i++) {
            outputs = sd.output(Map.of("x", input), "logits", "kvOut");
        }

        // Pad KV to static size (like padKvToStaticSize)
        INDArray kvCache = outputs.get("kvOut");
        assertNotNull(kvCache);
        long[] kvShape = kvCache.shape();
        assertEquals(4, kvShape.length, "kvOut should be 4D [1, 1, seqLen, hiddenSize], got shape: "
                + java.util.Arrays.toString(kvShape));
        INDArray padded = Nd4j.zeros(DataType.FLOAT, kvShape[0], kvShape[1], maxKvLen, kvShape[3]);
        // Copy prefill KV into padded buffer
        padded.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, seqLen), NDArrayIndex.all())
                .assign(kvCache);

        // Close logits (like GenerationPipeline line 1767)
        outputs.get("logits").close();

        // Now do getRow().reshape().dup() on the embedding weight
        INDArray embedArr = sd.getVariable("embedTable").getArr();
        assertNotNull(embedArr);

        INDArray row = embedArr.getRow(5);
        INDArray reshaped = row.reshape(1, 1, hiddenSize);
        INDArray duped = reshaped.dup();

        assertNotNull(duped);
        assertFalse(Float.isNaN(duped.getFloat(0)), "Value should not be NaN");
        assertTrue(duped.length() == hiddenSize, "Should have hiddenSize elements");

        duped.close();
        padded.close();
    }

    /**
     * Tests multiple DSP outputs being closed, followed by weight access.
     * Stresses the close/GC path to see if closing DSP outputs corrupts
     * unrelated constant buffers.
     */
    @ParameterizedTest(name = "multiOutputClose[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON", "EMULATED_REPLAY"}, mode = org.junit.jupiter.params.provider.EnumSource.Mode.INCLUDE)
    @DisplayName("Closing multiple DSP outputs must not corrupt weight buffers")
    public void testMultiOutputCloseThenWeightAccess(GraphExecutionMode mode) {
        int hiddenSize = 32;
        int seqLen = 8;

        sd = SameDiff.create();
        SDVariable w1 = sd.var("w1", Nd4j.linspace(DataType.FLOAT, -0.05, 0.001, hiddenSize * hiddenSize)
                .reshape(hiddenSize, hiddenSize));
        SDVariable w2 = sd.var("w2", Nd4j.linspace(DataType.FLOAT, 0.01, 0.001, hiddenSize * hiddenSize)
                .reshape(hiddenSize, hiddenSize));
        SDVariable w3 = sd.var("w3", Nd4j.linspace(DataType.FLOAT, -0.02, 0.001, hiddenSize * 4)
                .reshape(hiddenSize, 4));
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, seqLen, hiddenSize);

        SDVariable xFlat = sd.reshape("xFlat", x, seqLen, hiddenSize);
        SDVariable h1 = sd.mmul("h1", xFlat, w1);
        SDVariable h2 = sd.mmul("h2", xFlat, w2);
        SDVariable out1 = sd.reshape("out1", h1, 1, seqLen, hiddenSize);
        SDVariable out2 = sd.reshape("out2", h2, 1, seqLen, hiddenSize);
        SDVariable out3 = sd.mmul("out3", h1, w3);

        sd.setOutputs("out1", "out2", "out3");
        if (mode == GraphExecutionMode.SLOT_BY_SLOT) {
            sd.setDspAutoCompileEnabled(false);
        }
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.linspace(DataType.FLOAT, -0.01, 0.001, 1 * seqLen * hiddenSize)
                .reshape(1, seqLen, hiddenSize);

        Map<String, INDArray> outputs = null;
        for (int i = 0; i < 6; i++) {
            outputs = sd.output(Map.of("x", input), "out1", "out2", "out3");
        }

        // Close ALL outputs — this is more aggressive than the VLM pattern
        for (INDArray arr : outputs.values()) {
            arr.close();
        }

        // Now access weights — should not crash
        for (String varName : new String[]{"w1", "w2", "w3"}) {
            INDArray weightArr = sd.getVariable(varName).getArr();
            assertNotNull(weightArr, varName + " should not be null after output close");
            assertFalse(weightArr.isEmpty(), varName + " should not be empty");

            // Do a reshape → dup — the exact operation that crashes in the VLM benchmark.
            // dup() forces a proper device-to-host sync + copy.
            long[] shape = weightArr.shape();
            INDArray reshaped = weightArr.reshape(1, shape[0], shape[1]);
            INDArray duped = reshaped.dup();
            assertNotNull(duped);

            // Verify the duped values are not garbage (NaN/Inf)
            float dupedFirst = duped.getFloat(0);
            assertFalse(Float.isNaN(dupedFirst), varName + " reshape.dup() value should not be NaN");
            assertFalse(Float.isInfinite(dupedFirst), varName + " reshape.dup() value should not be Inf");
            assertTrue(duped.length() == shape[0] * shape[1],
                    varName + " reshape.dup() should preserve total elements");
            duped.close();
        }

        // Run another execution — should still work
        Map<String, INDArray> outputs2 = sd.output(Map.of("x", input), "out1", "out2", "out3");
        assertNotNull(outputs2.get("out1"));
        assertNotNull(outputs2.get("out2"));
        assertNotNull(outputs2.get("out3"));
    }
}
