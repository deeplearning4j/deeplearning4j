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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.architecture.ArchitectureRegistry;
import org.nd4j.ggml.architecture.ModelArchitecture;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Mamba2SSM;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SquaredReLU;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for new model ops (squared_relu, mamba2_ssm) and architecture handlers
 * (Nemotron, LFM2, OLMo, OpenELM, GPT-OSS, Phi, Mistral).
 */
@Slf4j
public class TestNewModelOps {

    // ========================================================================
    // squared_relu tests
    // ========================================================================

    @Test
    public void testSquaredReluForwardFloat() {
        INDArray input = Nd4j.createFromArray(new float[]{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
        INDArray output = Nd4j.exec(new SquaredReLU(input))[0];

        // max(0, x)^2: negatives -> 0, positives -> x^2
        INDArray expected = Nd4j.createFromArray(new float[]{0.0f, 0.0f, 0.0f, 1.0f, 4.0f, 9.0f});
        assertEquals(expected, output);
    }

    @Test
    public void testSquaredReluForwardDouble() {
        INDArray input = Nd4j.createFromArray(new double[]{-3.0, -0.5, 0.0, 0.5, 2.0});
        INDArray output = Nd4j.exec(new SquaredReLU(input))[0];

        INDArray expected = Nd4j.createFromArray(new double[]{0.0, 0.0, 0.0, 0.25, 4.0});
        assertEquals(expected, output);
    }

    @Test
    public void testSquaredReluSameDiff() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", Nd4j.createFromArray(new float[]{-1.0f, 0.0f, 1.0f, 2.0f}));
        SDVariable result = sd.nn.squaredRelu(x);

        INDArray output = sd.outputSingle(null, result.name());
        INDArray expected = Nd4j.createFromArray(new float[]{0.0f, 0.0f, 1.0f, 4.0f});
        assertEquals(expected, output);
    }

    @Test
    public void testSquaredReluMultidimensional() {
        INDArray input = Nd4j.createFromArray(new float[][]{
                {-1.0f, 2.0f, -3.0f},
                {4.0f, -5.0f, 6.0f}
        });
        INDArray output = Nd4j.exec(new SquaredReLU(input))[0];

        assertEquals(input.shape()[0], output.shape()[0]);
        assertEquals(input.shape()[1], output.shape()[1]);

        // Check specific values
        assertEquals(0.0f, output.getFloat(0, 0), 1e-6);  // -1 -> 0
        assertEquals(4.0f, output.getFloat(0, 1), 1e-6);  // 2 -> 4
        assertEquals(0.0f, output.getFloat(0, 2), 1e-6);  // -3 -> 0
        assertEquals(16.0f, output.getFloat(1, 0), 1e-6); // 4 -> 16
        assertEquals(0.0f, output.getFloat(1, 1), 1e-6);  // -5 -> 0
        assertEquals(36.0f, output.getFloat(1, 2), 1e-6); // 6 -> 36
    }

    // ========================================================================
    // mamba2_ssm tests
    // ========================================================================

    @Test
    public void testMamba2SsmForward() {
        int B = 1, L = 4, H = 2, P = 3, N = 2;
        int D = H * P;  // 6

        INDArray x = Nd4j.rand(DataType.FLOAT, B, L, D);
        INDArray A = Nd4j.rand(DataType.FLOAT, H).muli(-1);  // negative log-space
        INDArray bArr = Nd4j.rand(DataType.FLOAT, B, L, N);
        INDArray cArr = Nd4j.rand(DataType.FLOAT, B, L, N);
        INDArray dt = Nd4j.rand(DataType.FLOAT, B, L, H).addi(0.1f);  // positive dt

        INDArray[] outputs = Nd4j.exec(new Mamba2SSM(x, A, bArr, cArr, dt, H, P, N));

        assertNotNull(outputs);
        assertEquals(2, outputs.length);

        INDArray y = outputs[0];
        INDArray stateOut = outputs[1];

        // y shape: [B, L, D]
        assertArrayEquals(new long[]{B, L, D}, y.shape());
        // stateOut shape: [B, H, N, P]
        assertArrayEquals(new long[]{B, H, N, P}, stateOut.shape());

        // Values should be finite
        assertFalse(y.isNaN().any());
        assertFalse(y.isInfinite().any());
        assertFalse(stateOut.isNaN().any());
        assertFalse(stateOut.isInfinite().any());
    }

    @Test
    public void testMamba2SsmWithSkipConnection() {
        int B = 1, L = 2, H = 2, P = 2, N = 2;
        int D = H * P;

        INDArray x = Nd4j.ones(DataType.FLOAT, B, L, D);
        INDArray A = Nd4j.zeros(DataType.FLOAT, H);  // zero decay in log space -> exp(0) = 1
        INDArray bArr = Nd4j.ones(DataType.FLOAT, B, L, N);
        INDArray cArr = Nd4j.ones(DataType.FLOAT, B, L, N);
        INDArray dt = Nd4j.ones(DataType.FLOAT, B, L, H);
        INDArray dSkip = Nd4j.ones(DataType.FLOAT, H);  // skip connection D=1

        INDArray[] outputs = Nd4j.exec(new Mamba2SSM(x, A, bArr, cArr, dt, dSkip, null, H, P, N));

        assertNotNull(outputs);
        assertEquals(2, outputs.length);

        INDArray y = outputs[0];
        assertArrayEquals(new long[]{B, L, D}, y.shape());
        // With skip connection D=1, output should include D*x contribution
        assertFalse(y.isNaN().any());
    }

    @Test
    public void testMamba2SsmStatePropagation() {
        int B = 1, L = 1, H = 1, P = 2, N = 2;
        int D = H * P;

        INDArray x = Nd4j.ones(DataType.FLOAT, B, L, D);
        INDArray A = Nd4j.createFromArray(new float[]{-0.5f});  // some decay
        INDArray bArr = Nd4j.ones(DataType.FLOAT, B, L, N);
        INDArray cArr = Nd4j.ones(DataType.FLOAT, B, L, N);
        INDArray dt = Nd4j.ones(DataType.FLOAT, B, L, H);

        // First step: no initial state
        INDArray[] out1 = Nd4j.exec(new Mamba2SSM(x, A, bArr, cArr, dt, H, P, N));
        INDArray state1 = out1[1];

        // Second step: pass state from first step
        INDArray[] out2 = Nd4j.exec(new Mamba2SSM(x, A, bArr, cArr, dt, null, state1, H, P, N));
        INDArray state2 = out2[1];

        // State should differ between steps (decay + accumulation)
        assertFalse(state1.equals(state2), "States should differ due to accumulation");
    }

    @Test
    public void testMamba2SsmSameDiff() {
        SameDiff sd = SameDiff.create();
        int B = 1, L = 3, H = 2, P = 2, N = 2;
        int D = H * P;

        SDVariable x = sd.var("x", Nd4j.rand(DataType.FLOAT, B, L, D));
        SDVariable A = sd.var("A", Nd4j.rand(DataType.FLOAT, H).muli(-1));
        SDVariable bVar = sd.var("B", Nd4j.rand(DataType.FLOAT, B, L, N));
        SDVariable cVar = sd.var("C", Nd4j.rand(DataType.FLOAT, B, L, N));
        SDVariable dt = sd.var("dt", Nd4j.rand(DataType.FLOAT, B, L, H).addi(0.1f));

        SDVariable[] results = sd.nn.mamba2Ssm(x, A, bVar, cVar, dt, H, P, N);
        assertNotNull(results);
        assertEquals(2, results.length);
    }

    // ========================================================================
    // Architecture registry tests
    // ========================================================================

    @ParameterizedTest
    @ValueSource(strings = {"nemotron", "nemotron_h", "nemotron_h_moe"})
    public void testNemotronArchitectureRegistered(String variant) {
        assertTrue(ArchitectureRegistry.hasArchitecture(variant),
                "Architecture '" + variant + "' should be registered");
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(variant);
        assertNotNull(arch);
        assertEquals("nemotron", arch.getName());
    }

    @ParameterizedTest
    @ValueSource(strings = {"lfm2", "lfm2moe"})
    public void testLFM2ArchitectureRegistered(String variant) {
        assertTrue(ArchitectureRegistry.hasArchitecture(variant),
                "Architecture '" + variant + "' should be registered");
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(variant);
        assertNotNull(arch);
        assertEquals("lfm2", arch.getName());
    }

    @ParameterizedTest
    @ValueSource(strings = {"olmo", "olmo2"})
    public void testOLMoArchitectureRegistered(String variant) {
        assertTrue(ArchitectureRegistry.hasArchitecture(variant),
                "Architecture '" + variant + "' should be registered");
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(variant);
        assertNotNull(arch);
        assertEquals("olmo", arch.getName());
    }

    @Test
    public void testOpenELMArchitectureRegistered() {
        assertTrue(ArchitectureRegistry.hasArchitecture("openelm"),
                "Architecture 'openelm' should be registered");
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("openelm");
        assertNotNull(arch);
        assertEquals("openelm", arch.getName());
    }

    @ParameterizedTest
    @ValueSource(strings = {"gpt-oss", "gptoss"})
    public void testGptOssArchitectureRegistered(String variant) {
        assertTrue(ArchitectureRegistry.hasArchitecture(variant),
                "Architecture '" + variant + "' should be registered");
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(variant);
        assertNotNull(arch);
        assertEquals("gpt-oss", arch.getName());
    }

    @ParameterizedTest
    @ValueSource(strings = {"phi", "phi2", "phi3", "phi3.5", "phi4"})
    public void testPhiArchitectureRegistered(String variant) {
        assertTrue(ArchitectureRegistry.hasArchitecture(variant),
                "Architecture '" + variant + "' should be registered");
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(variant);
        assertNotNull(arch);
        assertEquals("phi", arch.getName());
    }

    @ParameterizedTest
    @ValueSource(strings = {"mistral", "mixtral", "codestral", "pixtral", "mistral_nemo"})
    public void testMistralArchitectureRegistered(String variant) {
        assertTrue(ArchitectureRegistry.hasArchitecture(variant),
                "Architecture '" + variant + "' should be registered");
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture(variant);
        assertNotNull(arch);
        assertEquals("mistral", arch.getName());
    }

    @Test
    public void testMistralNotInLlamaVariants() {
        // Mistral and Mixtral should be handled by MistralArchitecture, not LLaMA
        ModelArchitecture mistralArch = ArchitectureRegistry.getArchitecture("mistral");
        ModelArchitecture llamaArch = ArchitectureRegistry.getArchitecture("llama");
        assertNotEquals(mistralArch, llamaArch,
                "Mistral should have its own architecture handler, not fall back to LLaMA");
    }

    @Test
    public void testAllNewArchitecturesPresent() {
        var registered = ArchitectureRegistry.getRegisteredArchitectures();
        // Verify all new architectures are present
        assertTrue(registered.contains("nemotron"), "nemotron missing");
        assertTrue(registered.contains("lfm2"), "lfm2 missing");
        assertTrue(registered.contains("olmo"), "olmo missing");
        assertTrue(registered.contains("openelm"), "openelm missing");
        assertTrue(registered.contains("gpt-oss"), "gpt-oss missing");
        assertTrue(registered.contains("phi"), "phi missing");
        assertTrue(registered.contains("mistral"), "mistral missing");
        // Original architectures still present
        assertTrue(registered.contains("llama"), "llama missing");
        assertTrue(registered.contains("gemma"), "gemma missing");
        assertTrue(registered.contains("generic"), "generic missing");
    }
}
