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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.AwqMatmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ColumnParallelLinear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DecoderMaskedMha;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Fp8Matmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGemmSwiglu;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedNormQuantize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GpuTopKSample;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GpuTopPSample;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MoeGate;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MultiLoraMatmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RowParallelLinear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SelectiveScan;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SmoothQuant;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for TRT-LLM feature parity ops using the SameDiff API.
 *
 * Each test builds a SameDiff graph with the op, executes it, and verifies
 * output shapes and basic correctness (non-null, expected dimensions).
 *
 * These ops correspond to TensorRT-LLM kernel fusions and quantized inference
 * patterns: FP8 matmul, SmoothQuant, AWQ, GPU sampling, decoder MHA,
 * SwiGLU, fused norm+quantize, MoE gating, selective scan (Mamba),
 * multi-LoRA, and tensor-parallel linear layers.
 */
public class TRTLLMFeatureParityOpsTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    private boolean isCudaBackend() {
        return Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda");
    }

    @Test
    public void testFp8Matmul() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        // FP8 matmul: INT8 tensors as FP8 stand-ins with per-tensor scales
        // C = (scaleA * A_fp8) @ (scaleB * B_fp8)
        SameDiff sd = SameDiff.create();

        int M = 4, K = 8, N = 6;
        // FP8 quantized stored as INT8
        SDVariable a = sd.var("a", Nd4j.ones(DataType.INT8, M, K));
        SDVariable b = sd.var("b", Nd4j.ones(DataType.INT8, K, N));
        SDVariable scaleA = sd.var("scaleA", Nd4j.scalar(DataType.FLOAT, 0.1f));
        SDVariable scaleB = sd.var("scaleB", Nd4j.scalar(DataType.FLOAT, 0.2f));

        SDVariable output = new Fp8Matmul(sd, a, b, scaleA, scaleB).outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result, "FP8 matmul output should not be null");
        assertArrayEquals(new long[]{M, N}, result.shape(),
                "FP8 matmul output shape should be [M, N]");
    }

    @Test
    public void testSmoothQuant() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        // SmoothQuant W8A8: Y = deq(quant(X * diag(s)^-1) @ quant(diag(s) * W))
        SameDiff sd = SameDiff.create();

        int batch = 2, inFeatures = 16, outFeatures = 8;
        SDVariable x = sd.var("x", Nd4j.rand(DataType.FLOAT, batch, inFeatures));
        SDVariable wQuantized = sd.var("wQuantized", Nd4j.ones(DataType.INT8, outFeatures, inFeatures));
        SDVariable smoothScale = sd.var("smoothScale", Nd4j.ones(DataType.FLOAT, inFeatures));
        SDVariable actScale = sd.var("actScale", Nd4j.scalar(DataType.FLOAT, 1.0f));
        SDVariable weightScale = sd.var("weightScale", Nd4j.ones(DataType.FLOAT, outFeatures));

        SDVariable output = new SmoothQuant(
                sd, x, wQuantized, smoothScale, actScale, weightScale).outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result, "SmoothQuant output should not be null");
        assertArrayEquals(new long[]{batch, outFeatures}, result.shape(),
                "SmoothQuant output shape should be [batch, outFeatures]");
    }

    @Test
    public void testAwqMatmul() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        // AWQ: output = input @ dequant(weightPacked, scales, zeros)
        SameDiff sd = SameDiff.create();

        int M = 2, K = 32, N = 16;
        int groupSize = 16;
        // packed weights: [K/2, N] since 4-bit packing
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, M, K));
        SDVariable weightPacked = sd.var("weightPacked", Nd4j.ones(DataType.INT8, K / 2, N));
        SDVariable scales = sd.var("scales", Nd4j.ones(DataType.FLOAT, K / groupSize, N));
        SDVariable zeros = sd.var("zeros", Nd4j.zeros(DataType.FLOAT, K / groupSize, N));

        SDVariable output = new AwqMatmul(sd, input, weightPacked, scales, zeros,
                null, groupSize, 4).outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result, "AWQ matmul output should not be null");
        assertArrayEquals(new long[]{M, N}, result.shape(),
                "AWQ matmul output shape should be [M, N]");
    }

    @Test
    public void testGpuTopKSample() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        // GPU top-K sampling: logits -> top-K filter -> softmax -> multinomial
        SameDiff sd = SameDiff.create();

        int batch = 1, vocabSize = 100;
        SDVariable logits = sd.var("logits", Nd4j.rand(DataType.FLOAT, batch, vocabSize));

        int k = 10;
        GpuTopKSample op = new GpuTopKSample(sd, logits, k, 1.0, 42);
        SDVariable[] outputs = op.outputVariables();

        Map<String, INDArray> results = sd.outputAll(null);

        // Output 0: sampled token IDs [batch]
        INDArray tokenIds = results.get(outputs[0].name());
        assertNotNull(tokenIds, "Top-K sampled token IDs should not be null");
        assertEquals(1, tokenIds.rank(), "Token IDs should be rank 1");
        assertEquals(batch, tokenIds.shape()[0], "Token IDs batch dimension should match");

        // Output 1: sampled probabilities [batch]
        INDArray probs = results.get(outputs[1].name());
        assertNotNull(probs, "Top-K sampled probabilities should not be null");
        assertEquals(batch, probs.shape()[0], "Probabilities batch dimension should match");
    }

    @Test
    public void testGpuTopPSample() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        // GPU top-P (nucleus) sampling: logits -> softmax -> sort -> cumsum -> nucleus filter -> sample
        SameDiff sd = SameDiff.create();

        int batch = 1, vocabSize = 100;
        SDVariable logits = sd.var("logits", Nd4j.rand(DataType.FLOAT, batch, vocabSize));

        double p = 0.9;
        GpuTopPSample op = new GpuTopPSample(sd, logits, p, 1.0, 42);
        SDVariable[] outputs = op.outputVariables();

        Map<String, INDArray> results = sd.outputAll(null);

        // Output 0: sampled token IDs [batch]
        INDArray tokenIds = results.get(outputs[0].name());
        assertNotNull(tokenIds, "Top-P sampled token IDs should not be null");
        assertEquals(1, tokenIds.rank(), "Token IDs should be rank 1");
        assertEquals(batch, tokenIds.shape()[0], "Token IDs batch dimension should match");

        // Output 1: sampled probabilities [batch]
        INDArray probs = results.get(outputs[1].name());
        assertNotNull(probs, "Top-P sampled probabilities should not be null");
        assertEquals(batch, probs.shape()[0], "Probabilities batch dimension should match");
    }

    @Test
    public void testDecoderMaskedMha() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        // Decoder masked multi-head attention with KV cache
        // Inputs: hidden [B,1,H], QKV weight [H,3H], out weight [H,H], past KV
        SameDiff sd = SameDiff.create();

        int B = 1, H = 64, numHeads = 4, headDim = H / numHeads; // headDim=16
        int pastSeqLen = 5;

        SDVariable hidden = sd.var("hidden", Nd4j.rand(DataType.FLOAT, B, 1, H));
        SDVariable qkvWeight = sd.var("qkvWeight", Nd4j.rand(DataType.FLOAT, H, 3 * H));
        SDVariable outWeight = sd.var("outWeight", Nd4j.rand(DataType.FLOAT, H, H));
        SDVariable pastKey = sd.var("pastKey", Nd4j.rand(DataType.FLOAT, B, numHeads, pastSeqLen, headDim));
        SDVariable pastValue = sd.var("pastValue", Nd4j.rand(DataType.FLOAT, B, numHeads, pastSeqLen, headDim));

        DecoderMaskedMha op = new DecoderMaskedMha(sd, hidden, qkvWeight, outWeight, pastKey, pastValue,
                numHeads, numHeads, headDim, 0, 10000, -3.4028235e+38f);
        SDVariable[] outputs = op.outputVariables();

        Map<String, INDArray> results = sd.outputAll(null);

        // Output 0: attention output [B, 1, H]
        INDArray attnOutput = results.get(outputs[0].name());
        assertNotNull(attnOutput, "Decoder MHA output should not be null");
        assertArrayEquals(new long[]{B, 1, H}, attnOutput.shape(),
                "Attention output shape should be [B, 1, H]");

        // Output 1: present key [B, heads, seq+1, dim]
        INDArray presentKey = results.get(outputs[1].name());
        assertNotNull(presentKey, "Present key should not be null");
        assertEquals(pastSeqLen + 1, presentKey.shape()[2],
                "Present key seq length should be pastSeqLen + 1");

        // Output 2: present value [B, heads, seq+1, dim]
        INDArray presentValue = results.get(outputs[2].name());
        assertNotNull(presentValue, "Present value should not be null");
        assertEquals(pastSeqLen + 1, presentValue.shape()[2],
                "Present value seq length should be pastSeqLen + 1");
    }

    @Test
    public void testFusedGemmSwiglu() {
        // Fused GEMM + SwiGLU: output = (input @ wGate) * silu(input @ wUp)
        SameDiff sd = SameDiff.create();

        int M = 2, K = 32, N = 16;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, M, K));
        SDVariable wGate = sd.var("wGate", Nd4j.rand(DataType.FLOAT, K, N));
        SDVariable wUp = sd.var("wUp", Nd4j.rand(DataType.FLOAT, K, N));

        SDVariable output = new FusedGemmSwiglu(sd, input, wGate, wUp).outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result, "FusedGemmSwiglu output should not be null");
        assertArrayEquals(new long[]{M, N}, result.shape(),
                "FusedGemmSwiglu output shape should be [M, N]");
    }

    @Test
    public void testFusedNormQuantize() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        // Fused normalization + quantization: RMSNorm -> INT8 quantize
        SameDiff sd = SameDiff.create();

        int B = 2, H = 64;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, B, H));
        SDVariable weight = sd.var("weight", Nd4j.ones(DataType.FLOAT, H));

        FusedNormQuantize op = new FusedNormQuantize(sd, input, weight);
        SDVariable[] outputs = op.outputVariables();

        Map<String, INDArray> results = sd.outputAll(null);

        // Output 0: quantized [B, H] INT8
        INDArray quantized = results.get(outputs[0].name());
        assertNotNull(quantized, "FusedNormQuantize quantized output should not be null");
        assertArrayEquals(new long[]{B, H}, quantized.shape(),
                "Quantized output shape should be [B, H]");
        assertEquals(DataType.INT8, quantized.dataType(),
                "Quantized output should be INT8");

        // Output 1: scale [B] FLOAT32
        INDArray scale = results.get(outputs[1].name());
        assertNotNull(scale, "FusedNormQuantize scale output should not be null");
        assertEquals(DataType.FLOAT, scale.dataType(),
                "Scale output should be FLOAT");
    }

    @Test
    public void testMoeGate() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        // MoE gating: top-K expert routing with load-balancing loss
        SameDiff sd = SameDiff.create();

        int T = 4, H = 32, E = 8;
        int topK = 2;
        SDVariable hidden = sd.var("hidden", Nd4j.rand(DataType.FLOAT, T, H));
        SDVariable gateWeight = sd.var("gateWeight", Nd4j.rand(DataType.FLOAT, H, E));

        MoeGate op = new MoeGate(sd, hidden, gateWeight, topK, E, 0.01f);
        SDVariable[] outputs = op.outputVariables();

        Map<String, INDArray> results = sd.outputAll(null);

        // Output 0: expert indices [T, K] INT64
        INDArray expertIndices = results.get(outputs[0].name());
        assertNotNull(expertIndices, "MoE expert indices should not be null");
        assertArrayEquals(new long[]{T, topK}, expertIndices.shape(),
                "Expert indices shape should be [T, topK]");
        assertEquals(DataType.INT64, expertIndices.dataType(),
                "Expert indices should be INT64");

        // Output 1: gate weights [T, K]
        INDArray gateWeights = results.get(outputs[1].name());
        assertNotNull(gateWeights, "MoE gate weights should not be null");
        assertArrayEquals(new long[]{T, topK}, gateWeights.shape(),
                "Gate weights shape should be [T, topK]");

        // Output 2: aux loss [1] scalar
        INDArray auxLoss = results.get(outputs[2].name());
        assertNotNull(auxLoss, "MoE auxiliary loss should not be null");
    }

    @Test
    public void testSelectiveScan() {
        // Selective scan (Mamba S6): h_t = A_t * h_{t-1} + B_t * x_t; y_t = C_t * h_t + D * x_t
        SameDiff sd = SameDiff.create();

        int B = 1, L = 10, D = 16, S = 4;
        SDVariable x = sd.var("x", Nd4j.rand(DataType.FLOAT, B, L, D));
        SDVariable a = sd.var("a", Nd4j.rand(DataType.FLOAT, B, L, S));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, B, L, S));
        SDVariable c = sd.var("c", Nd4j.rand(DataType.FLOAT, B, L, S));
        SDVariable d = sd.var("d", Nd4j.rand(DataType.FLOAT, D));

        SDVariable output = new SelectiveScan(sd, x, a, b, c, d).outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result, "Selective scan output should not be null");
        assertArrayEquals(new long[]{B, L, D}, result.shape(),
                "Selective scan output shape should be [B, L, D]");
    }

    @Test
    public void testMultiLoraMatmul() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        // Multi-LoRA: output[i] = input[i] @ baseWeight + alpha * input[i] @ loraA[id] @ loraB[id]
        SameDiff sd = SameDiff.create();

        int batchSize = 3, I = 32, O = 16, numAdapters = 2, R = 4;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, batchSize, I));
        SDVariable baseWeight = sd.var("baseWeight", Nd4j.rand(DataType.FLOAT, I, O));
        SDVariable loraA = sd.var("loraA", Nd4j.rand(DataType.FLOAT, numAdapters, I, R));
        SDVariable loraB = sd.var("loraB", Nd4j.rand(DataType.FLOAT, numAdapters, R, O));
        // Adapter IDs: assign samples to adapters 0, 1, 0
        SDVariable adapterIds = sd.var("adapterIds", Nd4j.createFromArray(new long[]{0, 1, 0}));

        SDVariable output = new MultiLoraMatmul(sd, input, baseWeight, loraA, loraB, adapterIds, 1.0f)
                .outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result, "Multi-LoRA output should not be null");
        assertArrayEquals(new long[]{batchSize, O}, result.shape(),
                "Multi-LoRA output shape should be [batch, O]");
    }

    @Test
    public void testColumnParallelLinear() {
        // Column-parallel: output_shard = input @ weightShard
        // With tpSize=1, this is just a standard matmul
        SameDiff sd = SameDiff.create();

        int B = 2, I = 32, O_shard = 8;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, B, I));
        SDVariable weightShard = sd.var("weightShard", Nd4j.rand(DataType.FLOAT, I, O_shard));

        SDVariable output = new ColumnParallelLinear(sd, input, weightShard).outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result, "Column-parallel linear output should not be null");
        // With tpSize=1 and gatherOutput=1 (default), output is [B, O_shard]
        assertArrayEquals(new long[]{B, O_shard}, result.shape(),
                "Column-parallel linear output shape should be [B, O/tp]");
    }

    @Test
    public void testRowParallelLinear() {
        // Row-parallel: partial = inputShard @ weightShard
        // With tpSize=1, this is just a standard matmul
        SameDiff sd = SameDiff.create();

        int B = 2, I_shard = 8, O = 16;
        SDVariable inputShard = sd.var("inputShard", Nd4j.rand(DataType.FLOAT, B, I_shard));
        SDVariable weightShard = sd.var("weightShard", Nd4j.rand(DataType.FLOAT, I_shard, O));

        SDVariable output = new RowParallelLinear(sd, inputShard, weightShard).outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result, "Row-parallel linear output should not be null");
        assertArrayEquals(new long[]{B, O}, result.shape(),
                "Row-parallel linear output shape should be [B, O]");
    }

    @Test
    public void testFP8DataType() {
        // Verify DataType.FLOAT8 enum exists and has correct properties
        DataType fp8 = DataType.FLOAT8;
        assertNotNull(fp8, "DataType.FLOAT8 should exist");

        // toInt() should return 2
        assertEquals(2, fp8.toInt(), "FLOAT8 toInt() should return 2");

        // fromInt(2) should return FLOAT8
        assertEquals(DataType.FLOAT8, DataType.fromInt(2),
                "DataType.fromInt(2) should return FLOAT8");

        // isFPType should be true
        assertTrue(fp8.isFPType(), "FLOAT8 should be a floating point type");

        // width should be 1 byte
        assertEquals(1, fp8.width(), "FLOAT8 width should be 1 byte");

        // FP8 alias should work
        assertEquals(DataType.FLOAT8, DataType.FP8, "DataType.FP8 alias should equal FLOAT8");
    }
}
