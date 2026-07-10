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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LoraMatMul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MultiLoraMatmul;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Numerical validation for the LoRA op generalizations added alongside QLoRA:
 * <ul>
 *   <li>rank-3 {@code lora_matmul} (batched [B,S,in] activations)</li>
 *   <li>{@code multi_lora_matmul} + its new backward {@code multi_lora_matmul_bp}
 *       (per-row adapter selection for batched multi-adapter serving)</li>
 * </ul>
 * These paths were previously either rank-2-only or forward-only.
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("LoRA Generalization Op Validation")
public class TestLoraGeneralizationOps
        extends org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation.BaseOpValidation {

    @Override
    public long getTimeoutMilliseconds() { return 120_000L; }

    // ── rank-3 lora_matmul ────────────────────────────────────────────────────

    /** rank-3 [B,S,in] lora_matmul must equal the (known-good) rank-2 path applied to [B*S,in]. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("lora_matmul rank-3 forward == rank-2 on flattened [B*S,in]")
    public void testLoraMatMulRank3ForwardMatchesReshaped2D(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(11);
        int B = 3, S = 5, in = 8, out = 6, r = 2;
        double scaling = 1.5;

        INDArray input3d = Nd4j.rand(DataType.FLOAT, B, S, in).muli(0.1);
        INDArray weight  = Nd4j.rand(DataType.FLOAT, out, in).muli(0.1);   // [out,in]
        INDArray loraA   = Nd4j.rand(DataType.FLOAT, r, in).muli(0.1);     // [r,in]
        INDArray loraB   = Nd4j.rand(DataType.FLOAT, out, r).muli(0.1);    // [out,r]

        INDArray out3d = Nd4j.exec(new LoraMatMul(
            input3d.dup(), weight.dup(), loraA.dup(), loraB.dup(), scaling, true))[0];
        assertArrayEquals(new long[]{B, S, out}, out3d.shape(), "rank-3 output shape");

        // Reference: flatten to 2D, run the rank-2 path, reshape back
        INDArray input2d = input3d.reshape(B * S, in);
        INDArray out2d = Nd4j.exec(new LoraMatMul(
            input2d.dup(), weight.dup(), loraA.dup(), loraB.dup(), scaling, true))[0];
        INDArray ref = out2d.reshape(B, S, out);

        double maxDiff = ref.sub(out3d).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-4, "rank-3 lora_matmul must match flattened rank-2; maxDiff=" + maxDiff);
        log.info("lora_matmul rank-3 forward parity passed: maxDiff={}", maxDiff);
    }

    /** rank-3 lora_matmul backprop must produce correctly-shaped, finite gradients that
     *  agree with the flattened rank-2 gradients. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("lora_matmul rank-3 gradients == flattened rank-2 gradients")
    public void testLoraMatMulRank3Gradients(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12);
        int B = 2, S = 4, in = 8, out = 6, r = 3;
        double scaling = 1.0;

        INDArray input3d = Nd4j.rand(DataType.FLOAT, B, S, in).muli(0.1);
        INDArray weight  = Nd4j.rand(DataType.FLOAT, out, in).muli(0.1);
        INDArray loraAv  = Nd4j.rand(DataType.FLOAT, r, in).muli(0.1);
        INDArray loraBv  = Nd4j.rand(DataType.FLOAT, out, r).muli(0.1);

        // rank-3 graph
        SameDiff sd3 = SameDiff.create();
        SDVariable i3   = sd3.var("in", input3d.dup());
        SDVariable w3   = sd3.constant("w", weight.dup());
        SDVariable a3   = sd3.var("A", loraAv.dup());
        SDVariable b3   = sd3.var("B", loraBv.dup());
        SDVariable o3   = new LoraMatMul(sd3, i3, w3, a3, b3, scaling).outputVariable();
        sd3.setLossVariables(o3.sum().name());
        Map<String, INDArray> g3 = sd3.calculateGradients(null, a3.name(), b3.name());

        // rank-2 reference on flattened input
        SameDiff sd2 = SameDiff.create();
        SDVariable i2   = sd2.var("in", input3d.reshape(B * S, in));
        SDVariable w2   = sd2.constant("w", weight.dup());
        SDVariable a2   = sd2.var("A", loraAv.dup());
        SDVariable b2   = sd2.var("B", loraBv.dup());
        SDVariable o2   = new LoraMatMul(sd2, i2, w2, a2, b2, scaling).outputVariable();
        sd2.setLossVariables(o2.sum().name());
        Map<String, INDArray> g2 = sd2.calculateGradients(null, a2.name(), b2.name());

        double dA = g2.get(a2.name()).sub(g3.get(a3.name())).amaxNumber().doubleValue();
        double dB = g2.get(b2.name()).sub(g3.get(b3.name())).amaxNumber().doubleValue();
        assertTrue(dA < 1e-4, "rank-3 dLoraA must match flattened rank-2; maxDiff=" + dA);
        assertTrue(dB < 1e-4, "rank-3 dLoraB must match flattened rank-2; maxDiff=" + dB);
        log.info("lora_matmul rank-3 gradient parity passed: dLoraA={}, dLoraB={}", dA, dB);
    }

    // ── multi_lora_matmul (per-row adapter selection) ─────────────────────────

    /** multi_lora_matmul forward must match a per-row manual reference across mixed adapters. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("multi_lora_matmul forward == per-row manual reference")
    public void testMultiLoraForwardMatchesManualReference(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(21);
        int B = 4, I = 6, O = 5, R = 2, A = 2;
        double alpha = 1.5;

        INDArray input   = Nd4j.rand(DataType.FLOAT, B, I).muli(0.1);
        INDArray baseW   = Nd4j.rand(DataType.FLOAT, I, O).muli(0.1);   // [I,O]
        INDArray loraA   = Nd4j.rand(DataType.FLOAT, A, I, R).muli(0.1);
        INDArray loraB   = Nd4j.rand(DataType.FLOAT, A, R, O).muli(0.1);
        INDArray ids     = Nd4j.createFromArray(new long[]{0, 1, 0, 1}).castTo(DataType.INT64);

        INDArray fused = Nd4j.exec(new MultiLoraMatmul(
            input.dup(), baseW.dup(), loraA.dup(), loraB.dup(), ids.dup(), alpha))[0];
        assertArrayEquals(new long[]{B, O}, fused.shape());

        // Manual per-row reference
        INDArray ref = Nd4j.zeros(DataType.FLOAT, B, O);
        for (int i = 0; i < B; i++) {
            int a = (int) ids.getLong(i);
            INDArray xi = input.getRow(i).reshape(1, I);                 // [1,I]
            INDArray base = xi.mmul(baseW);                             // [1,O]
            INDArray Aa = loraA.get(NDArrayIndex.point(a), NDArrayIndex.all(), NDArrayIndex.all()); // [I,R]
            INDArray Ba = loraB.get(NDArrayIndex.point(a), NDArrayIndex.all(), NDArrayIndex.all()); // [R,O]
            INDArray delta = xi.mmul(Aa).mmul(Ba).mul(alpha);          // [1,O]
            ref.putRow(i, base.add(delta));
        }
        double maxDiff = ref.sub(fused).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-4, "multi_lora_matmul forward must match manual reference; maxDiff=" + maxDiff);
        log.info("multi_lora_matmul forward parity passed: maxDiff={}", maxDiff);
    }

    /** multi_lora_matmul_bp: with all rows on adapter 0, gradients must equal the hand-computed
     *  single-adapter gradients, and the UNUSED adapter's gradient must be zero. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("multi_lora_matmul_bp gradients (single active adapter) match closed form")
    public void testMultiLoraBackwardSingleAdapter(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(22);
        int B = 4, I = 6, O = 5, R = 2, A = 2;
        double alpha = 1.25;

        INDArray input = Nd4j.rand(DataType.FLOAT, B, I).muli(0.1);
        INDArray baseW = Nd4j.rand(DataType.FLOAT, I, O).muli(0.1);
        INDArray loraA = Nd4j.rand(DataType.FLOAT, A, I, R).muli(0.1);
        INDArray loraB = Nd4j.rand(DataType.FLOAT, A, R, O).muli(0.1);
        INDArray ids   = Nd4j.zeros(DataType.INT64, B);   // all rows use adapter 0

        SameDiff sd = SameDiff.create();
        SDVariable in  = sd.var("in", input.dup());
        SDVariable bw  = sd.constant("bw", baseW.dup());
        SDVariable la  = sd.var("A", loraA.dup());
        SDVariable lb  = sd.var("B", loraB.dup());
        SDVariable idv = sd.constant("ids", ids.dup());
        SDVariable out = new MultiLoraMatmul(sd, in, bw, la, lb, idv, alpha).outputVariable();
        sd.setLossVariables(out.sum().name());
        Map<String, INDArray> g = sd.calculateGradients(null, in.name(), la.name(), lb.name());

        // Closed form for loss = sum(out), all rows adapter 0 (A0=[I,R], B0=[R,O]):
        //   dOut = ones[B,O]
        //   dB0  = alpha * (input @ A0)^T @ dOut
        //   dA0  = alpha * input^T @ (dOut @ B0^T)
        //   dInput = dOut @ baseW^T + alpha * (dOut @ B0^T) @ A0^T
        INDArray A0 = loraA.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()); // [I,R]
        INDArray B0 = loraB.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()); // [R,O]
        INDArray dOut = Nd4j.ones(DataType.FLOAT, B, O);
        INDArray dB0ref = input.mmul(A0).transpose().mmul(dOut).mul(alpha);        // [R,O]
        INDArray dA0ref = input.transpose().mmul(dOut.mmul(B0.transpose())).mul(alpha); // [I,R]
        INDArray dInref = dOut.mmul(baseW.transpose())
            .add(dOut.mmul(B0.transpose()).mmul(A0.transpose()).mul(alpha));       // [B,I]

        INDArray dA = g.get(la.name());  // [A,I,R]
        INDArray dB = g.get(lb.name());  // [A,R,O]
        INDArray dIn = g.get(in.name()); // [B,I]

        INDArray dA0 = dA.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray dB0 = dB.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray dA1 = dA.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray dB1 = dB.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all());

        log.info("multi_lora bp: dA0.amax={} dA0ref.amax={} | dB0.amax={} dB0ref.amax={} | dIn.amax={} dInref.amax={} | dA1.amax={} dB1.amax={}",
            dA0.amaxNumber(), dA0ref.amaxNumber(), dB0.amaxNumber(), dB0ref.amaxNumber(),
            dIn.amaxNumber(), dInref.amaxNumber(), dA1.amaxNumber(), dB1.amaxNumber());

        assertTrue(dA0ref.sub(dA0).amaxNumber().doubleValue() < 1e-3, "dLoraA[0] mismatch");
        assertTrue(dB0ref.sub(dB0).amaxNumber().doubleValue() < 1e-3, "dLoraB[0] mismatch");
        assertTrue(dInref.sub(dIn).amaxNumber().doubleValue() < 1e-3, "dInput mismatch");
        assertTrue(dA1.amaxNumber().doubleValue() < 1e-6, "unused adapter dLoraA[1] must be zero");
        assertTrue(dB1.amaxNumber().doubleValue() < 1e-6, "unused adapter dLoraB[1] must be zero");
        log.info("multi_lora_matmul_bp closed-form gradient check passed (used adapter matches, unused adapter zero)");
    }
}
