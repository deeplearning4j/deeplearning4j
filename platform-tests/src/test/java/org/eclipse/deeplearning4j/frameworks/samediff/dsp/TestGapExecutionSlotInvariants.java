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
package org.eclipse.deeplearning4j.frameworks.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Regression test for gap-execution slot invariants.
 *
 * <p>When a DSP plan contains a "gap" (an op that is not DSP-executable and therefore
 * runs on the standard path in between two DSP segments), the slot cache state MUST
 * be consistent around the gap:</p>
 * <ul>
 *   <li>DSP output slots consumed by the gap must be materialized to real INDArrays.</li>
 *   <li>Gap outputs that feed the next DSP segment must be registered as fresh inputs.</li>
 *   <li>Running the same input twice — once with a gap, once without — must produce
 *       bit-for-bit identical outputs.</li>
 * </ul>
 *
 * <p>A failure here typically manifests as non-determinism: the same input produces
 * different outputs depending on whether a value-dependent op was in the middle of the
 * plan. This is exactly the class of bug that the "No DSP Array Invalidation" rule in
 * MEMORY.md warns against.</p>
 */
@Slf4j
@Tag("dsp")
@Tag("invariant")
public class TestGapExecutionSlotInvariants extends DspRegressionHarness {

    private static final int HIDDEN = 64;
    private static final int VOCAB  = 256;

    /** Plain DSP-friendly fixture (no gap op). */
    @Override
    protected SameDiff buildFixture() {
        return buildTinyDecodeFixture(HIDDEN, VOCAB);
    }

    /**
     * Fixture that deliberately inserts a value-dependent op between matmul and softmax.
     * Using {@code max} with a scalar is a simple stand-in: it's unlikely to be fused
     * with either neighbour and forces the slot cache to materialize between segments.
     */
    private SameDiff buildFixtureWithGap() {
        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.randn(DataType.FLOAT, HIDDEN, VOCAB);
        sd.constant("W", wArr);
        sd.placeHolder("x", DataType.FLOAT, 1, HIDDEN);
        SDVariable mm  = sd.mmul(sd.getVariable("x"), sd.getVariable("W"));
        // Value-dependent "gap" op: clip to [-1e9, 1e9]. Mathematically a no-op for
        // normal inputs but structurally forces a segment boundary.
        SDVariable gap = sd.math().max(mm, sd.constant(Nd4j.scalar(-1e9f)));
        SDVariable gap2 = sd.math().min(gap, sd.constant(Nd4j.scalar(1e9f)));
        sd.nn().softmax("logits", gap2, 1);
        return sd;
    }

    /**
     * Run identical inputs through the plain and gap fixtures and assert that the
     * outputs are bit-for-bit identical. Because the gap ops are mathematical no-ops
     * for the input range we use, any divergence indicates a slot-cache invariant
     * violation (stale output slot reused without re-zero, etc.).
     */
    @Test
    public void testGapExecutionMatchesNoGapExecution() {
        assumeTrue(isCudaAvailable(), "CUDA backend required for DSP gap execution invariant test");
        SameDiff noGap  = buildFixture();
        SameDiff withGap = buildFixtureWithGap();

        // Use a fixed seed-ish input so comparisons are reproducible.
        INDArray x = Nd4j.linspace(DataType.FLOAT, -1.0, 0.01, HIDDEN).reshape(1, HIDDEN);
        Map<String, INDArray> inputs = Collections.singletonMap("x", x);

        INDArray outNoGap  = noGap.output(inputs, "logits").get("logits");
        INDArray outWithGap = withGap.output(inputs, "logits").get("logits");

        // Both arrays must be exactly equal. We compare byte-by-byte rather than using
        // a tolerance because the gap ops are mathematical no-ops — any difference
        // indicates a slot-cache invariant violation.
        assertEquals(outNoGap, outWithGap,
                "Gap execution produced different output than no-gap execution. " +
                "This indicates a DSP slot-cache invariant violation.");
    }
}
