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
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for SameDiff/DSP teardown ordering.
 *
 * <p>Root cause pinned 2026-07-02 (hs_err-confirmed): {@code GenerationPipeline.close()}
 * called {@code SameDiffMemoryUtils.freeModelArrays(decoder)} BEFORE
 * {@code decoder.close()}. The native DSP plan teardown inside {@code close()}
 * ({@code NativeDynamicShapePlan::releaseGpuIntermediates}) walks slot NDArrays whose
 * buffers reference the model's DataBuffers; freeing those buffers first leaves the
 * teardown reading freed heap — corruption then clobbered a function-local
 * {@code unordered_set<int>} and crashed in {@code _M_find_before_node} (SIGSEGV).
 * The fix swaps the order: close the SameDiff (native plan release) FIRST, then free
 * the model arrays (idempotent — already-closed buffers are skipped).</p>
 *
 * <p>These tests pin the SAFE order at the SameDiff level on a DSP-warmed graph.
 * The unsafe order (free-then-close) is documented in a {@link Disabled} test below:
 * it is a use-after-free, so it crashes the JVM nondeterministically rather than
 * failing an assertion — it can only become a live test once the native teardown is
 * hardened to tolerate closed buffers (defense-in-depth follow-up).</p>
 *
 * <p>Run:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dbackend.artifactId=nd4j-native \
 *       -Dtest=DspTeardownOrderTest 2&gt;&amp;1 | tee /tmp/dsp-teardown-order.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP teardown ordering regression")
public class DspTeardownOrderTest {

    /** Small graph with a weight VARIABLE so the plan holds slots referencing model buffers. */
    private static SameDiff buildAndWarm() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 64);
        SDVariable w = sd.var("weight", Nd4j.randn(DataType.FLOAT, 64, 64).muli(0.1));
        SDVariable b = sd.var("bias", Nd4j.zeros(DataType.FLOAT, 64));
        sd.nn.tanh("out", x.mmul(w).add(b));

        // Multiple executions at a stable shape: warm the DSP plan past compilation so
        // teardown has real native state (slot arrays, plan handle) to release.
        Map<String, INDArray> feed = Collections.singletonMap("x", Nd4j.ones(DataType.FLOAT, 2, 64));
        for (int i = 0; i < 3; i++) {
            INDArray out = sd.output(feed, "out").get("out");
            assertFalse(out.isNaN().any(), "warmup forward must be finite");
        }
        return sd;
    }

    @Test
    @DisplayName("SAFE order: close() then freeModelArrays() — no crash, idempotent")
    public void testCloseThenFreeModelArrays() {
        SameDiff sd = buildAndWarm();
        assertDoesNotThrow(() -> {
            sd.close();                              // native plan teardown with buffers intact
            SameDiffMemoryUtils.freeModelArrays(sd); // then free: skips already-closed buffers
        }, "close() followed by freeModelArrays() is the supported teardown order");
    }

    @Test
    @DisplayName("SAFE order is repeat-safe: double close + double free")
    public void testTeardownIdempotence() {
        SameDiff sd = buildAndWarm();
        assertDoesNotThrow(() -> {
            sd.close();
            SameDiffMemoryUtils.freeModelArrays(sd);
            // Second round must be a no-op, not a double-free.
            sd.close();
            SameDiffMemoryUtils.freeModelArrays(sd);
        }, "Repeated teardown must be a no-op");
    }

    @Test
    @DisplayName("Teardown after teardown of a SIBLING model does not cross-contaminate")
    public void testIndependentModelsTearDownIndependently() {
        // Two DSP-warmed models; destroying one must not poison the other's plan
        // (guards the cross-plan shared-state class of teardown bugs).
        SameDiff a = buildAndWarm();
        SameDiff b = buildAndWarm();

        assertDoesNotThrow(() -> {
            a.close();
            SameDiffMemoryUtils.freeModelArrays(a);
        });

        // b must still execute after a's full teardown...
        Map<String, INDArray> feed = Collections.singletonMap("x", Nd4j.ones(DataType.FLOAT, 2, 64));
        INDArray out = b.output(feed, "out").get("out");
        assertFalse(out.isNaN().any(), "Sibling model must stay executable after a's teardown");

        // ...and then tear down cleanly itself.
        assertDoesNotThrow(() -> {
            b.close();
            SameDiffMemoryUtils.freeModelArrays(b);
        });
    }

    /**
     * The UNSAFE order that caused the original SIGSEGV. It is a native use-after-free:
     * when it goes wrong it kills the JVM (uncatchable), and because UAF is timing/heap
     * dependent it can also pass silently — one survival is NOT evidence of safety
     * (the GraphOptimizer example survived the same path that killed the distillation
     * example 2-for-2). Enable only after {@code releaseGpuIntermediates} is hardened
     * to skip slots whose DataBuffers are already closed.
     */
    @Test
    @Disabled("free-before-close is a native UAF (SIGSEGV, not an exception) until releaseGpuIntermediates tolerates closed buffers")
    @DisplayName("UNSAFE order: freeModelArrays() then close() — requires native hardening")
    public void testFreeModelArraysBeforeCloseRequiresHardening() {
        SameDiff sd = buildAndWarm();
        assertDoesNotThrow(() -> {
            SameDiffMemoryUtils.freeModelArrays(sd);
            sd.close();
        });
    }
}
