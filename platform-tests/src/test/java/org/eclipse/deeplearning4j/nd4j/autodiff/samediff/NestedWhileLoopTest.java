/*
 *  ******************************************************************************
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
 *  *  SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.BaseCompatOp;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Enter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A while loop in another loop's body runs to completion on every pass of the outer loop. The
 * interpreter runs the nested loop where its first Merge falls in the outer body. A DSP plan lays
 * each loop out as one block, the nested loop's inside its parent's, and restarts a loop from its
 * first Merge after its last NextIteration while its predicate holds. Both paths give the nested
 * sums on every execution, through the plan's frozen and replay phases, and so does the graph
 * after a FlatBuffers round trip. Loops side by side, in one body or at the top level, run in the
 * order their values flow.
 */
@Slf4j
public class NestedWhileLoopTest {

    private boolean dspEnabled;

    @BeforeEach
    void rememberDsp() {
        dspEnabled = InferenceSession.isDynamicShapePlanEnabled();
    }

    @AfterEach
    void restoreDsp() {
        InferenceSession.setDynamicShapePlanEnabled(dspEnabled);
    }

    /** sum over i = 5..1 of (sum over j = i..1 of j) = 15 + 10 + 6 + 3 + 1. */
    @ParameterizedTest(name = "dsp={0}")
    @ValueSource(booleans = {true, false})
    void nestedLoopSumsEveryInnerLoop(boolean dsp) {
        InferenceSession.setDynamicShapePlanEnabled(dsp);
        try (SameDiff sd = SameDiff.create()) {
            String out = nestedSum(sd);
            assertEveryExecution(sd, out, 35);
            assertRanOnPlan(sd, dsp, out, 2);
        }
    }

    /** Three levels: the sum over i = 4..1, j = i..1, k = j..1 of k (20 + 10 + 4 + 1 = 35). */
    @ParameterizedTest(name = "dsp={0}")
    @ValueSource(booleans = {true, false})
    void threeLevelsOfLoops(boolean dsp) {
        InferenceSession.setDynamicShapePlanEnabled(dsp);
        try (SameDiff sd = SameDiff.create()) {
            SDVariable zero = sd.constant(0);
            SDVariable[] outer = sd.whileLoop(new SDVariable[]{sd.constant(4), sd.constant(0)},
                    (s, i) -> i[0].gt(0),
                    (s, i) -> new SDVariable[]{i[0].sub(1), i[1].add(s.whileLoop(new SDVariable[]{i[0], zero},
                            (s2, j) -> j[0].gt(0),
                            (s2, j) -> new SDVariable[]{j[0].sub(1), j[1].add(s2.whileLoop(new SDVariable[]{j[0], zero},
                                    (s3, k) -> k[0].gt(0),
                                    (s3, k) -> new SDVariable[]{k[0].sub(1), k[1].add(k[0])})[1])})[1])});
            String out = outer[1].name();
            assertEveryExecution(sd, out, expectedThreeLevelSum(4));
            assertRanOnPlan(sd, dsp, out, 3);
        }
    }

    /**
     * Two loops in one outer body, the second reading the first: for i = 3..1, the sum over
     * j = i..1 of j, doubled i times (6*8 + 3*4 + 1*2).
     */
    @ParameterizedTest(name = "dsp={0}")
    @ValueSource(booleans = {true, false})
    void siblingLoopsInOneBody(boolean dsp) {
        InferenceSession.setDynamicShapePlanEnabled(dsp);
        try (SameDiff sd = SameDiff.create()) {
            SDVariable zero = sd.constant(0);
            SDVariable[] outer = sd.whileLoop(new SDVariable[]{sd.constant(3), sd.constant(0)},
                    (s, i) -> i[0].gt(0),
                    (s, i) -> {
                        SDVariable triangle = s.whileLoop(new SDVariable[]{i[0], zero},
                                (s2, j) -> j[0].gt(0),
                                (s2, j) -> new SDVariable[]{j[0].sub(1), j[1].add(j[0])})[1];
                        SDVariable doubled = s.whileLoop(new SDVariable[]{i[0], triangle},
                                (s2, k) -> k[0].gt(0),
                                (s2, k) -> new SDVariable[]{k[0].sub(1), k[1].mul(2)})[1];
                        return new SDVariable[]{i[0].sub(1), i[1].add(doubled)};
                    });
            String out = outer[1].name();
            int expected = 0;
            for (int i = 3; i > 0; i--) {
                expected += (i * (i + 1) / 2) << i;
            }
            assertEveryExecution(sd, out, expected);
            assertRanOnPlan(sd, dsp, out, 3);
        }
    }

    /** Two top-level loops, the second doubling the first's sum three times: 15 * 8. */
    @ParameterizedTest(name = "dsp={0}")
    @ValueSource(booleans = {true, false})
    void consecutiveLoopsChain(boolean dsp) {
        InferenceSession.setDynamicShapePlanEnabled(dsp);
        try (SameDiff sd = SameDiff.create()) {
            SDVariable sum = sd.whileLoop(new SDVariable[]{sd.constant(5), sd.constant(0)},
                    (s, i) -> i[0].gt(0),
                    (s, i) -> new SDVariable[]{i[0].sub(1), i[1].add(i[0])})[1];
            SDVariable doubled = sd.whileLoop(new SDVariable[]{sd.constant(3), sum},
                    (s, k) -> k[0].gt(0),
                    (s, k) -> new SDVariable[]{k[0].sub(1), k[1].mul(2)})[1];
            String out = doubled.name();
            assertEveryExecution(sd, out, 120);
            assertRanOnPlan(sd, dsp, out, 2);
        }
    }

    /**
     * asFlatBuffers then fromFlatBuffers keeps the graph: every op its class, inputs, outputs,
     * control dependencies, frame and constant-enter flag, every variable its producer and
     * consumers, the op order; and the restored graph sums the same.
     */
    @ParameterizedTest(name = "dsp={0}")
    @ValueSource(booleans = {true, false})
    void nestedLoopSurvivesAFlatBuffersRoundTrip(boolean dsp) throws IOException {
        InferenceSession.setDynamicShapePlanEnabled(dsp);
        try (SameDiff sd = SameDiff.create()) {
            String out = nestedSum(sd);
            try (SameDiff restored = SameDiff.fromFlatBuffers(sd.asFlatBuffers(false))) {
                List<String> differences = differences(sd, restored);
                differences.forEach(d -> log.info("[NESTED_WHILE_SERDE] {}", d));
                assertTrue(differences.isEmpty(), "the round trip changed the graph: " + differences);
                assertEveryExecution(restored, out, 35);
                assertRanOnPlan(restored, dsp, out, 2);
            }
        }
    }

    /** Five executions: the plan's first, its frozen ones and its replays all give the value. */
    private static void assertEveryExecution(SameDiff sd, String out, int expected) {
        for (int execution = 1; execution <= 5; execution++) {
            assertEquals(expected, sd.output(Collections.emptyMap(), out).get(out).getInt(0),
                    "execution " + execution);
        }
    }

    /** With DSP on, the value comes from a plan with one loop region per while loop. */
    private static void assertRanOnPlan(SameDiff sd, boolean dsp, String out, int loops) {
        if (!dsp) return;
        DynamicShapePlan plan = sd.getCachedDynamicShapePlan(Collections.singleton(out));
        assertNotNull(plan, "the loops run on a DSP plan");
        assertEquals(loops, plan.getLoopRegions() == null ? 0 : plan.getLoopRegions().length,
                "one loop region per while loop");
    }

    private static String nestedSum(SameDiff sd) {
        SDVariable innerSumIn = sd.constant(0);
        SDVariable[] sum = sd.whileLoop(new SDVariable[]{sd.constant(5), sd.constant(0)},
                (s, vars) -> vars[0].gt(0),
                (s, vars) -> new SDVariable[]{vars[0].sub(1),
                        vars[1].add(s.whileLoop(new SDVariable[]{vars[0], innerSumIn},
                                (s2, inner) -> inner[0].gt(0),
                                (s2, inner) -> new SDVariable[]{inner[0].sub(1), inner[1].add(inner[0])})[1])});
        return sum[1].name();
    }

    private static int expectedThreeLevelSum(int n) {
        int total = 0;
        for (int i = n; i > 0; i--) {
            for (int j = i; j > 0; j--) {
                for (int k = j; k > 0; k--) {
                    total += k;
                }
            }
        }
        return total;
    }

    private static List<String> differences(SameDiff sd, SameDiff restored) {
        List<String> differences = new ArrayList<>();
        if (!new ArrayList<>(sd.getOps().keySet()).equals(new ArrayList<>(restored.getOps().keySet()))) {
            differences.add("op order: " + sd.getOps().keySet() + " became " + restored.getOps().keySet());
        }
        for (SameDiffOp op : sd.getOps().values()) {
            SameDiffOp other = restored.getOps().get(op.getName());
            if (other == null) {
                differences.add("op " + op.getName() + " is missing");
                continue;
            }
            DifferentialFunction fn = op.getOp();
            DifferentialFunction otherFn = other.getOp();
            same(differences, op.getName() + " class", fn.getClass(), otherFn.getClass());
            same(differences, op.getName() + " inputs", op.getInputsToOp(), other.getInputsToOp());
            same(differences, op.getName() + " outputs", op.getOutputsOfOp(), other.getOutputsOfOp());
            same(differences, op.getName() + " control deps", op.getControlDeps(), other.getControlDeps());
            same(differences, op.getName() + " variable control deps", op.getVarControlDeps(), other.getVarControlDeps());
            same(differences, op.getName() + " control dep for", op.getControlDepFor(), other.getControlDepFor());
            if (fn instanceof BaseCompatOp && otherFn instanceof BaseCompatOp) {
                same(differences, op.getName() + " frame", ((BaseCompatOp) fn).getFrameName(),
                        ((BaseCompatOp) otherFn).getFrameName());
            }
            if (fn instanceof Enter && otherFn instanceof Enter) {
                same(differences, op.getName() + " constant enter", ((Enter) fn).isConstant(), ((Enter) otherFn).isConstant());
            }
        }
        for (Map.Entry<String, Variable> entry : sd.getVariables().entrySet()) {
            Variable variable = entry.getValue();
            Variable other = restored.getVariables().get(entry.getKey());
            if (other == null) {
                differences.add("variable " + entry.getKey() + " is missing");
                continue;
            }
            same(differences, entry.getKey() + " type", variable.getVariable().getVariableType(),
                    other.getVariable().getVariableType());
            same(differences, entry.getKey() + " dtype", variable.getVariable().dataType(), other.getVariable().dataType());
            same(differences, entry.getKey() + " producer", variable.getOutputOfOp(), other.getOutputOfOp());
            // Which ops consume a variable matters, not the order they are listed in
            same(differences, entry.getKey() + " consumers", asSet(variable.getInputsForOp()), asSet(other.getInputsForOp()));
            same(differences, entry.getKey() + " control deps", variable.getControlDeps(), other.getControlDeps());
        }
        return differences;
    }

    private static void same(List<String> differences, String what, Object before, Object after) {
        if (!Objects.equals(emptyAsNull(before), emptyAsNull(after))) {
            differences.add(what + ": " + before + " became " + after);
        }
    }

    private static Set<String> asSet(List<String> values) {
        return values == null ? Collections.emptySet() : new HashSet<>(values);
    }

    private static Object emptyAsNull(Object value) {
        if (value instanceof Set && ((Set<?>) value).isEmpty()) return null;
        return value instanceof List && ((List<?>) value).isEmpty() ? null : value;
    }
}
