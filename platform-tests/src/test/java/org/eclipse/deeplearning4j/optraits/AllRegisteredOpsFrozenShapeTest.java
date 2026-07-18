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
package org.eclipse.deeplearning4j.optraits;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.ops.CustomOpDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Exhaustive enumeration of every op registered with libnd4j's {@code OpRegistrator},
 * with per-op trait-bit sanity assertions.
 *
 * <p>This test is <b>fully parameterized</b>: there is one test case per registered
 * op, so a JUnit report names the specific offender ({@code testOpTraitSanity[op=gather]}).
 * There are <b>no {@code assumeTrue} skips</b> — every op either passes or fails.
 *
 * <p>Coverage:
 * <ol>
 *   <li>{@link #testOpTraitSanity(String)} — for EVERY registered op, assert that
 *       {@code getOpTraits(name) != 0}. An op returning 0 has neither descriptor-local
 *       traits nor class-hierarchy-derived traits, so DSP and fusion cannot classify it.</li>
 *   <li>{@link #testNonValdepOps(String)} — for the authoritative list of ops whose
 *       output shape is computed from INPUT SHAPES only (no tensor-value read), assert
 *       that {@code VALUE_DEPENDENT_SHAPE} is NOT set. A false positive here causes
 *       the SHAPES_FROZEN slot executor to report "value-dependent output shape
 *       changed" when in fact an input shape legitimately changed (e.g. batch
 *       1→18).</li>
 *   <li>{@link #testValdepOps(String)} — for the authoritative list of ops whose
 *       shape fn reads tensor DATA (dynamic shape tensor, begin/end tensors, etc.),
 *       assert that {@code VALUE_DEPENDENT_SHAPE} IS set.</li>
 *   <li>{@link #testRegistryInventory()} — meta-check that the registry returns a
 *       non-empty map (sanity gate for the entire test).</li>
 * </ol>
 *
 * <p>Op discovery uses {@code Nd4j.getExecutioner().getCustomOperations()}, which
 * is the Java-side view onto libnd4j's {@code OpRegistrator} via
 * {@code NativeOps::getAllCustomOps()}. The list matches the registered op table
 * exactly (one entry per DECLARE_*_OP macro invocation).
 */
@Slf4j
@DisplayName("every registered libnd4j op declares trait bits — no silent omissions, no misclassified VALDEP")
public class AllRegisteredOpsFrozenShapeTest {

    // ── Trait bits — mirror OpDescriptor.h exactly ───────────────────────────
    private static final int OP_TRAIT_UNARY_ELEMENTWISE     = 1 << 0;
    private static final int OP_TRAIT_BINARY_ELEMENTWISE    = 1 << 1;
    private static final int OP_TRAIT_TERNARY_ELEMENTWISE   = 1 << 2;
    private static final int OP_TRAIT_REDUCTION             = 1 << 3;
    private static final int OP_TRAIT_NORMALIZATION         = 1 << 4;
    private static final int OP_TRAIT_MATMUL                = 1 << 5;
    private static final int OP_TRAIT_FULLY_WRITING         = 1 << 6;
    private static final int OP_TRAIT_VIEW_PRODUCING        = 1 << 7;
    private static final int OP_TRAIT_VALUE_DEPENDENT_SHAPE = 1 << 8;
    private static final int OP_TRAIT_DATA_DEPENDENT        = 1 << 9;
    private static final int OP_TRAIT_SHAPE_ONLY_OUTPUT     = 1 << 10;
    private static final int OP_TRAIT_ACTIVATION            = 1 << 11;
    private static final int OP_TRAIT_COMPARISON            = 1 << 12;
    private static final int OP_TRAIT_LOGICAL               = 1 << 13;
    private static final int OP_TRAIT_IDENTITY              = 1 << 14;
    private static final int OP_TRAIT_DATA_MOVEMENT         = 1 << 15;
    private static final int OP_TRAIT_CONSTANT_GENERATION   = 1 << 16;
    private static final int OP_TRAIT_ATTENTION             = 1 << 17;
    private static final int OP_TRAIT_GATHER                = 1 << 18;
    private static final int OP_TRAIT_GATHER_ND             = 1 << 19;
    private static final int OP_TRAIT_CONCAT                = 1 << 20;
    private static final int OP_TRAIT_SPLIT                 = 1 << 21;
    private static final int OP_TRAIT_SPLIT_V               = 1 << 22;
    private static final int OP_TRAIT_STACK                 = 1 << 23;
    private static final int OP_TRAIT_SLICE                 = 1 << 24;
    private static final int OP_TRAIT_TILE                  = 1 << 25;
    private static final int OP_TRAIT_SCATTER_ND            = 1 << 26;
    private static final int OP_TRAIT_SCATTER_ND_UPDATE     = 1 << 27;
    private static final int OP_TRAIT_CAST                  = 1 << 28;

    // ── Ops whose output shape must NOT be value-dependent ──────────────────
    // Shape is derived ONLY from input shapes + iArgs/attributes — no tensor data
    // read. If any of these carry VALUE_DEPENDENT_SHAPE, the SHAPES_FROZEN recovery
    // path (POST_SEAL_SHAPE_CHANGE) incorrectly routes legitimate input-shape changes
    // into the value-dep assertion branch.
    private static final Set<String> MUST_NOT_BE_VALDEP = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            "gather", "gather_nd", "concat", "stack", "split", "split_v",
            "expand_dims", "squeeze", "flatten", "flatten_2d", "permute",
            "repeat", "transpose"
            // `reshape_no_copy` reads a shape tensor, while `tile` also accepts tensor
            // multiples, so their local descriptors conservatively declare VALDEP.
    )));

    // ── Ops whose output shape IS legitimately value-dependent ──────────────
    // The shape fn dereferences input tensor DATA (dynamic shape tensor, begin/end
    // tensors, multiples tensor, range parameters, etc.). These MUST carry
    // VALUE_DEPENDENT_SHAPE — otherwise cached-shape recovery breaks.
    private static final Set<String> MUST_BE_VALDEP = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            "fill",
            "reshape",          // takes shape tensor → VALDEP
            "reshape_no_copy",  // takes shape tensor → VALDEP
            "slice",
            "pad",
            "broadcast_to",
            "strided_slice",
            "tile",
            "range"
            // The registered op is `lin_space`; this list intentionally uses only
            // canonical OpRegistrator names.

    )));

    // ── Ops we explicitly allow to return 0 traits (temporarily) ────────────
    // Empty by design: every registered op declares its metadata on its own descriptor.
    // Any exemption must document why the op cannot be classified locally.
    private static final Set<String> TRAIT_EXEMPTIONS = Collections.emptySet();

    // ─────────────────────────────────────────────────────────────────────────
    // Native-op trait query helper
    // ─────────────────────────────────────────────────────────────────────────

    private static int traits(String opName) {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().getOpTraits(opName);
    }

    /** Authoritative list of EVERY op that libnd4j's OpRegistrator knows about. */
    private static List<String> allRegisteredOps() {
        Map<String, CustomOpDescriptor> all = Nd4j.getExecutioner().getCustomOperations();
        if (all == null || all.isEmpty()) {
            throw new IllegalStateException(
                    "Nd4j.getExecutioner().getCustomOperations() returned empty — "
                            + "libnd4j may not have initialized or OpRegistrator is broken. "
                            + "This test cannot proceed.");
        }
        // Sorted for deterministic test ordering / report order.
        return new ArrayList<>(new TreeSet<>(all.keySet()));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // @MethodSource providers
    // ─────────────────────────────────────────────────────────────────────────

    static Stream<Arguments> everyRegisteredOp() {
        return allRegisteredOps().stream().map(Arguments::of);
    }

    static Stream<Arguments> nonValdepOps() {
        return MUST_NOT_BE_VALDEP.stream().map(Arguments::of);
    }

    static Stream<Arguments> valdepOps() {
        return MUST_BE_VALDEP.stream().map(Arguments::of);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Per-op trait-presence assertion
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "op={0}")
    @MethodSource("everyRegisteredOp")
    @DisplayName("every registered op must expose non-zero descriptor-local traits")
    public void testOpTraitSanity(String opName) {
        int actual = traits(opName);

        if (TRAIT_EXEMPTIONS.contains(opName)) {
            // Explicit allow-list — but force the exemption to actually be needed.
            // If traits are non-zero, the exemption is stale and should be removed.
            assertEquals(0, actual,
                    "op '" + opName + "' is in TRAIT_EXEMPTIONS but now "
                            + "has traits=0x" + Integer.toHexString(actual)
                            + " — remove it from the exemption set");
            return;
        }

        assertNotEquals(0, actual,
                "op '" + opName + "' has no descriptor-local or class-derived traits. "
                        + "Declare them with addTraits in the op's DECLARE_TYPES block, "
                        + "or add a justified TRAIT_EXEMPTIONS entry.");

        // Every op that sets a "category" bit must also carry at least one of the
        // shape-classification bits (FULLY_WRITING, VIEW_PRODUCING, VALUE_DEPENDENT_SHAPE,
        // SHAPE_ONLY_OUTPUT, DATA_DEPENDENT, CONSTANT_GENERATION, DATA_MOVEMENT, IDENTITY).
        // This catches ops that set only e.g. UNARY_ELEMENTWISE but forget FULLY_WRITING.
        int categoryMask = OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_BINARY_ELEMENTWISE
                | OP_TRAIT_TERNARY_ELEMENTWISE | OP_TRAIT_REDUCTION | OP_TRAIT_NORMALIZATION
                | OP_TRAIT_MATMUL | OP_TRAIT_ACTIVATION | OP_TRAIT_COMPARISON
                | OP_TRAIT_LOGICAL | OP_TRAIT_ATTENTION | OP_TRAIT_DATA_MOVEMENT
                | OP_TRAIT_CONSTANT_GENERATION;
        int shapeClassifierMask = OP_TRAIT_FULLY_WRITING | OP_TRAIT_VIEW_PRODUCING
                | OP_TRAIT_VALUE_DEPENDENT_SHAPE | OP_TRAIT_SHAPE_ONLY_OUTPUT
                | OP_TRAIT_DATA_DEPENDENT | OP_TRAIT_CONSTANT_GENERATION
                | OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_IDENTITY;
        if ((actual & categoryMask) != 0) {
            assertNotEquals(0, actual & shapeClassifierMask,
                    "op '" + opName + "' has a category bit set "
                            + "(0x" + Integer.toHexString(actual & categoryMask) + ") but "
                            + "no shape-classifier bit — this is an inconsistent trait "
                            + "entry. Full traits=0x" + Integer.toHexString(actual));
        }

        log.debug("op={} traits=0x{}", opName, Integer.toHexString(actual));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Must-NOT-be-VALDEP assertions
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "op={0}")
    @MethodSource("nonValdepOps")
    @DisplayName("shape-determined ops must NOT carry VALUE_DEPENDENT_SHAPE")
    public void testNonValdepOps(String opName) {
        int actual = traits(opName);
        assertNotEquals(0, actual,
                "op '" + opName + "' returned 0 traits — it is on the "
                        + "MUST_NOT_BE_VALDEP list but is unregistered or lacks local traits. "
                        + "Fix the op declaration or amend the list.");

        int valdep = actual & OP_TRAIT_VALUE_DEPENDENT_SHAPE;
        if (valdep != 0) {
            fail("op '" + opName + "' is tagged VALUE_DEPENDENT_SHAPE "
                    + "(traits=0x" + Integer.toHexString(actual) + ") but its output shape "
                    + "is fully determined by INPUT SHAPES + iArgs. This causes the "
                    + "SHAPES_FROZEN slot executor to report 'value-dependent output shape "
                    + "changed' on legitimate input-shape changes. Fix addTraits in the '"
                    + opName + "' DECLARE_TYPES block.");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Must-BE-VALDEP assertions
    // ─────────────────────────────────────────────────────────────────────────

    @ParameterizedTest(name = "op={0}")
    @MethodSource("valdepOps")
    @DisplayName("ops whose shape fn reads tensor data MUST carry VALUE_DEPENDENT_SHAPE")
    public void testValdepOps(String opName) {
        int actual = traits(opName);
        assertNotEquals(0, actual,
                "op '" + opName + "' returned 0 traits — it is on the MUST_BE_VALDEP "
                        + "list but is unregistered or lacks descriptor-local traits.");

        int valdep = actual & OP_TRAIT_VALUE_DEPENDENT_SHAPE;
        if (valdep == 0) {
            fail("op '" + opName + "' is NOT tagged VALUE_DEPENDENT_SHAPE "
                    + "(traits=0x" + Integer.toHexString(actual) + ") but its shape fn "
                    + "DOES dereference input tensor data. Without this bit, the "
                    + "SHAPES_FROZEN slot executor will not take the cached-shape "
                    + "recompute branch on input-data changes. Add "
                    + "OP_TRAIT_VALUE_DEPENDENT_SHAPE in the '" + opName
                    + "' DECLARE_TYPES block.");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inventory / meta-check
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("registry inventory — enumerate registered ops and summarise trait classification")
    public void testRegistryInventory() {
        List<String> all = allRegisteredOps();
        assertFalse(all.isEmpty(), "OpRegistrator returned zero ops — backend is broken");

        int traited = 0;
        int untraited = 0;
        int valdep = 0;
        int nonValdep = 0;
        List<String> unclassifiedOps = new ArrayList<>();

        for (String op : all) {
            int t = traits(op);
            if (t == 0) {
                untraited++;
                unclassifiedOps.add(op);
            } else {
                traited++;
                if ((t & OP_TRAIT_VALUE_DEPENDENT_SHAPE) != 0) valdep++;
                else nonValdep++;
            }
        }

        // Report: log everything so CI output includes a snapshot of current state.
        log.info("═══ Op-trait inventory ═══");
        log.info("Total registered ops:       {}", all.size());
        log.info("Ops with non-zero traits:   {}", traited);
        log.info("Ops MISSING trait bits:     {}", untraited);
        log.info("  → value-dep classified:   {}", valdep);
        log.info("  → non-value-dep classified: {}", nonValdep);
        if (!unclassifiedOps.isEmpty()) {
            log.warn("Ops with NO descriptor-local trait classification:");
            for (String m : unclassifiedOps) {
                log.warn("  - {}", m);
            }
        }

        // Sanity bounds. If the registry drops below ~200 ops, something is very wrong.
        assertTrue(all.size() >= 200,
                "Registered op count looks wrong: " + all.size()
                        + " — expected 200+. libnd4j initialization may have failed.");
    }

    @Test
    @DisplayName("sanity: the must-NOT-be-VALDEP list and the must-BE-VALDEP list are disjoint")
    public void testListsAreDisjoint() {
        Set<String> both = new LinkedHashSet<>(MUST_BE_VALDEP);
        both.retainAll(MUST_NOT_BE_VALDEP);
        assertTrue(both.isEmpty(),
                "MUST_BE_VALDEP and MUST_NOT_BE_VALDEP overlap on: " + both
                        + " — fix the lists, an op cannot be both.");
    }

    @Test
    @DisplayName("sanity: every op on the classification lists is actually registered")
    public void testClassificationListsAreRegistered() {
        Set<String> registered = new HashSet<>(allRegisteredOps());
        List<String> notRegistered = new ArrayList<>();
        for (String op : MUST_NOT_BE_VALDEP) {
            if (!registered.contains(op)) notRegistered.add("[MUST_NOT_BE_VALDEP] " + op);
        }
        for (String op : MUST_BE_VALDEP) {
            if (!registered.contains(op)) notRegistered.add("[MUST_BE_VALDEP] " + op);
        }
        assertTrue(notRegistered.isEmpty(),
                "Classification lists reference unregistered ops: " + notRegistered);
    }
}
