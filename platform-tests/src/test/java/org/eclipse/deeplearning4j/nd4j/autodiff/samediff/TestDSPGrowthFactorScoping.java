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
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the thread-local scoped growth factor in ArrayCacheMemoryMgr.
 *
 * <p>The growth factor controls over-allocation in the array cache. DSP execution
 * uses a scoped growth factor to prevent leakage into the standard op-by-op path
 * when DSP returns null (no compiled plan).</p>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestDSPGrowthFactorScoping
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPGrowthFactorScoping extends BaseNd4jTestWithBackends {

    private SameDiff sd;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    @Test
    @Order(1)
    @DisplayName("Scoped growth factor: inside scope, effectiveGrowthFactor returns scoped value")
    public void testScopedGrowthFactor() throws Exception {
        double globalGf = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        log.info("Global growth factor before scope: {}", globalGf);

        try (AutoCloseable scope = ArrayCacheMemoryMgr.withGrowthFactor(1.25)) {
            double scopedGf = ArrayCacheMemoryMgr.effectiveGrowthFactor();
            assertEquals(1.25, scopedGf, 1e-9,
                    "Inside scope, effectiveGrowthFactor must return the scoped value");
        }
    }

    @Test
    @Order(2)
    @DisplayName("Scope restore on exit: after scope exits, effectiveGrowthFactor returns global default")
    public void testScopeRestoreOnExit() throws Exception {
        double globalGf = ArrayCacheMemoryMgr.effectiveGrowthFactor();

        try (AutoCloseable scope = ArrayCacheMemoryMgr.withGrowthFactor(2.0)) {
            assertEquals(2.0, ArrayCacheMemoryMgr.effectiveGrowthFactor(), 1e-9);
        }

        double afterScope = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        assertEquals(globalGf, afterScope, 1e-9,
                "After scope exits, growth factor must be restored to global default");
    }

    @Test
    @Order(3)
    @DisplayName("Nested scopes: inner restores to outer's value, not global")
    public void testNestedScopes() throws Exception {
        double globalGf = ArrayCacheMemoryMgr.effectiveGrowthFactor();

        try (AutoCloseable outer = ArrayCacheMemoryMgr.withGrowthFactor(1.5)) {
            assertEquals(1.5, ArrayCacheMemoryMgr.effectiveGrowthFactor(), 1e-9,
                    "Outer scope should set growth factor to 1.5");

            try (AutoCloseable inner = ArrayCacheMemoryMgr.withGrowthFactor(1.1)) {
                assertEquals(1.1, ArrayCacheMemoryMgr.effectiveGrowthFactor(), 1e-9,
                        "Inner scope should set growth factor to 1.1");
            }

            assertEquals(1.5, ArrayCacheMemoryMgr.effectiveGrowthFactor(), 1e-9,
                    "After inner scope exits, growth factor must restore to outer's 1.5");
        }

        assertEquals(globalGf, ArrayCacheMemoryMgr.effectiveGrowthFactor(), 1e-9,
                "After all scopes exit, growth factor must restore to global");
    }

    @Test
    @Order(4)
    @DisplayName("DSP does not leak growth factor: global growthFactor unchanged after DSP execution")
    public void testDSPDoesNotLeakGrowthFactor() {
        double globalGfBefore = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        log.info("Global growth factor before DSP: {}", globalGfBefore);

        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        sd.mmul("output", x, w);
        enableDsp(sd);

        // Execute DSP path
        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);
        try {
            sd.output(Collections.singletonMap("input", input), "output");
        } catch (Exception e) {
            log.info("DSP execution may have failed (acceptable for this test): {}", e.getMessage());
        }

        double globalGfAfter = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        assertEquals(globalGfBefore, globalGfAfter, 1e-9,
                "DSP execution must NOT leak growth factor into global state");
    }

    @Test
    @Order(5)
    @DisplayName("Scope restore on exception: scope restores even if body throws")
    public void testScopeRestoreOnException() {
        double globalGf = ArrayCacheMemoryMgr.effectiveGrowthFactor();

        try {
            try (AutoCloseable scope = ArrayCacheMemoryMgr.withGrowthFactor(3.0)) {
                assertEquals(3.0, ArrayCacheMemoryMgr.effectiveGrowthFactor(), 1e-9);
                throw new RuntimeException("Simulated exception inside scope");
            }
        } catch (Exception e) {
            log.info("Caught expected exception: {}", e.getMessage());
        }

        double afterException = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        assertEquals(globalGf, afterException, 1e-9,
                "After exception, growth factor must be restored to global default");
    }
}
