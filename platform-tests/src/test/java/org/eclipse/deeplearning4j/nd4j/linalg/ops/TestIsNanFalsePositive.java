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
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproduces and verifies fix for isNaN().any() false positive on CUDA.
 *
 * The bug: isNaN().any() returns true on arrays that contain NO NaN values.
 * Manual element-by-element scan via getFloat() confirms zero NaN elements,
 * yet isNaN().any() reports true.
 */
@Slf4j
public class TestIsNanFalsePositive extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @DisplayName("commit() alone as first operation")
    public void testCommitAloneFirst() {
        // Test: commit() before isNaN, but read HOST values first to verify randn output
        INDArray arr = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();

        // Read ALL host values BEFORE isNaN to check if randn produced NaN
        int hostNanCount = 0;
        for (long i = 0; i < arr.length(); i++) {
            float val = arr.getFloat(i);
            if (Float.isNaN(val)) {
                hostNanCount++;
                if (hostNanCount <= 5) log.info("HOST NaN at index {} = {}", i, val);
            }
        }
        log.info("Host NaN count before isNaN: {}/{}", hostNanCount, arr.length());

        // Now test isNaN - note that getFloat loop above triggered commit+syncToPrimary
        INDArray isNanResult = arr.isNaN();
        int trueCount = 0;
        for (long i = 0; i < isNanResult.length(); i++) {
            if (isNanResult.getFloat(i) != 0) trueCount++;
        }
        boolean anyResult = isNanResult.any();
        log.info("commit-first: BOOL trueCount={}/{}, any()={}", trueCount, isNanResult.length(), anyResult);
        assertEquals(0, hostNanCount, "randn should not produce NaN values");
        assertEquals(0, trueCount, "isNaN() BOOL should have 0 true values");
        assertFalse(anyResult, "isNaN().any() should be false after commit()");
    }

    @Test
    @DisplayName("isNaN without ANY prior sync/commit")
    public void testAaIsNanVeryFirst() {
        // Earliest-running test (alphabetically first). NO commit, NO sync, NO getFloat.
        // If this passes, the bug requires a prior sync/commit trigger.
        INDArray arr = Nd4j.randn(DataType.FLOAT, 10, 100);
        // DO NOT call commit(), syncToHost(), or getFloat() here
        boolean result = arr.isNaN().any();
        log.info("Very first isNaN (no prior sync): {}", result);
        assertFalse(result, "isNaN().any() should be false without any prior sync");
    }

    @Test
    @DisplayName("isNaN on ones after commit — ones uses memset not kernel")
    public void testAbIsNanOnesAfterCommit() {
        INDArray arr = Nd4j.ones(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        INDArray isNanResult = arr.isNaN();
        int trueCount = 0;
        for (long i = 0; i < isNanResult.length(); i++) {
            if (isNanResult.getFloat(i) != 0) trueCount++;
        }
        log.info("ones after commit: BOOL trueCount={}/{}", trueCount, isNanResult.length());
        assertEquals(0, trueCount, "isNaN on ones should have 0 true values");
    }

    @Test
    @DisplayName("isNaN BOOL: linspace vs randn after commit")
    public void testAcDupVsOriginal() {
        // Test 1: linspace (host-created) + commit + isNaN
        INDArray lin = Nd4j.linspace(1, 1000, 1000, DataType.FLOAT).reshape(10, 100);
        Nd4j.getExecutioner().commit();
        INDArray linNan = lin.isNaN();
        int linCount = 0;
        for (long i = 0; i < linNan.length(); i++) {
            if (linNan.getFloat(i) != 0) linCount++;
        }
        log.info("linspace + commit + isNaN: {} / {}", linCount, linNan.length());

        // Test 2: randn (device-created) + dup (forces D2H+H2D) + commit + isNaN
        INDArray arr = Nd4j.randn(DataType.FLOAT, 10, 100);
        INDArray arrDup = arr.dup();  // dup allocates host, copies D2H, copies H2D
        Nd4j.getExecutioner().commit();
        INDArray dupNan = arrDup.isNaN();
        int dupCount = 0;
        for (long i = 0; i < dupNan.length(); i++) {
            if (dupNan.getFloat(i) != 0) dupCount++;
        }
        log.info("randn.dup + commit + isNaN: {} / {}", dupCount, dupNan.length());

        // Test 3: randn (device-only) + commit + isNaN (the failing case)
        INDArray arr2 = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        INDArray arr2Nan = arr2.isNaN();
        int arr2Count = 0;
        for (long i = 0; i < arr2Nan.length(); i++) {
            if (arr2Nan.getFloat(i) != 0) arr2Count++;
        }
        log.info("randn + commit + isNaN: {} / {}", arr2Count, arr2Nan.length());

        // Test 4: randn + commit + gt(0.5) — does a DIFFERENT MatchCondition work?
        INDArray arr3 = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        INDArray arr3Gt = arr3.gt(0.5);
        int gtCount = 0;
        for (long i = 0; i < arr3Gt.length(); i++) {
            if (arr3Gt.getFloat(i) != 0) gtCount++;
        }
        // For randn, ~31% should be > 0.5. Expect ~310 ± 50
        log.info("randn + commit + gt(0.5): {} / {} (expect ~310)", gtCount, arr3Gt.length());

        // Test 5: randn + commit + isNaN with extraArgs verification
        // Manually create MatchConditionTransform with isNaN condition
        INDArray arr4 = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        // Use IsNaN transform directly (opNum 2, no extraArgs) instead of MatchCondition (opNum 5, with extraArgs)
        INDArray isNanResult = Nd4j.create(DataType.BOOL, arr4.shape());
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN(arr4, isNanResult));
        int directCount = 0;
        for (long i = 0; i < isNanResult.length(); i++) {
            if (isNanResult.getFloat(i) != 0) directCount++;
        }
        log.info("randn + commit + IsNaN(direct): {} / {}", directCount, isNanResult.length());

        assertEquals(0, linCount, "linspace + commit + isNaN should be 0");
        assertEquals(0, dupCount, "randn.dup + commit + isNaN should be 0");
        assertEquals(0, directCount, "Direct IsNaN should be 0");
        assertEquals(0, arr2Count, "randn + commit + isNaN should be 0");
    }

    @Test
    @DisplayName("isNaN: host-created vs device-only arrays")
    public void testAdHostVsDevice() {
        // Test with Java-side created array (NOT curand)
        float[] data = new float[1000];
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < 1000; i++) data[i] = (float)(rng.nextGaussian());
        INDArray hostArr = Nd4j.create(data).reshape(10, 100);
        Nd4j.getExecutioner().commit();
        INDArray hostNan = hostArr.isNaN();
        int hostCount = 0;
        for (long i = 0; i < hostNan.length(); i++) {
            if (hostNan.getFloat(i) != 0) hostCount++;
        }
        log.info("Java-created array + commit + isNaN: {} / {}", hostCount, hostNan.length());

        // Test with curand array
        INDArray deviceArr = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        INDArray deviceNan = deviceArr.isNaN();
        int deviceCount = 0;
        for (long i = 0; i < deviceNan.length(); i++) {
            if (deviceNan.getFloat(i) != 0) deviceCount++;
        }
        log.info("curand array + commit + isNaN: {} / {}", deviceCount, deviceNan.length());

        // Test: multiply array by 1.0 to force kernel execution on device, creating host-free output
        INDArray mulArr = Nd4j.create(data).reshape(10, 100);
        INDArray mulResult = mulArr.mul(1.0);  // device-side op, result has device buffer
        Nd4j.getExecutioner().commit();
        INDArray mulNan = mulResult.isNaN();
        int mulCount = 0;
        for (long i = 0; i < mulNan.length(); i++) {
            if (mulNan.getFloat(i) != 0) mulCount++;
        }
        log.info("mul(1.0) result + commit + isNaN: {} / {}", mulCount, mulNan.length());

        assertEquals(0, hostCount, "Host-created should have 0 NaN");
        assertEquals(0, deviceCount, "curand should have 0 NaN");
        assertEquals(0, mulCount, "mul result should have 0 NaN");
    }

    @Test
    @DisplayName("isNaN kernel: definitive execution test with real NaN values")
    public void testAeDefinitive() {
        // Put REAL NaN values in array and verify isNaN detects them
        INDArray arr = Nd4j.randn(DataType.FLOAT, 10, 100);
        arr.putScalar(0, Float.NaN);
        arr.putScalar(50, Float.NaN);
        arr.putScalar(999, Float.NaN);
        Nd4j.getExecutioner().commit();

        // Test 1: arr.isNaN() should find exactly 3
        INDArray result1 = arr.isNaN();
        int count1 = 0;
        for (long i = 0; i < result1.length(); i++) {
            if (result1.getFloat(i) != 0) count1++;
        }
        log.info("isNaN with 3 real NaN: {} (expect 3)", count1);

        // Test 2: Direct IsNaN op (opNum 2, no extraArgs)
        INDArray z2 = Nd4j.create(DataType.BOOL, arr.shape());
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN(arr, z2));
        int count2 = 0;
        for (long i = 0; i < z2.length(); i++) {
            if (z2.getFloat(i) != 0) count2++;
        }
        log.info("Direct IsNaN with 3 real NaN: {} (expect 3)", count2);

        // Test 3: Array WITHOUT NaN, after commit
        INDArray clean = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        INDArray cleanResult = clean.isNaN();
        int cleanCount = 0;
        for (long i = 0; i < cleanResult.length(); i++) {
            if (cleanResult.getFloat(i) != 0) cleanCount++;
        }
        log.info("isNaN on clean randn after commit: {} (expect 0)", cleanCount);

        // Test 4: Check if specific indices are flagged correctly
        if (count1 != 3) {
            StringBuilder sb = new StringBuilder();
            for (long i = 0; i < result1.length(); i++) {
                if (result1.getFloat(i) != 0) {
                    float val = arr.getFloat(i);
                    sb.append(String.format("  idx=%d val=%s isNaN=%b\n", i, val, Float.isNaN(val)));
                }
            }
            log.info("Flagged indices:\n{}", sb);
        }

        assertEquals(3, count1, "Should find exactly 3 NaN values");
        assertEquals(3, count2, "Direct IsNaN should find exactly 3 NaN");
        assertEquals(0, cleanCount, "Clean array should have 0 NaN");
    }

    @Test
    @DisplayName("MatchCondition kernel execution after commit")
    public void testAeKernelExecution() {
        // Isolated test: gt(0.5) via INDArray.gt() vs manual MatchConditionTransform
        INDArray arr = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();

        // Method 1: INDArray.gt() — this is the code path that worked before
        INDArray gtResult = arr.gt(0.5);
        int gt1Count = 0;
        for (long i = 0; i < gtResult.length(); i++) {
            if (gtResult.getFloat(i) != 0) gt1Count++;
        }
        log.info("arr.gt(0.5) after commit: {} / {} (expect ~310)", gt1Count, gtResult.length());

        // Method 2: Manual MatchConditionTransform
        INDArray z2 = Nd4j.createUninitialized(DataType.BOOL, arr.shape());
        var op2 = new org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform(
                arr, z2, org.nd4j.linalg.indexing.conditions.Conditions.greaterThan(0.5));
        Nd4j.getExecutioner().exec(op2);
        int gt2Count = 0;
        for (long i = 0; i < z2.length(); i++) {
            if (z2.getFloat(i) != 0) gt2Count++;
        }
        log.info("Manual MatchCondition gt(0.5) after commit: {} / {} (expect ~310)", gt2Count, z2.length());

        // Method 3: isNaN via INDArray.isNaN()
        INDArray nanResult = arr.isNaN();
        int nanCount = 0;
        for (long i = 0; i < nanResult.length(); i++) {
            if (nanResult.getFloat(i) != 0) nanCount++;
        }
        log.info("arr.isNaN() after commit: {} / {}", nanCount, nanResult.length());

        // Method 4: Manual MatchConditionTransform isNaN
        INDArray z4 = Nd4j.createUninitialized(DataType.BOOL, arr.shape());
        var op4 = new org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform(
                arr, z4, org.nd4j.linalg.indexing.conditions.Conditions.isNan());
        Nd4j.getExecutioner().exec(op4);
        int nan4Count = 0;
        for (long i = 0; i < z4.length(); i++) {
            if (z4.getFloat(i) != 0) nan4Count++;
        }
        log.info("Manual MatchCondition isNaN after commit: {} / {}", nan4Count, z4.length());

        assertTrue(gt1Count > 200, "arr.gt(0.5) should find >200 matches, got " + gt1Count);
        assertTrue(gt2Count > 200, "Manual gt(0.5) should find >200 matches, got " + gt2Count);
        assertEquals(0, nanCount, "arr.isNaN() should be 0");
        assertEquals(0, nan4Count, "Manual isNaN should be 0");
    }

    @Test
    @DisplayName("isNaN: zeroed output vs uninitialized output")
    public void testAeZeroedOutput() {
        // Test 1: MatchConditionTransform isNaN with ZEROED output buffer
        INDArray arr = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        INDArray zeroedZ = Nd4j.zeros(DataType.BOOL, arr.shape());
        var op = new org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform(
                arr, zeroedZ, org.nd4j.linalg.indexing.conditions.Conditions.isNan());
        Nd4j.getExecutioner().exec(op);
        int zeroedCount = 0;
        for (long i = 0; i < zeroedZ.length(); i++) {
            if (zeroedZ.getFloat(i) != 0) zeroedCount++;
        }
        log.info("isNaN with ZEROED output: {} / {}", zeroedCount, zeroedZ.length());

        // Test 2: MatchConditionTransform isNaN with UNINITIALIZED output (original bug)
        INDArray arr2 = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        INDArray uninitZ = Nd4j.createUninitialized(DataType.BOOL, arr2.shape());
        var op2 = new org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform(
                arr2, uninitZ, org.nd4j.linalg.indexing.conditions.Conditions.isNan());
        Nd4j.getExecutioner().exec(op2);
        int uninitCount = 0;
        for (long i = 0; i < uninitZ.length(); i++) {
            if (uninitZ.getFloat(i) != 0) uninitCount++;
        }
        log.info("isNaN with UNINIT output: {} / {}", uninitCount, uninitZ.length());

        // Test 3: Does the SAME MatchCondition opNum 5 with mode=3 (GT) also fail with uninit output?
        INDArray arr3 = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        INDArray uninitGt = Nd4j.createUninitialized(DataType.BOOL, arr3.shape());
        var op3 = new org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform(
                arr3, uninitGt, org.nd4j.linalg.indexing.conditions.Conditions.greaterThan(0.5));
        Nd4j.getExecutioner().exec(op3);
        int gtCount = 0;
        for (long i = 0; i < uninitGt.length(); i++) {
            if (uninitGt.getFloat(i) != 0) gtCount++;
        }
        log.info("gt(0.5) with UNINIT output: {} / {} (expect ~310)", gtCount, uninitGt.length());

        // Test 4: MatchCondition isNaN WITHOUT commit
        INDArray arr4 = Nd4j.randn(DataType.FLOAT, 10, 100);
        // NO commit
        INDArray uninitZ4 = Nd4j.createUninitialized(DataType.BOOL, arr4.shape());
        var op4 = new org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform(
                arr4, uninitZ4, org.nd4j.linalg.indexing.conditions.Conditions.isNan());
        Nd4j.getExecutioner().exec(op4);
        int noCommitCount = 0;
        for (long i = 0; i < uninitZ4.length(); i++) {
            if (uninitZ4.getFloat(i) != 0) noCommitCount++;
        }
        log.info("isNaN NO commit: {} / {}", noCommitCount, uninitZ4.length());

        assertEquals(0, zeroedCount, "isNaN with zeroed output should be 0");
        assertEquals(0, noCommitCount, "isNaN without commit should be 0");
        assertEquals(0, uninitCount, "isNaN with uninit output should be 0");
    }

    @Test
    @DisplayName("IsNan reduce (opNum 2) vs MatchCondition (opNum 5)")
    public void testAeIsNanReduceVsTransform() {
        // Compare MatchConditionTransform (TRANSFORM_BOOL opNum 5) with IsNan (REDUCE_FLOAT opNum 58)
        INDArray arr = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();

        // Method 1: arr.isNaN() — uses MatchConditionTransform TRANSFORM_BOOL
        INDArray isNanBool = arr.isNaN();
        int boolTrueCount = 0;
        for (long i = 0; i < isNanBool.length(); i++) {
            if (isNanBool.getFloat(i) != 0) boolTrueCount++;
        }

        // Method 2: Direct reduce — counts NaN
        double nanCount = Nd4j.getExecutioner().exec(
            new org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero(arr.isNaN())
        ).getDouble(0);

        log.info("MatchCondition trueCount={}, CountNonZero of isNaN={}", boolTrueCount, nanCount);
        assertEquals(0, boolTrueCount, "MatchCondition should find 0 NaN");
    }

    @Test
    @DisplayName("isNaN().any() should be false for randn array (no NaN)")
    public void testIsNanFalsePositiveSmall() {
        INDArray arr = Nd4j.randn(DataType.FLOAT, 10, 100);
        verifyNoFalsePositive(arr, "small [10,100]");
    }

    @Test
    @DisplayName("Isolate syncToHost trigger: same array vs different array vs gt comparison")
    public void testSyncToHostIsolation() {
        // Test A: syncToHost on the SAME array, then isNaN
        INDArray arrA = Nd4j.randn(DataType.FLOAT, 10, 100);
        arrA.syncToHost();
        boolean resultA = arrA.isNaN().any();
        log.info("Test A (syncToHost on SAME array -> isNaN): {}", resultA);

        // Test B: syncToHost on a DIFFERENT array, then isNaN on original
        INDArray arrB = Nd4j.randn(DataType.FLOAT, 10, 100);
        INDArray dummy = Nd4j.randn(DataType.FLOAT, 5, 5);
        dummy.syncToHost();  // sync a DIFFERENT array
        boolean resultB = arrB.isNaN().any();
        log.info("Test B (syncToHost on DIFFERENT array -> isNaN): {}", resultB);

        // Test C: syncToHost on same array, then gt(0) instead of isNaN
        INDArray arrC = Nd4j.randn(DataType.FLOAT, 10, 100);
        arrC.syncToHost();
        INDArray gtResult = arrC.gt(0);
        // gt should return ~50% true for randn — manual count
        int gtTrueCount = 0;
        for (long i = 0; i < gtResult.length(); i++) {
            if (gtResult.getFloat(i) != 0) gtTrueCount++;
        }
        log.info("Test C (syncToHost -> gt(0)): trueCount={}/{}", gtTrueCount, arrC.length());

        // Test D: No syncToHost, but call commit() to sync streams
        INDArray arrD = Nd4j.randn(DataType.FLOAT, 10, 100);
        Nd4j.getExecutioner().commit();
        boolean resultD = arrD.isNaN().any();
        log.info("Test D (commit() only -> isNaN): {}", resultD);

        // Test E: syncToHost, then dup, then isNaN on dup
        INDArray arrE = Nd4j.randn(DataType.FLOAT, 10, 100);
        arrE.syncToHost();
        boolean resultE = arrE.dup().isNaN().any();
        log.info("Test E (syncToHost -> dup -> isNaN): {}", resultE);

        // Test F: getFloat(0) instead of syncToHost (also triggers D2H)
        INDArray arrF = Nd4j.randn(DataType.FLOAT, 10, 100);
        float firstVal = arrF.getFloat(0);  // triggers D2H sync
        boolean resultF = arrF.isNaN().any();
        log.info("Test F (getFloat(0)={} -> isNaN): {}", firstVal, resultF);

        // Assertions
        assertFalse(resultA, "Test A: syncToHost on SAME array should not cause false positive");
        assertFalse(resultB, "Test B: syncToHost on DIFFERENT array should not affect isNaN");
        assertFalse(resultD, "Test D: commit() should not cause false positive");
        assertFalse(resultE, "Test E: syncToHost + dup should not cause false positive");
        assertFalse(resultF, "Test F: getFloat(0) should not cause false positive");
    }

    @Test
    @DisplayName("isNaN().any() should be false for large randn array")
    public void testIsNanFalsePositiveLarge() {
        // This size matches decoder logits: [1, 15, 49280] = 739,200 elements
        INDArray arr = Nd4j.randn(DataType.FLOAT, 1, 15, 49280);
        verifyNoFalsePositive(arr, "large [1,15,49280]");
    }

    @Test
    @DisplayName("isNaN().any() should be false for ones array")
    public void testIsNanOnes() {
        INDArray arr = Nd4j.ones(DataType.FLOAT, 1, 15, 49280);
        verifyNoFalsePositive(arr, "ones [1,15,49280]");
    }

    @Test
    @DisplayName("isNaN().any() should be false for zeros array")
    public void testIsNanZeros() {
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 1, 15, 49280);
        verifyNoFalsePositive(arr, "zeros [1,15,49280]");
    }

    @Test
    @DisplayName("isNaN().any() should be true when NaN is actually present")
    public void testIsNanTruePositive() {
        INDArray arr = Nd4j.randn(DataType.FLOAT, 1, 15, 49280);
        // Insert a single NaN
        arr.putScalar(new long[]{0, 7, 25000}, Float.NaN);

        INDArray isNanResult = arr.isNaN();
        boolean any = isNanResult.any();
        log.info("True positive test: isNaN().any()={}", any);
        assertTrue(any, "isNaN().any() should be true when NaN is present");
    }

    @Test
    @DisplayName("isNaN().any() should be false after matmul (simulates decoder)")
    public void testIsNanAfterMatmul() {
        // Simulate decoder: matmul produces logits, check for false positive
        INDArray a = Nd4j.randn(DataType.FLOAT, 1, 576).mul(0.1);
        INDArray b = Nd4j.randn(DataType.FLOAT, 576, 49280).mul(0.02);
        INDArray logits = a.mmul(b);

        log.info("Matmul result: shape={}, min={}, max={}", Arrays.toString(logits.shape()),
                logits.minNumber(), logits.maxNumber());
        verifyNoFalsePositive(logits, "matmul [1,49280]");
    }

    @Test
    @DisplayName("isNaN().any() should be false after chained ops")
    public void testIsNanAfterChainedOps() {
        // Chain of ops that mimics decoder execution
        INDArray x = Nd4j.randn(DataType.FLOAT, 1, 50, 576).mul(0.1);
        INDArray w1 = Nd4j.randn(DataType.FLOAT, 576, 1536).mul(0.02);
        INDArray w2 = Nd4j.randn(DataType.FLOAT, 1536, 49280).mul(0.02);

        // Reshape to 2D for matmul
        INDArray x2d = x.reshape(50, 576);
        INDArray hidden = Nd4j.nn().relu(x2d.mmul(w1), 0);
        INDArray logits = hidden.mmul(w2).reshape(1, 50, 49280);

        log.info("Chained ops result: shape={}, min={}, max={}", Arrays.toString(logits.shape()),
                logits.minNumber(), logits.maxNumber());
        verifyNoFalsePositive(logits, "chained [1,50,49280]");
    }

    @Test
    @DisplayName("isInfinite().any() should be false for finite array")
    public void testIsInfiniteFalsePositive() {
        INDArray arr = Nd4j.randn(DataType.FLOAT, 1, 15, 49280);

        INDArray isInfResult = arr.isInfinite();
        boolean any = isInfResult.any();

        // Also manual check
        int infCount = 0;
        for (long i = 0; i < arr.length(); i++) {
            if (Float.isInfinite(arr.getFloat(i))) infCount++;
        }

        log.info("isInfinite().any()={}, manual inf count={}", any, infCount);
        assertEquals(0, infCount, "Manual scan should find 0 Inf");
        assertFalse(any, "isInfinite().any() should be false for finite array");
    }

    @Test
    @DisplayName("isNaN on dup'd array should match original")
    public void testIsNanAfterDup() {
        INDArray arr = Nd4j.randn(DataType.FLOAT, 1, 15, 49280);
        INDArray dup = arr.dup();

        boolean origResult = arr.isNaN().any();
        boolean dupResult = dup.isNaN().any();

        log.info("isNaN().any(): original={}, dup={}", origResult, dupResult);

        // Both should be false (no NaN in randn)
        // Manual verification
        int nanCount = 0;
        for (long i = 0; i < arr.length(); i++) {
            if (Float.isNaN(arr.getFloat(i))) nanCount++;
        }
        log.info("Manual NaN count: {}", nanCount);
        assertEquals(0, nanCount, "No NaN should exist in randn array");
        assertFalse(origResult, "isNaN().any() false positive on original");
        assertFalse(dupResult, "isNaN().any() false positive on dup");
    }

    @Test
    @DisplayName("Diagnostic: isolate isNaN transform vs any reduce")
    public void testDiagnosticIsolation() {
        INDArray arr = Nd4j.randn(DataType.FLOAT, 10, 100);

        // Step A: Get the isNaN BOOL result
        INDArray isNanResult = arr.isNaN();

        // Step B: Manually scan the BOOL result array
        int trueCountInBool = 0;
        for (long i = 0; i < isNanResult.length(); i++) {
            float val = isNanResult.getFloat(i);
            if (val != 0) {
                trueCountInBool++;
                if (trueCountInBool <= 5) {
                    log.info("BOOL[{}] = {} (non-zero!)", i, val);
                }
            }
        }
        log.info("Manual scan of BOOL array: {} non-zero out of {}", trueCountInBool, isNanResult.length());

        // Step C: Call any() on the BOOL result
        boolean anyResult = isNanResult.any();
        log.info("any() result: {}", anyResult);

        // Step D: Create a known all-false BOOL array and test any()
        INDArray allFalse = Nd4j.zeros(DataType.BOOL, 10, 100);
        boolean anyFalse = allFalse.any();
        log.info("zeros(BOOL).any() = {}", anyFalse);

        // Step E: Test without manual scan first (no getFloat sync before isNaN)
        INDArray arr2 = Nd4j.randn(DataType.FLOAT, 10, 100);
        boolean directResult = arr2.isNaN().any();
        log.info("Direct isNaN().any() (no prior getFloat) = {}", directResult);

        // Step F: Test with explicit sync before isNaN — ISOLATE THE BUG
        INDArray arr3 = Nd4j.randn(DataType.FLOAT, 10, 100);
        arr3.syncToHost();  // force D2H sync
        INDArray isNanResult3 = arr3.isNaN();  // get BOOL array
        boolean afterSyncResult = isNanResult3.any();  // test any() FIRST
        // NOW scan the BOOL array to see if isNaN() produced wrong output
        int trueCountAfterSync = 0;
        for (long i = 0; i < isNanResult3.length(); i++) {
            float val = isNanResult3.getFloat(i);
            if (val != 0) {
                trueCountAfterSync++;
                if (trueCountAfterSync <= 5) {
                    log.info("Step F BOOL[{}] = {} (non-zero!), orig float={}", i, val, arr3.getFloat(i));
                }
            }
        }
        log.info("Step F: syncToHost->isNaN().any()={}, BOOL trueCount={}/{}",
                afterSyncResult, trueCountAfterSync, isNanResult3.length());
        if (trueCountAfterSync == 0 && afterSyncResult) {
            log.error("BUG IS IN any() REDUCE: BOOL array has 0 true values but any() returns true after syncToHost");
        } else if (trueCountAfterSync > 0) {
            log.error("BUG IS IN isNaN() TRANSFORM: produces {} true values after syncToHost", trueCountAfterSync);
        }

        // Step G: Test with dup first
        INDArray arr4 = Nd4j.randn(DataType.FLOAT, 10, 100);
        boolean afterDupResult = arr4.dup().isNaN().any();
        log.info("dup().isNaN().any() = {}", afterDupResult);

        // Report
        if (trueCountInBool > 0) {
            log.error("BUG IS IN isNaN() TRANSFORM: produces {} true values in BOOL array for non-NaN input",
                    trueCountInBool);
        } else if (anyResult) {
            log.error("BUG IS IN any() REDUCE: BOOL array has 0 true values but any() returns true");
        } else {
            log.info("No bug detected in this run");
        }

        assertEquals(0, trueCountInBool, "isNaN() should produce all-false for non-NaN input");
        assertFalse(anyResult, "any() should return false for all-false BOOL array");
        assertFalse(anyFalse, "zeros(BOOL).any() should be false");
        assertFalse(directResult, "Direct isNaN().any() should be false");
        assertFalse(afterSyncResult, "After sync, isNaN().any() should be false");
        assertFalse(afterDupResult, "dup().isNaN().any() should be false");
    }

    private void verifyNoFalsePositive(INDArray arr, String label) {
        // Step 1: Manual element-by-element check (ground truth)
        int nanCount = 0;
        int infCount = 0;
        long length = arr.length();
        for (long i = 0; i < length; i++) {
            float val = arr.getFloat(i);
            if (Float.isNaN(val)) nanCount++;
            if (Float.isInfinite(val)) infCount++;
        }

        // Step 2: isNaN().any() check
        INDArray isNanResult = arr.isNaN();
        boolean anyNaN = isNanResult.any();

        log.info("[{}] shape={}, length={}, manual: {}NaN {}Inf, isNaN().any()={}",
                label, Arrays.toString(arr.shape()), length, nanCount, infCount, anyNaN);

        // Verify
        assertEquals(0, nanCount, label + ": manual scan should find 0 NaN");
        assertEquals(0, infCount, label + ": manual scan should find 0 Inf");
        assertFalse(anyNaN, label + ": isNaN().any() should be false when no NaN exists");
    }
}
