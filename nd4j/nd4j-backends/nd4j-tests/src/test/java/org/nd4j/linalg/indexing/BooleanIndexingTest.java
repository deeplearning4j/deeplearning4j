/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.indexing;

import com.google.common.base.Function;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.impl.controlflow.WhereNumpy;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.conditions.AbsValueGreaterThan;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.conditions.GreaterThan;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class BooleanIndexingTest extends BaseNd4jTest {
    public BooleanIndexingTest(Nd4jBackend backend) {
        super(backend);
    }
    /*
        1D array checks
     */

    @Test
    public void testAnd1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertTrue(BooleanIndexing.and(array, Conditions.greaterThan(0.5f)));
    }

    @Test
    public void testAnd2() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertTrue(BooleanIndexing.and(array, Conditions.lessThan(6.0f)));
    }

    @Test
    public void testAnd3() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertFalse(BooleanIndexing.and(array, Conditions.lessThan(5.0f)));
    }

    @Test
    public void testAnd4() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertFalse(BooleanIndexing.and(array, Conditions.greaterThan(4.0f)));
    }

    @Test
    public void testAnd5() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-5f, 1e-5f, 1e-5f, 1e-5f, 1e-5f});

        assertTrue(BooleanIndexing.and(array, Conditions.greaterThanOrEqual(1e-5f)));
    }

    @Test
    public void testAnd6() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-5f, 1e-5f, 1e-5f, 1e-5f, 1e-5f});

        assertFalse(BooleanIndexing.and(array, Conditions.lessThan(1e-5f)));
    }

    @Test
    public void testAnd7() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-5f, 1e-5f, 1e-5f, 1e-5f, 1e-5f});

        assertTrue(BooleanIndexing.and(array, Conditions.equals(1e-5f)));
    }

    @Test
    public void testOr1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertTrue(BooleanIndexing.or(array, Conditions.greaterThan(3.0f)));
    }

    @Test
    public void testOr2() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertTrue(BooleanIndexing.or(array, Conditions.lessThan(3.0f)));
    }

    @Test
    public void testOr3() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertFalse(BooleanIndexing.or(array, Conditions.greaterThan(6.0f)));
    }

    @Test
    public void testApplyWhere1() throws Exception {
        INDArray array = Nd4j.create(new float[] {-1f, -1f, -1f, -1f, -1f});

        BooleanIndexing.applyWhere(array, Conditions.lessThan(Nd4j.EPS_THRESHOLD), new Value(Nd4j.EPS_THRESHOLD));

        //System.out.println("Array contains: " + Arrays.toString(array.data().asFloat()));

        assertTrue(BooleanIndexing.and(array, Conditions.equals(Nd4j.EPS_THRESHOLD)));
    }

    @Test
    public void testApplyWhere2() throws Exception {
        INDArray array = Nd4j.create(new float[] {0f, 0f, 0f, 0f, 0f});

        BooleanIndexing.applyWhere(array, Conditions.lessThan(1.0f), new Value(1.0f));

        assertTrue(BooleanIndexing.and(array, Conditions.equals(1.0f)));
    }

    @Test
    public void testApplyWhere3() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-18f, 1e-18f, 1e-18f, 1e-18f, 1e-18f});

        BooleanIndexing.applyWhere(array, Conditions.lessThan(1e-12f), new Value(1e-12f));

        //System.out.println("Array contains: " + Arrays.toString(array.data().asFloat()));

        assertTrue(BooleanIndexing.and(array, Conditions.equals(1e-12f)));
    }

    @Test
    public void testApplyWhere4() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-18f, Float.NaN, 1e-18f, 1e-18f, 1e-18f});

        BooleanIndexing.applyWhere(array, Conditions.lessThan(1e-12f), new Value(1e-12f));

        //System.out.println("Array contains: " + Arrays.toString(array.data().asFloat()));

        BooleanIndexing.applyWhere(array, Conditions.isNan(), new Value(1e-16f));

        System.out.println("Array contains: " + Arrays.toString(array.data().asFloat()));

        assertFalse(BooleanIndexing.or(array, Conditions.isNan()));

        assertTrue(BooleanIndexing.or(array, Conditions.equals(1e-12f)));

        assertTrue(BooleanIndexing.or(array, Conditions.equals(1e-16f)));
    }

    /*
        2D array checks
     */

    @Test
    public void test2dAnd1() throws Exception {
        INDArray array = Nd4j.zeros(10, 10);

        assertTrue(BooleanIndexing.and(array, Conditions.equals(0f)));
    }

    @Test
    public void test2dAnd2() throws Exception {
        INDArray array = Nd4j.zeros(10, 10);

        array.slice(4).putScalar(2, 1e-5f);


        System.out.println(array);

        assertFalse(BooleanIndexing.and(array, Conditions.equals(0f)));


    }

    @Test
    public void test2dAnd3() throws Exception {
        INDArray array = Nd4j.zeros(10, 10);

        array.slice(4).putScalar(2, 1e-5f);

        assertFalse(BooleanIndexing.and(array, Conditions.greaterThan(0f)));
    }

    @Test
    public void test2dAnd4() throws Exception {
        INDArray array = Nd4j.zeros(10, 10);

        array.slice(4).putScalar(2, 1e-5f);

        assertTrue(BooleanIndexing.or(array, Conditions.greaterThan(1e-6f)));
    }

    @Test
    public void test2dApplyWhere1() throws Exception {
        INDArray array = Nd4j.ones(4, 4);

        array.slice(3).putScalar(2, 1e-5f);

        //System.out.println("Array before: " + Arrays.toString(array.data().asFloat()));

        BooleanIndexing.applyWhere(array, Conditions.lessThan(1e-4f), new Value(1e-12f));

        //System.out.println("Array after 1: " + Arrays.toString(array.data().asFloat()));

        assertTrue(BooleanIndexing.or(array, Conditions.equals(1e-12f)));

        assertTrue(BooleanIndexing.or(array, Conditions.equals(1.0f)));

        assertFalse(BooleanIndexing.and(array, Conditions.equals(1e-12f)));
    }

    /**
     * This test fails, because it highlights current mechanics on SpecifiedIndex stuff.
     * Internally there's
     *
     * @throws Exception
     */
    @Test
    public void testSliceAssign1() throws Exception {
        INDArray array = Nd4j.zeros(4, 4);

        INDArray patch = Nd4j.create(new float[] {1e-5f, 1e-5f, 1e-5f});

        INDArray slice = array.slice(1);
        int[] idx = new int[] {0, 1, 3};
        INDArrayIndex[] range = new INDArrayIndex[] {new SpecifiedIndex(idx)};

        INDArray subarray = slice.get(range);

        System.out.println("Subarray: " + Arrays.toString(subarray.data().asFloat()) + " isView: " + subarray.isView());

        slice.put(range, patch);

        System.out.println("Array after being patched: " + Arrays.toString(array.data().asFloat()));

        assertFalse(BooleanIndexing.and(array, Conditions.equals(0f)));
    }

    @Test
    public void testConditionalAssign1() throws Exception {
        INDArray array1 = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7});
        INDArray array2 = Nd4j.create(new double[] {7, 6, 5, 4, 3, 2, 1});
        INDArray comp = Nd4j.create(new double[] {1, 2, 3, 4, 3, 2, 1});

        BooleanIndexing.replaceWhere(array1, array2, Conditions.greaterThan(4));

        assertEquals(comp, array1);
    }

    @Test
    public void testCaSTransform1() throws Exception {
        INDArray array = Nd4j.create(new double[] {1, 2, 0, 4, 5});
        INDArray comp = Nd4j.create(new double[] {1, 2, 3, 4, 5});

        Nd4j.getExecutioner().exec(new CompareAndSet(array, 3, Conditions.equals(0)));

        assertEquals(comp, array);
    }

    @Test
    public void testCaSTransform2() throws Exception {
        INDArray array = Nd4j.create(new double[] {1, 2, 0, 4, 5});
        INDArray comp = Nd4j.create(new double[] {3, 2, 3, 4, 5});

        Nd4j.getExecutioner().exec(new CompareAndSet(array, 3.0, Conditions.lessThan(2)));

        assertEquals(comp, array);
    }

    @Test
    public void testCaSPairwiseTransform1() throws Exception {
        INDArray array = Nd4j.create(new double[] {1, 2, 0, 4, 5});
        INDArray comp = Nd4j.create(new double[] {1, 2, 3, 4, 5});

        Nd4j.getExecutioner().exec(new CompareAndSet(array, comp, Conditions.lessThan(5)));

        assertEquals(comp, array);
    }

    @Test
    public void testCaRPairwiseTransform1() throws Exception {
        INDArray array = Nd4j.create(new double[] {1, 2, 0, 4, 5});
        INDArray comp = Nd4j.create(new double[] {1, 2, 3, 4, 5});

        Nd4j.getExecutioner().exec(new CompareAndReplace(array, comp, Conditions.lessThan(1)));

        assertEquals(comp, array);
    }

    @Test
    public void testCaSPairwiseTransform2() throws Exception {
        INDArray x = Nd4j.create(new double[] {1, 2, 0, 4, 5});
        INDArray y = Nd4j.create(new double[] {2, 4, 3, 0, 5});
        INDArray comp = Nd4j.create(new double[] {2, 4, 3, 4, 5});

        Nd4j.getExecutioner().exec(new CompareAndSet(x, y, Conditions.epsNotEquals(0.0)));

        assertEquals(comp, x);
    }

    @Test
    public void testCaRPairwiseTransform2() throws Exception {
        INDArray x = Nd4j.create(new double[] {1, 2, 0, 4, 5});
        INDArray y = Nd4j.create(new double[] {2, 4, 3, 4, 5});
        INDArray comp = Nd4j.create(new double[] {2, 4, 0, 4, 5});

        Nd4j.getExecutioner().exec(new CompareAndReplace(x, y, Conditions.epsNotEquals(0.0)));

        assertEquals(comp, x);
    }

    @Test
    public void testCaSPairwiseTransform3() throws Exception {
        INDArray x = Nd4j.create(new double[] {1, 2, 0, 4, 5});
        INDArray y = Nd4j.create(new double[] {2, 4, 3, 4, 5});
        INDArray comp = Nd4j.create(new double[] {2, 4, 3, 4, 5});

        Nd4j.getExecutioner().exec(new CompareAndReplace(x, y, Conditions.lessThan(4)));

        assertEquals(comp, x);
    }

    @Test
    public void testCaRPairwiseTransform3() throws Exception {
        INDArray x = Nd4j.create(new double[] {1, 2, 0, 4, 5});
        INDArray y = Nd4j.create(new double[] {2, 4, 3, 4, 5});
        INDArray comp = Nd4j.create(new double[] {2, 2, 3, 4, 5});

        Nd4j.getExecutioner().exec(new CompareAndReplace(x, y, Conditions.lessThan(2)));

        assertEquals(comp, x);
    }


    @Test
    public void testMatchConditionAllDimensions1() throws Exception {
        INDArray array = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        int val = (int) Nd4j.getExecutioner().exec(new MatchCondition(array, Conditions.lessThan(5)), Integer.MAX_VALUE)
                .getDouble(0);

        assertEquals(5, val);
    }

    @Test
    public void testMatchConditionAllDimensions2() throws Exception {
        INDArray array = Nd4j.create(new double[] {0, 1, 2, 3, Double.NaN, 5, 6, 7, 8, 9});

        int val = (int) Nd4j.getExecutioner().exec(new MatchCondition(array, Conditions.isNan()), Integer.MAX_VALUE)
                .getDouble(0);

        assertEquals(1, val);
    }

    @Test
    public void testMatchConditionAllDimensions3() throws Exception {
        INDArray array = Nd4j.create(new double[] {0, 1, 2, 3, Double.NEGATIVE_INFINITY, 5, 6, 7, 8, 9});

        int val = (int) Nd4j.getExecutioner()
                .exec(new MatchCondition(array, Conditions.isInfinite()), Integer.MAX_VALUE).getDouble(0);

        assertEquals(1, val);
    }

    @Test
    public void testAbsValueGreaterThan() {
        final double threshold = 2;

        Condition absValueCondition = new AbsValueGreaterThan(threshold);
        Function<Number, Number> clipFn = new Function<Number, Number>() {
            @Override
            public Number apply(Number number) {
                System.out.println("Number: " + number.doubleValue());
                return (number.doubleValue() > threshold ? threshold : -threshold);
            }
        };

        Nd4j.getRandom().setSeed(12345);
        INDArray orig = Nd4j.rand(1, 20).muli(6).subi(3); //Random numbers: -3 to 3
        INDArray exp = orig.dup();
        INDArray after = orig.dup();

        for (int i = 0; i < exp.length(); i++) {
            double d = exp.getDouble(i);
            if (d > threshold) {
                exp.putScalar(i, threshold);
            } else if (d < -threshold) {
                exp.putScalar(i, -threshold);
            }
        }

        BooleanIndexing.applyWhere(after, absValueCondition, clipFn);

        System.out.println(orig);
        System.out.println(exp);
        System.out.println(after);

        assertEquals(exp, after);
    }

    @Test
    public void testMatchConditionAlongDimension1() throws Exception {
        INDArray array = Nd4j.ones(3, 10);
        array.getRow(2).assign(0.0);

        boolean result[] = BooleanIndexing.and(array, Conditions.equals(0.0), 1);
        boolean comp[] = new boolean[] {false, false, true};

        System.out.println("Result: " + Arrays.toString(result));
        assertArrayEquals(comp, result);
    }

    @Test
    public void testMatchConditionAlongDimension2() throws Exception {
        INDArray array = Nd4j.ones(3, 10);
        array.getRow(2).assign(0.0).putScalar(0, 1.0);

        System.out.println("Array: " + array);

        boolean result[] = BooleanIndexing.or(array, Conditions.lessThan(0.9), 1);
        boolean comp[] = new boolean[] {false, false, true};

        System.out.println("Result: " + Arrays.toString(result));
        assertArrayEquals(comp, result);
    }

    @Test
    public void testMatchConditionAlongDimension3() throws Exception {
        INDArray array = Nd4j.ones(3, 10);
        array.getRow(2).assign(0.0).putScalar(0, 1.0);

        boolean result[] = BooleanIndexing.and(array, Conditions.lessThan(0.0), 1);
        boolean comp[] = new boolean[] {false, false, false};

        System.out.println("Result: " + Arrays.toString(result));
        assertArrayEquals(comp, result);
    }


    @Test
    public void testConditionalUpdate() {
        INDArray arr = Nd4j.linspace(-2, 2, 5);
        INDArray ones = Nd4j.ones(5);
        INDArray exp = Nd4j.create(new double[] {1, 1, 0, 1, 1});


        Nd4j.getExecutioner().exec(new CompareAndSet(ones, arr, ones, Conditions.equals(0.0)));

        assertEquals(exp, ones);
    }


    @Test
    public void testFirstIndex1() {
        INDArray arr = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
        INDArray result = BooleanIndexing.firstIndex(arr, Conditions.greaterThanOrEqual(3));

        assertEquals(2, result.getDouble(0), 0.0);
    }

    @Test
    public void testFirstIndex2() {
        INDArray arr = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
        INDArray result = BooleanIndexing.firstIndex(arr, Conditions.lessThan(3));

        assertEquals(0, result.getDouble(0), 0.0);
    }

    @Test
    public void testLastIndex1() {
        INDArray arr = Nd4j.create(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
        INDArray result = BooleanIndexing.lastIndex(arr, Conditions.greaterThanOrEqual(3));

        assertEquals(8, result.getDouble(0), 0.0);
    }

    @Test
    public void testFirstIndex2D() {
        INDArray arr = Nd4j.create(new double[] {1, 2, 3, 0, 1, 3, 7, 8, 9}).reshape('c', 3, 3);
        INDArray result = BooleanIndexing.firstIndex(arr, Conditions.greaterThanOrEqual(2), 1);
        INDArray exp = Nd4j.create(new double[] {1, 2, 0});

        assertEquals(exp, result);
    }

    @Test
    public void testLastIndex2D() {
        INDArray arr = Nd4j.create(new double[] {1, 2, 3, 0, 1, 3, 7, 8, 0}).reshape('c', 3, 3);
        INDArray result = BooleanIndexing.lastIndex(arr, Conditions.greaterThanOrEqual(2), 1);
        INDArray exp = Nd4j.create(new double[] {2, 2, 1});

        assertEquals(exp, result);
    }

    @Test
    public void testEpsEquals1() throws Exception {
        INDArray array = Nd4j.create(new double[] {-1, -1, -1e-8, 1e-8, 1, 1});
        MatchCondition condition = new MatchCondition(array, Conditions.epsEquals(0.0));
        int numZeroes = Nd4j.getExecutioner().exec(condition, Integer.MAX_VALUE).getInt(0);

        assertEquals(2, numZeroes);
    }

    @Test
    public void testChooseNonZero() {
        INDArray testArr = Nd4j.create(new double[] {
                0.00,  0.51,  0.68,  0.69,  0.86,  0.91,  0.96,  0.97,  0.97,  1.03,  1.13,  1.16,  1.16,  1.17,  1.19,  1.25,  1.25,  1.26,  1.27,  1.28,  1.29,  1.29,  1.29,  1.30,  1.31,  1.32,  1.33,  1.33,  1.35,  1.35,  1.36,  1.37,  1.38,  1.40,  1.41,  1.42,  1.43,  1.44,  1.44,  1.45,  1.45,  1.47,  1.47,  1.51,  1.51,  1.51,  1.52,  1.53,  1.56,  1.57,  1.58,  1.59,  1.61,  1.62,  1.63,  1.63,  1.64,  1.64,  1.66,  1.66,  1.67,  1.67,  1.70,  1.70,  1.70,  1.72,  1.72,  1.72,  1.72,  1.73,  1.74,  1.74,  1.76,  1.76,  1.77,  1.77,  1.80,  1.80,  1.81,  1.82,  1.83,  1.83,  1.84,  1.84,  1.84,  1.85,  1.85,  1.85,  1.86,  1.86,  1.87,  1.88,  1.89,  1.89,  1.89,  1.89,  1.89,  1.91,  1.91,  1.91,  1.92,  1.94,  1.95,  1.97,  1.98,  1.98,  1.98,  1.98,  1.98,  1.99,  2.00,  2.00,  2.01,  2.01,  2.02,  2.03,  2.03,  2.03,  2.04,  2.04,  2.05,  2.06,  2.07,  2.08,  2.08,  2.08,  2.08,  2.09,  2.09,  2.10,  2.10,  2.11,  2.11,  2.11,  2.12,  2.12,  2.13,  2.13,  2.14,  2.14,  2.14,  2.14,  2.15,  2.15,  2.16,  2.16,  2.16,  2.16,  2.16,  2.17
        });

        INDArray filtered = BooleanIndexing.chooseFrom(new INDArray[]{testArr},Arrays.asList(0.0), Collections.emptyList(),new GreaterThan());
        assertFalse(filtered.getDouble(0) == 0);
        assertEquals(testArr.length() - 1,filtered.length());

    }

    @Test
    public void testChooseBasic() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(true);
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray filtered = BooleanIndexing.chooseFrom(new INDArray[]{arr},Arrays.asList(2.0), Collections.emptyList(),new GreaterThan());
        assertEquals(4,filtered.length());
    }


    @Test
    public void testChooseGreaterThanZero() {
        INDArray zero = Nd4j.linspace(0,4,4);
        INDArray filtered = BooleanIndexing.chooseFrom(new INDArray[]{zero},Arrays.asList(0.0), Collections.emptyList(),new GreaterThan());
        assertEquals(3,filtered.length());
    }

    @Test
    public void testChooseNone() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(true);
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray filtered = BooleanIndexing.chooseFrom(new INDArray[]{arr},Arrays.asList(5.0), Collections.emptyList(),new GreaterThan());
        assertNull(filtered);
    }


    @Test
    public void testWhere() {
        INDArray data = Nd4j.create(4);
        INDArray mask = Nd4j.create(4);
        INDArray put = Nd4j.create(4);
        INDArray resultData = Nd4j.create(4);
        INDArray assertion = Nd4j.create(4);
        for (int i = 0; i < 4; i++) {
            data.putScalar(i,i);
            if (i > 1) {
                assertion.putScalar(i, 5.0);
                mask.putScalar(i, 1);
            } else {
                assertion.putScalar(i, i);
                mask.putScalar(i, 0.0);
            }

            put.putScalar(i, 5.0);
            resultData.putScalar(i, 0.0);
        }


        Nd4j.getExecutioner().exec(new WhereNumpy(new INDArray[]{mask,data,put},new INDArray[]{resultData}));
        assertEquals(assertion,resultData);
    }

    @Test
    public void testEpsStuff_1() throws Exception {
        val dtype = Nd4j.dataType();
        val array = Nd4j.create(new float[]{0.001f, 5e-6f, 5e-6f, 5e-6f, 5e-6f});
        val exp = Nd4j.create(new float[]{0.001f, 1.0f, 1.0f, 1.0f, 1.0f});
        BooleanIndexing.replaceWhere(array, 1.0f, Conditions.epsEquals(0));

        assertEquals(exp, array);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
