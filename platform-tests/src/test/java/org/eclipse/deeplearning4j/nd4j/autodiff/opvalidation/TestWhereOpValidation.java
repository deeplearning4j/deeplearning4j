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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validation tests for the Where operation.
 * Tests both single-input (coordinate extraction) and three-input (element-wise select) modes.
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class TestWhereOpValidation extends BaseOpValidation {

    private DataType initialType;

    @BeforeEach
    public void before() {
        Nd4j.create(1);
        initialType = Nd4j.dataType();
        Nd4j.setDataType(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @AfterEach
    public void after() {
        if (initialType != null) {
            Nd4j.setDataType(initialType);
        }
    }

    /**
     * Test single-input Where: returns coordinates of true elements
     * For a 1D boolean array [false, true, false, true, true]
     * Should return [[1], [3], [4]] (coordinates of true elements)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereSingleInput1D(Nd4jBackend backend) {
        INDArray condition = Nd4j.create(new boolean[]{false, true, false, true, true});

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);

        INDArray result = results[0];
        log.info("Where 1D result shape: {}", Arrays.toString(result.shape()));
        log.info("Where 1D result: {}", result);

        // Should have 3 true elements, each with 1 coordinate (rank 1)
        assertEquals(2, result.rank());
        assertEquals(3, result.size(0)); // 3 true elements
        assertEquals(1, result.size(1)); // 1D array, so 1 coordinate per element

        // Verify coordinates: should be 1, 3, 4
        assertEquals(1, result.getLong(0, 0));
        assertEquals(3, result.getLong(1, 0));
        assertEquals(4, result.getLong(2, 0));
    }

    /**
     * Test single-input Where with 2D array
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereSingleInput2D(Nd4jBackend backend) {
        // Create a 3x3 boolean array with specific true positions
        INDArray condition = Nd4j.create(new boolean[][]{
            {true,  false, false},
            {false, true,  false},
            {false, false, true}
        });

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);

        INDArray result = results[0];
        log.info("Where 2D result shape: {}", Arrays.toString(result.shape()));
        log.info("Where 2D result: {}", result);

        // Should have 3 true elements, each with 2 coordinates (rank 2)
        assertEquals(2, result.rank());
        assertEquals(3, result.size(0)); // 3 true elements
        assertEquals(2, result.size(1)); // 2D array, so 2 coordinates per element

        // Verify coordinates: diagonal elements (0,0), (1,1), (2,2)
        assertEquals(0, result.getLong(0, 0));
        assertEquals(0, result.getLong(0, 1));
        assertEquals(1, result.getLong(1, 0));
        assertEquals(1, result.getLong(1, 1));
        assertEquals(2, result.getLong(2, 0));
        assertEquals(2, result.getLong(2, 1));
    }

    /**
     * Test single-input Where with all false (empty result)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereAllFalseReturnsEmpty(Nd4jBackend backend) {
        INDArray condition = Nd4j.create(DataType.BOOL, 10).assign(false);

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);

        INDArray result = results[0];
        log.info("Where all-false result shape: {}", Arrays.toString(result.shape()));

        // Result should be empty (0 true elements)
        assertTrue(result.isEmpty() || result.size(0) == 0);
    }

    /**
     * Test three-input Where: condition ? x : y (element-wise)
     * Same shape for all inputs
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereThreeInputSameShape(Nd4jBackend backend) {
        INDArray condition = Nd4j.create(new boolean[]{true, false, true, false, true});
        INDArray x = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray y = Nd4j.create(new double[]{10, 20, 30, 40, 50});

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition, x, y)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        assertEquals(1, results.length);

        INDArray result = results[0];
        log.info("Where 3-input same shape result: {}", result);

        // Expected: [1, 20, 3, 40, 5]
        INDArray expected = Nd4j.create(new double[]{1, 20, 3, 40, 5});
        assertEquals(expected, result);
    }

    /**
     * Test three-input Where with 2D arrays
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereThreeInput2D(Nd4jBackend backend) {
        INDArray condition = Nd4j.create(new boolean[][]{
            {true,  false},
            {false, true}
        });
        INDArray x = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        INDArray y = Nd4j.create(new double[][]{{10, 20}, {30, 40}});

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition, x, y)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        INDArray result = results[0];
        log.info("Where 3-input 2D result: {}", result);

        // Expected: [[1, 20], [30, 4]]
        INDArray expected = Nd4j.create(new double[][]{{1, 20}, {30, 4}});
        assertEquals(expected, result);
    }

    /**
     * Test three-input Where with broadcasting
     * Condition is scalar, x and y are arrays
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereThreeInputBroadcast(Nd4jBackend backend) {
        // Condition broadcasts from [1] to [5]
        INDArray condition = Nd4j.create(new boolean[]{true});
        INDArray x = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray y = Nd4j.create(new double[]{10, 20, 30, 40, 50});

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition, x, y)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        INDArray result = results[0];
        log.info("Where broadcast result: {}", result);

        // Condition is true, so should select all from x
        INDArray expected = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        assertEquals(expected, result);
    }

    /**
     * Test Where with SameDiff (single input mode)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereSameDiffSingleInput(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        INDArray conditionArr = Nd4j.create(new boolean[]{false, true, true, false, true});
        SDVariable condition = sd.var("condition", conditionArr);

        SDVariable result = sd.where(condition);

        Map<String, INDArray> output = sd.output((Map<String, INDArray>) null, result.name());
        INDArray resultArr = output.get(result.name());

        log.info("SameDiff Where single input result: {}", resultArr);

        // Should have 3 true elements at indices 1, 2, 4
        assertEquals(3, resultArr.size(0));
        assertEquals(1, resultArr.getLong(0, 0));
        assertEquals(2, resultArr.getLong(1, 0));
        assertEquals(4, resultArr.getLong(2, 0));
    }

    /**
     * Test Where with SameDiff (three input mode)
     * Note: SameDiff where signature is where(x, y, condition) not where(condition, x, y)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereSameDiffThreeInput(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        // SameDiff where requires boolean condition - create as BOOL type
        INDArray conditionArr = Nd4j.create(DataType.BOOL, 4);
        conditionArr.putScalar(0, 1); // true
        conditionArr.putScalar(1, 0); // false
        conditionArr.putScalar(2, 1); // true
        conditionArr.putScalar(3, 0); // false

        INDArray xArr = Nd4j.create(new double[]{1, 2, 3, 4});
        INDArray yArr = Nd4j.create(new double[]{10, 20, 30, 40});

        SDVariable condition = sd.var("condition", conditionArr);
        SDVariable x = sd.var("x", xArr);
        SDVariable y = sd.var("y", yArr);

        // Note: SameDiff where signature is (x, y, condition) not (condition, x, y)
        SDVariable result = sd.where(x, y, condition);

        Map<String, INDArray> output = sd.output((Map<String, INDArray>) null, result.name());
        INDArray resultArr = output.get(result.name());

        log.info("SameDiff Where three input result: {}", resultArr);

        // Expected: [1, 20, 3, 40]
        INDArray expected = Nd4j.create(new double[]{1, 20, 3, 40});
        assertEquals(expected, resultArr);
    }

    /**
     * Test Where with different data types for condition
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereWithIntCondition(Nd4jBackend backend) {
        // Non-zero values are treated as true
        // Use createFromArray to ensure correct interpretation as values
        INDArray condition = Nd4j.createFromArray(0, 1, 0, 2, 3).castTo(DataType.INT32);
        INDArray x = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray y = Nd4j.create(new double[]{10, 20, 30, 40, 50});

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition, x, y)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        INDArray result = results[0];
        log.info("Where int condition result: {}", result);

        // Expected: [10, 2, 30, 4, 5] (0 is false, non-zero is true)
        INDArray expected = Nd4j.create(new double[]{10, 2, 30, 4, 5});
        assertEquals(expected, result);
    }

    /**
     * Test Where with float condition (non-zero is true)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereWithFloatCondition(Nd4jBackend backend) {
        INDArray condition = Nd4j.create(new double[]{0.0, 0.5, 0.0, -1.0, 1.0});
        INDArray x = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray y = Nd4j.create(new double[]{10, 20, 30, 40, 50});

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition, x, y)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        INDArray result = results[0];
        log.info("Where float condition result: {}", result);

        // Expected: [10, 2, 30, 4, 5]
        INDArray expected = Nd4j.create(new double[]{10, 2, 30, 4, 5});
        assertEquals(expected, result);
    }

    /**
     * Test Where with larger arrays for performance validation
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereLargeArray(Nd4jBackend backend) {
        int size = 10000;

        // Create random boolean condition
        INDArray condition = Nd4j.rand(DataType.DOUBLE, size).gt(0.5);
        INDArray x = Nd4j.linspace(1, size, size, DataType.DOUBLE);
        INDArray y = Nd4j.linspace(size + 1, 2 * size, size, DataType.DOUBLE);

        long startTime = System.currentTimeMillis();

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition, x, y)
                .build();

        INDArray[] results = Nd4j.exec(op);

        long endTime = System.currentTimeMillis();
        log.info("Where large array ({} elements) execution time: {} ms", size, (endTime - startTime));

        assertNotNull(results);
        assertEquals(1, results.length);
        assertEquals(size, results[0].length());

        // Verify a few random samples
        for (int i = 0; i < 10; i++) {
            int idx = Nd4j.getRandom().nextInt(size);
            boolean cond = condition.getDouble(idx) != 0;
            double expected = cond ? x.getDouble(idx) : y.getDouble(idx);
            assertEquals(expected, results[0].getDouble(idx), 1e-5,
                "Mismatch at index " + idx);
        }
    }

    /**
     * Test Where with attention mask style pattern (2D boolean mask)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereAttentionMaskPattern(Nd4jBackend backend) {
        int seqLen = 128;

        // Create lower triangular mask (causal attention pattern)
        INDArray condition = Nd4j.ones(DataType.BOOL, seqLen, seqLen);
        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                condition.putScalar(i, j, 0);
            }
        }

        INDArray attentionScores = Nd4j.rand(DataType.DOUBLE, seqLen, seqLen);
        INDArray maskValue = Nd4j.ones(DataType.DOUBLE, seqLen, seqLen).mul(-1e9);

        long startTime = System.currentTimeMillis();

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition, attentionScores, maskValue)
                .build();

        INDArray[] results = Nd4j.exec(op);

        long endTime = System.currentTimeMillis();
        log.info("Where attention mask ({}x{}) execution time: {} ms", seqLen, seqLen, (endTime - startTime));

        assertNotNull(results);
        INDArray result = results[0];

        // Verify upper triangle is masked
        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 5; j++) {
                assertEquals(-1e9, result.getDouble(i, j), 1e-5,
                    "Upper triangle should be masked at (" + i + "," + j + ")");
            }
        }

        // Verify lower triangle has original values
        for (int i = 1; i < 5; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(attentionScores.getDouble(i, j), result.getDouble(i, j), 1e-5,
                    "Lower triangle should have original value at (" + i + "," + j + ")");
            }
        }
    }

    /**
     * Test Where single input with 3D array
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereSingleInput3D(Nd4jBackend backend) {
        // 2x2x2 boolean array with one true value at (1,0,1)
        INDArray condition = Nd4j.zeros(DataType.BOOL, 2, 2, 2);
        condition.putScalar(new int[]{1, 0, 1}, 1);

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        INDArray result = results[0];
        log.info("Where 3D result: {}", result);

        // Should have 1 true element with 3 coordinates
        assertEquals(1, result.size(0));
        assertEquals(3, result.size(1));

        // Verify coordinates: (1, 0, 1)
        assertEquals(1, result.getLong(0, 0));
        assertEquals(0, result.getLong(0, 1));
        assertEquals(1, result.getLong(0, 2));
    }

    /**
     * Test Where with all true values
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereAllTrue(Nd4jBackend backend) {
        INDArray condition = Nd4j.ones(DataType.BOOL, 5);
        INDArray x = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray y = Nd4j.create(new double[]{10, 20, 30, 40, 50});

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition, x, y)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        INDArray result = results[0];

        // All condition true -> all from x
        assertEquals(x, result);
    }

    /**
     * Test Where with all false values (3-input mode)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereAllFalseThreeInput(Nd4jBackend backend) {
        INDArray condition = Nd4j.zeros(DataType.BOOL, 5);
        INDArray x = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray y = Nd4j.create(new double[]{10, 20, 30, 40, 50});

        DynamicCustomOp op = DynamicCustomOp.builder("Where")
                .addInputs(condition, x, y)
                .build();

        INDArray[] results = Nd4j.exec(op);

        assertNotNull(results);
        INDArray result = results[0];

        // All condition false -> all from y
        assertEquals(y, result);
    }
}
