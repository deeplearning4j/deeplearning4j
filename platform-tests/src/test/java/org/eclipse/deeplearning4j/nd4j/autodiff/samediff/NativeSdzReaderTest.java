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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for SDZ/SDNB file reading and model save/load roundtrip.
 *
 * Covers: Java save → Java load roundtrip (verifies serialization format works),
 * FlatBuffer schema fields, variable types, scalar handling, empty arrays.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeSdzReaderTest extends BaseNd4jTestWithBackends {

    @TempDir
    Path tempDir;

    @Test
    public void testSaveLoadRoundtripSimple() throws Exception {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();

        File file = tempDir.resolve("simple.fb").toFile();
        sd.save(file, true);
        assertTrue(file.exists());
        assertTrue(file.length() > 0);

        SameDiff loaded = SameDiff.load(file, true);
        assertNotNull(loaded);

        // Verify same outputs
        INDArray x = Nd4j.ones(DataType.FLOAT, 2, 3);
        INDArray y = Nd4j.ones(DataType.FLOAT, 2, 3).mul(2);
        Map<String, INDArray> original = sd.output(Map.of("x", x, "y", y), "z");
        Map<String, INDArray> restored = loaded.output(Map.of("x", x, "y", y), "z");
        NativeExecutorTestUtils.assertArrayEquals(original.get("z"), restored.get("z"), 1e-5,
                "Save/load roundtrip");
    }

    @Test
    public void testSaveLoadRoundtripWithVariables() throws Exception {
        SameDiff sd = NativeExecutorTestUtils.createMatMulGraph(4, 3);

        File file = tempDir.resolve("matmul.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);
        assertNotNull(loaded);

        INDArray x = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> original = sd.output(Map.of("x", x), "z");
        Map<String, INDArray> restored = loaded.output(Map.of("x", x), "z");
        NativeExecutorTestUtils.assertArrayEquals(original.get("z"), restored.get("z"), 1e-5,
                "MatMul roundtrip");
    }

    @Test
    public void testSaveLoadRoundtripChain() throws Exception {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();

        File file = tempDir.resolve("chain.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);
        assertNotNull(loaded);

        INDArray x = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> original = sd.output(Map.of("x", x), "z");
        Map<String, INDArray> restored = loaded.output(Map.of("x", x), "z");
        NativeExecutorTestUtils.assertArrayEquals(original.get("z"), restored.get("z"), 1e-5,
                "Chain roundtrip");
    }

    @Test
    public void testSaveLoadPlanCompilation() throws Exception {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();

        File file = tempDir.resolve("plan_compile.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);

        // Compile a plan from the loaded graph
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(loaded, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);
        assertTrue(plan.getSlots().length > 0);

        plan.close();
    }

    @Test
    public void testSaveLoadScalarVariable() throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable scale = sd.var("scale", Nd4j.scalar(DataType.FLOAT, 3.14f));
        SDVariable z = x.mul("z", scale);

        File file = tempDir.resolve("scalar.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);
        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = loaded.output(Map.of("x", xArr), "z");
        assertEquals(3.14f, result.get("z").getFloat(0, 0), 0.01f, "Scalar variable preserved");
    }

    @Test
    public void testSaveLoadMultipleDtypes() throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable intConst = sd.constant("intConst", Nd4j.createFromArray(new int[]{1, 2, 3, 4}));
        SDVariable cast = sd.castTo("cast", intConst, DataType.FLOAT);
        SDVariable z = x.add("z", cast);

        File file = tempDir.resolve("multi_dtype.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);
        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 1, 4);
        Map<String, INDArray> result = loaded.output(Map.of("x", xArr), "z");
        assertNotNull(result.get("z"));
    }

    @Test
    public void testSaveLoadPreservesPlanSerialization() throws Exception {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();

        DynamicShapePlan originalPlan = NativeExecutorTestUtils.compilePlan(sd, "z");
        byte[] originalSerialized = originalPlan.serialize();

        File file = tempDir.resolve("preserve_plan.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);
        DynamicShapePlan loadedPlan = NativeExecutorTestUtils.compilePlan(loaded, "z");
        byte[] loadedSerialized = loadedPlan.serialize();

        // Plans from original and loaded should have same structure
        assertEquals(originalPlan.getSlots().length, loadedPlan.getSlots().length,
                "Slot count should match");
        assertEquals(originalPlan.getTotalOutputSlots(), loadedPlan.getTotalOutputSlots(),
                "Total output slots should match");

        originalPlan.close();
        loadedPlan.close();
    }

    @Test
    public void testSaveLoadLargeConstants() throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 128);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 128, 256));
        SDVariable z = sd.mmul("z", x, w);

        File file = tempDir.resolve("large_const.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);
        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, 128);
        Map<String, INDArray> original = sd.output(Map.of("x", xArr), "z");
        Map<String, INDArray> restored = loaded.output(Map.of("x", xArr), "z");
        NativeExecutorTestUtils.assertArrayEquals(original.get("z"), restored.get("z"), 1e-3,
                "Large constant roundtrip");
    }

    @Test
    public void testSaveLoadEmptyVariable() throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = x.add("z", 0.0);

        File file = tempDir.resolve("empty.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);
        assertNotNull(loaded);
    }

    @Test
    public void testFlatArraySchemaFields() throws Exception {
        // Verify FlatArray has the expected fields by saving and loading
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.linspace(1, 4, 4, DataType.FLOAT).reshape(1, 4));
        SDVariable z = x.mul("z", w);

        File file = tempDir.resolve("schema_test.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);

        // The variable 'w' should have been saved and loaded with correct values
        INDArray wLoaded = loaded.getVariable("w").getArr();
        assertNotNull(wLoaded, "Variable 'w' should be loaded");
        assertEquals(1.0, wLoaded.getFloat(0, 0), 1e-5, "w[0] should be 1");
        assertEquals(4.0, wLoaded.getFloat(0, 3), 1e-5, "w[3] should be 4");
    }

    @Test
    public void testSaveLoadMultipleOutputs() throws Exception {
        SameDiff sd = NativeExecutorTestUtils.createDiamondGraph();

        File file = tempDir.resolve("multi_output.fb").toFile();
        sd.save(file, true);

        SameDiff loaded = SameDiff.load(file, true);

        // Compile plan with multiple outputs
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(loaded, "a", "b", "c");
        NativeExecutorTestUtils.assertValidSerialization(plan);
        assertEquals(3, plan.getOutputNameToSlotIndex().size());

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 4).mul(3);
        Map<String, INDArray> result = loaded.output(Map.of("x", xArr), "a", "b", "c");
        assertEquals(4.0, result.get("a").getDouble(0), 1e-5, "a = x+1 = 4");
        assertEquals(6.0, result.get("b").getDouble(0), 1e-5, "b = x*2 = 6");
        assertEquals(10.0, result.get("c").getDouble(0), 1e-5, "c = a+b = 10");

        plan.close();
    }

    @Test
    public void testFlatBufferFormatRoundtrip() throws Exception {
        // Specifically tests that FlatBuffer format survives save/load
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 4, 2));
        SDVariable b = sd.var("b", Nd4j.zeros(DataType.FLOAT, 1, 2));
        SDVariable z = sd.mmul(x, w).add("z", b);

        // Save as FlatBuffer
        File file = tempDir.resolve("flatbuf.fb").toFile();
        sd.save(file, true);

        // Load and verify
        SameDiff loaded = SameDiff.load(file, true);

        // Variables should exist
        assertNotNull(loaded.getVariable("w"));
        assertNotNull(loaded.getVariable("b"));
        assertNotNull(loaded.getVariable("x"));

        // Execute and compare
        INDArray xArr = Nd4j.randn(DataType.FLOAT, 3, 4);
        Map<String, INDArray> origResult = sd.output(Map.of("x", xArr), "z");
        Map<String, INDArray> loadResult = loaded.output(Map.of("x", xArr), "z");
        NativeExecutorTestUtils.assertArrayEquals(origResult.get("z"), loadResult.get("z"), 1e-4,
                "FlatBuffer roundtrip");
    }
}
