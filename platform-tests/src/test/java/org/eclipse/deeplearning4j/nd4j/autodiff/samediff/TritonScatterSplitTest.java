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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Map;

/**
 * Tests for scatter and split operations with Triton GPU backend.
 *
 * Key pattern for scatterNdUpdate through the DSP/Triton path:
 * - The reference (target/data) array MUST be a placeHolder, fed via the inputs map.
 *   Using sd.var() with an embedded array causes a FLOAT/DOUBLE type mismatch in
 *   ScatterNdUpdate.calculateOutputDataTypes() because the DSP compilation path
 *   does not preserve the original array's DataType for non-placeholder variables.
 * - indices and updates should be sd.constant() with explicitly typed INDArrays.
 * - Use Nd4j.createFromArray(new int[][]{...}) for index arrays to guarantee INT32 type.
 * - Use Nd4j.create(DataType.FLOAT, shape) or Nd4j.ones(DataType.FLOAT, shape) for
 *   update arrays to guarantee FLOAT type (never rely on default type inference).
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonScatterSplitTest extends BaseNd4jTestWithBackends {

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    @Test
    @DisplayName("ScatterNd: Basic update - scatter ones into rows 0 and 2 of a 4x4 zero matrix")
    public void testScatterNdBasic() {
        SameDiff sd = SameDiff.create();

        // Target must be a placeholder (not sd.var) to preserve DataType.FLOAT through DSP compilation
        SDVariable target = sd.placeHolder("target", DataType.FLOAT, 4, 4);

        // Indices: [[0], [2]] meaning "update rows 0 and 2". Use createFromArray to guarantee INT32.
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new int[][]{{0}, {2}}));

        // Updates: 2 rows of ones, shape [2, 4]. Explicitly DataType.FLOAT.
        SDVariable updates = sd.constant("updates", Nd4j.ones(DataType.FLOAT, 2, 4));

        sd.scatterNdUpdate("result", target, indices, updates);

        INDArray zeros = Nd4j.zeros(DataType.FLOAT, 4, 4);
        TritonTestUtils.runOpTest("testScatterNdBasic", sd, Map.of("target", zeros), "result");
        sd.close();
    }

    @Test
    @DisplayName("ScatterNd: 2D updates - scatter into a 4x8 target with 2x8 updates")
    public void testScatterNd2DUpdates() {
        SameDiff sd = SameDiff.create();

        // Target placeholder: 4 rows, 8 columns
        SDVariable target = sd.placeHolder("target", DataType.FLOAT, 4, 8);

        // Indices: [[1], [3]] - update rows 1 and 3
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new int[][]{{1}, {3}}));

        // Updates: 2x8 ones, explicitly FLOAT
        SDVariable updates = sd.constant("updates", Nd4j.ones(DataType.FLOAT, 2, 8));

        sd.scatterNdUpdate("result", target, indices, updates);

        INDArray dataArr = Nd4j.zeros(DataType.FLOAT, 4, 8);
        TritonTestUtils.runOpTest("testScatterNd2DUpdates", sd, Map.of("target", dataArr), "result");
        sd.close();
    }

    @Test
    @DisplayName("ScatterNd: Non-zero data - scatter into a target with existing values")
    public void testScatterNdNonZeroData() {
        SameDiff sd = SameDiff.create();

        // Target placeholder: 4x4 with existing data
        SDVariable target = sd.placeHolder("target", DataType.FLOAT, 4, 4);

        // Indices: [[0], [2]] - update rows 0 and 2
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new int[][]{{0}, {2}}));

        // Updates: 2x4 with value 5.0f, explicitly FLOAT
        SDVariable updates = sd.constant("updates", Nd4j.valueArrayOf(new long[]{2, 4}, 5.0f, DataType.FLOAT));

        sd.scatterNdUpdate("result", target, indices, updates);

        // Non-zero initial data: fill with 1.0f
        INDArray dataArr = Nd4j.valueArrayOf(new long[]{4, 4}, 1.0f, DataType.FLOAT);
        TritonTestUtils.runOpTest("testScatterNdNonZeroData", sd, Map.of("target", dataArr), "result");
        sd.close();
    }

    @Test @DisplayName("Split: Axis 0, [8, 8] -> 2 x [4, 8], then add")
    public void testSplitAxis0() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable[] splits = sd.split(new String[]{"split0", "split1"}, x, 2, 0);
        splits[0].add("result", splits[1]);
        TritonTestUtils.runOpTest("testSplitAxis0", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Split: Axis 1, [4, 16] -> 2 x [4, 8], then add")
    public void testSplitAxis1() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable[] splits = sd.split(new String[]{"split0", "split1"}, x, 2, 1);
        splits[0].add("result", splits[1]);
        TritonTestUtils.runOpTest("testSplitAxis1", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("Split + Downstream: [4, 16] -> 4 x [4, 4] -> concat")
    public void testSplitDownstream() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable[] splits = sd.split(new String[]{"s0", "s1", "s2", "s3"}, x, 4, 1);
        SDVariable c = splits[0].add("add01", splits[1]);
        SDVariable c2 = splits[2].add("add23", splits[3]);
        sd.concat("result", 1, c, c2);
        TritonTestUtils.runOpTest("testSplitDownstream", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 16)), "result");
        sd.close();
    }
}
