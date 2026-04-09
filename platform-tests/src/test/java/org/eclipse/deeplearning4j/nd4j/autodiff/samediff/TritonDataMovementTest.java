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

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for data movement operations with Triton GPU backend.
 * Covers: gather, concat, stridedSlice, tile, reshape, permute, squeeze, expandDims.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonDataMovementTest extends BaseNd4jTestWithBackends {

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Gather ──────────────────────────────────────────────────────────────

    @Test @DisplayName("Gather: Axis 0, rows [16, 8] -> indices [0,3,7,1] -> [4, 8]")
    public void testGatherAxis0() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable gathered = sd.gather("gather", x, new int[]{0, 3, 7, 1}, 0);
        sd.nn().relu("result", gathered, 0);
        TritonTestUtils.runOpTest("testGatherAxis0", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 16, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Gather: Axis 1, columns [4, 16] -> indices [0,5,10] -> [4, 3]")
    public void testGatherAxis1() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, -1);
        SDVariable gathered = sd.gather("gather", x, new int[]{0, 5, 10}, 1);
        sd.nn().relu("result", gathered, 0);
        TritonTestUtils.runOpTest("testGatherAxis1", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 16)), "result");
        sd.close();
    }

    // ─── Concat ──────────────────────────────────────────────────────────────

    @Test @DisplayName("Concat: Axis 0, [2, 8] + [3, 8] -> [5, 8]")
    public void testConcatAxis0() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 3, 8));
        sd.concat("result", 0, x, y);
        TritonTestUtils.runOpTest("testConcatAxis0", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 2, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Concat: Axis 1, [4, 8] + [4, 8] -> [4, 16]")
    public void testConcatAxis1() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 4, 8));
        sd.concat("result", 1, x, y);
        TritonTestUtils.runOpTest("testConcatAxis1", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Concat: 3 tensors along axis 1, [4, 4] x 3 -> [4, 12]")
    public void testConcat3Tensors() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.constant("y", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable z = sd.constant("z", Nd4j.randn(DataType.FLOAT, 4, 4));
        sd.concat("result", 1, x, y, z);
        TritonTestUtils.runOpTest("testConcat3Tensors", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 4)), "result");
        sd.close();
    }

    // ─── StridedSlice ────────────────────────────────────────────────────────

    @Test @DisplayName("StridedSlice: Basic [8, 16] -> [4, 8]")
    public void testStridedSliceBasic() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        sd.stridedSlice("result", x, new long[]{0, 0}, new long[]{4, 8}, new long[]{1, 1});
        TritonTestUtils.runOpTest("testStridedSliceBasic", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    @Test @DisplayName("StridedSlice: Stride 2 [8, 16] -> [4, 8]")
    public void testStridedSliceStride2() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        sd.stridedSlice("result", x, new long[]{0, 0}, new long[]{8, 16}, new long[]{2, 2});
        TritonTestUtils.runOpTest("testStridedSliceStride2", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }

    // ─── Tile ────────────────────────────────────────────────────────────────

    @Test @DisplayName("Tile: [4, 8] -> tile(1, 3) -> [4, 24]")
    public void testTile() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        sd.tile("result", x, 1, 3);
        TritonTestUtils.runOpTest("testTile", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Reshape ─────────────────────────────────────────────────────────────

    @Test @DisplayName("Reshape: [4, 8] -> [2, 16]")
    public void testReshape() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        sd.reshape("result", x, 2, 16);
        TritonTestUtils.runOpTest("testReshape", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    // ─── Permute ─────────────────────────────────────────────────────────────
    // NOTE: permute produces a strided view (non-contiguous). The native execution
    // path copies the raw buffer into a contiguous INDArray, which does not respect
    // the stride reordering. Chaining a downstream op (relu) forces materialization
    // of the view into correct contiguous layout before comparison.
    // TODO: Fix native execution to handle strided-view outputs properly, then
    // remove the relu materialization and test standalone permute.

    @Test @DisplayName("Permute: 2D transpose [4, 8] -> [8, 4] via permute(1, 0)")
    public void testPermute2D() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable permuted = sd.permute("permuted", x, 1, 0);
        sd.nn().relu("result", permuted, 0);
        TritonTestUtils.runOpTest("testPermute2D", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Permute: 3D [4, 8, 16] -> permute(2, 0, 1) -> [16, 4, 8]")
    public void testPermute3D() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8, 16);
        SDVariable permuted = sd.permute("permuted", x, 2, 0, 1);
        sd.nn().relu("result", permuted, 0);
        TritonTestUtils.runOpTest("testPermute3D", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8, 16)), "result");
        sd.close();
    }

    // ─── Squeeze ─────────────────────────────────────────────────────────────

    @Test @DisplayName("Squeeze: [4, 1, 8] -> squeeze(1) -> [4, 8]")
    public void testSqueeze() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 1, 8);
        SDVariable squeezed = sd.squeeze("squeeze", x, 1);
        sd.nn().relu("result", squeezed, 0);
        TritonTestUtils.runOpTest("testSqueeze", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 1, 8)), "result");
        sd.close();
    }

    // ─── ExpandDims ──────────────────────────────────────────────────────────

    @Test @DisplayName("ExpandDims: [4, 8] -> expand(1) -> [4, 1, 8]")
    public void testExpandDims() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable expanded = sd.expandDims("expand", x, 1);
        sd.nn().relu("result", expanded, 0);
        TritonTestUtils.runOpTest("testExpandDims", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
        sd.close();
    }

    @Test @DisplayName("Gather + Concat chain [8, 16]")
    public void testGatherConcatChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable g1 = sd.gather("g1", x, new int[]{0, 1, 2}, 0);
        SDVariable g2 = sd.gather("g2", x, new int[]{4, 5, 6}, 0);
        sd.concat("result", 0, g1, g2);
        TritonTestUtils.runOpTest("testGatherConcatChain", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 8, 16)), "result");
        sd.close();
    }
}
