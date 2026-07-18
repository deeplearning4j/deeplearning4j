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

package org.eclipse.deeplearning4j.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.deeplearning4j.nn.layers.samediff.DL4JSameDiffMemoryMgr;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.autodiff.samediff.internal.memory.NoOpMemoryMgr;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.autodiff.samediff.internal.memory.WorkspaceSessionMemMgr;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@Tag(TagNames.WORKSPACES)
@Tag(TagNames.SMOKE)
@NativeTag
public class WorkspaceSessionMemMgrTest extends BaseNd4jTestWithBackends {

    @BeforeEach
    public void setUp() {
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @AfterEach
    public void tearDown() {
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBasicAllocationAndRelease(Nd4jBackend backend) {
        try (WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(10 * 1024 * 1024)) {
            mgr.scopeIn();

            // Allocate a non-detached array (from workspace)
            INDArray arr = mgr.allocate(false, DataType.FLOAT, 10, 10);
            assertNotNull(arr);
            assertEquals(DataType.FLOAT, arr.dataType());
            assertArrayEquals(new long[]{10, 10}, arr.shape());

            // Release should be a no-op (workspace handles lifecycle)
            mgr.release(arr);

            mgr.scopeOut();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDetachedArraysSurviveScope(Nd4jBackend backend) {
        INDArray detachedArr;
        try (WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(10 * 1024 * 1024)) {
            mgr.scopeIn();

            // Allocate a detached array (heap-allocated, survives scope)
            detachedArr = mgr.allocate(true, DataType.DOUBLE, 5, 5);
            assertNotNull(detachedArr);

            // Fill with values
            detachedArr.assign(42.0);

            mgr.scopeOut();

            // Detached array should still be usable after scope ends
            assertEquals(42.0, detachedArr.getDouble(0, 0), 1e-6);
            assertEquals(42.0, detachedArr.getDouble(4, 4), 1e-6);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScopeCycleReusesMemory(Nd4jBackend backend) {
        try (WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(10 * 1024 * 1024)) {
            // First cycle
            mgr.scopeIn();
            INDArray arr1 = mgr.allocate(false, DataType.FLOAT, 100, 100);
            assertNotNull(arr1);
            mgr.scopeOut();

            // Second cycle - should reuse workspace memory
            mgr.scopeIn();
            INDArray arr2 = mgr.allocate(false, DataType.FLOAT, 100, 100);
            assertNotNull(arr2);
            mgr.scopeOut();

            // Both allocations should succeed without running out of memory
            // (workspace memory is recycled between scope cycles)
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testUlikeAndDup(Nd4jBackend backend) {
        try (WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(10 * 1024 * 1024)) {
            mgr.scopeIn();

            INDArray original = Nd4j.create(DataType.FLOAT, 5, 5);
            original.assign(3.14);

            // ulike should create same-shaped array
            INDArray ulikeArr = mgr.ulike(original);
            assertNotNull(ulikeArr);
            assertArrayEquals(original.shape(), ulikeArr.shape());
            assertEquals(original.dataType(), ulikeArr.dataType());

            // dup should copy values
            INDArray dupArr = mgr.dup(original);
            assertNotNull(dupArr);
            assertEquals(3.14, dupArr.getDouble(0, 0), 1e-5);

            mgr.scopeOut();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScopeIdempotent(Nd4jBackend backend) {
        try (WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(10 * 1024 * 1024)) {
            // Multiple scopeIn calls should be safe
            mgr.scopeIn();
            mgr.scopeIn();  // Should be idempotent

            assertTrue(mgr.isScopeActive());

            INDArray arr = mgr.allocate(false, DataType.FLOAT, 10);
            assertNotNull(arr);

            mgr.scopeOut();
            assertFalse(mgr.isScopeActive());

            // Multiple scopeOut calls should be safe
            mgr.scopeOut();  // Should be idempotent
            assertFalse(mgr.isScopeActive());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAutoregressive_StyleRepeatedInference(Nd4jBackend backend) {
        // Simulate autoregressive inference: same-sized allocations repeated many times
        try (WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(10 * 1024 * 1024)) {
            for (int step = 0; step < 50; step++) {
                mgr.scopeIn();

                // Simulate intermediate arrays during a forward pass
                INDArray hidden = mgr.allocate(false, DataType.FLOAT, 1, 768);
                INDArray attention = mgr.allocate(false, DataType.FLOAT, 1, 12, 1, 64);
                INDArray ffn = mgr.allocate(false, DataType.FLOAT, 1, 3072);

                assertNotNull(hidden);
                assertNotNull(attention);
                assertNotNull(ffn);

                // Simulate an output that needs to escape the workspace
                INDArray logits = mgr.allocate(true, DataType.FLOAT, 1, 50257);
                assertNotNull(logits);

                mgr.scopeOut();

                // logits should still be accessible after scope
                assertEquals(DataType.FLOAT, logits.dataType());
                assertArrayEquals(new long[]{1, 50257}, logits.shape());
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFallbackWithoutScope(Nd4jBackend backend) {
        // Allocations without active scope should fall back to heap
        try (WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(10 * 1024 * 1024)) {
            assertFalse(mgr.isScopeActive());

            INDArray arr = mgr.allocate(false, DataType.FLOAT, 10, 10);
            assertNotNull(arr);
            assertEquals(DataType.FLOAT, arr.dataType());
            assertArrayEquals(new long[]{10, 10}, arr.shape());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceDescriptorLayoutPreservation(Nd4jBackend backend) {
        assertDescriptorLayoutPreservation(new WorkspaceSessionMemMgr(10 * 1024 * 1024));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayCacheDescriptorLayoutPreservation(Nd4jBackend backend) {
        assertDescriptorLayoutPreservation(new ArrayCacheMemoryMgr());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNoOpDescriptorLayoutPreservation(Nd4jBackend backend) {
        assertDescriptorLayoutPreservation(new NoOpMemoryMgr());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDl4jDescriptorLayoutPreservation(Nd4jBackend backend) {
        assertDescriptorLayoutPreservation(new DL4JSameDiffMemoryMgr(null, null, null, null));
    }

    private static void assertDescriptorLayoutPreservation(SessionMemMgr mgr) {
        long[] fShape = {3, 4};
        long[] emptyShape = {2, 0, 3};
        INDArray fTemplate = Nd4j.createUninitialized(DataType.FLOAT, fShape, 'f');
        INDArray emptyTemplate = Nd4j.emptyWithShape(emptyShape, DataType.FLOAT);

        mgr.scopeIn();
        try {
            INDArray fromLongDescriptor = mgr.allocate(false, fTemplate.shapeDescriptor());
            assertArrayEquals(fShape, fromLongDescriptor.shape(), "Long descriptor shape");
            assertArrayEquals(fTemplate.stride(), fromLongDescriptor.stride(), "Long descriptor strides");
            assertEquals('f', fromLongDescriptor.ordering(), "Long descriptor ordering");

            INDArray fromShapeInfo = mgr.allocateFromDescriptor(false, fTemplate.shapeInfoDataBuffer(), true);
            assertArrayEquals(fShape, fromShapeInfo.shape(), "Shape-info descriptor shape");
            assertArrayEquals(fTemplate.stride(), fromShapeInfo.stride(), "Shape-info descriptor strides");
            assertEquals('f', fromShapeInfo.ordering(), "Shape-info descriptor ordering");
            assertEquals(0.0, fromShapeInfo.sumNumber().doubleValue(), 0.0,
                    "requiresZeroed must still apply while preserving layout");

            INDArray emptyFromLongDescriptor = mgr.allocate(false, emptyTemplate.shapeDescriptor());
            assertTrue(emptyFromLongDescriptor.isEmpty(), "Long descriptor must preserve ARRAY_EMPTY");
            assertArrayEquals(emptyShape, emptyFromLongDescriptor.shape(), "Empty long descriptor shape");

            INDArray emptyFromShapeInfo = mgr.allocateFromDescriptor(false, emptyTemplate.shapeInfoDataBuffer(), true);
            assertTrue(emptyFromShapeInfo.isEmpty(), "Shape-info descriptor must preserve ARRAY_EMPTY");
            assertArrayEquals(emptyShape, emptyFromShapeInfo.shape(), "Empty shape-info descriptor shape");
        } finally {
            mgr.scopeOut();
            mgr.close();
            fTemplate.close();
            emptyTemplate.close();
        }
    }
}
