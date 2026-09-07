/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.TrainingSession;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Allocation-boundary tests shared by inference and training. No DSP execution is
 * needed: every copy returned by cast/dup must already have a cleanup owner before
 * compilation, migration, or even the next placeholder's allocation can fail.
 */
public class PlaceholderMaterializationOwnershipTest extends BaseND4JTest {
    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void castAndViewCopiesAreOwnedBeforeExecution(boolean training) throws Exception {
        try (SameDiff sd = graph();
             INDArray caller = Nd4j.ones(DataType.DOUBLE, 4);
             INDArray root = Nd4j.ones(DataType.FLOAT, 8);
             INDArray unchanged = Nd4j.ones(DataType.FLOAT, 4)) {
            InferenceSession session = session(sd, training);
            List<INDArray> copies = new ArrayList<>();
            try {
                INDArray view = root.get(NDArrayIndex.interval(0, 2, 8));
                assertTrue(view.isView());
                Map<String, INDArray> inputs = inputs(caller, view);
                inputs.put("unchanged", unchanged);
                Map<String, INDArray> output = materialize(session, inputs);
                copies.add(output.get("a"));
                copies.add(output.get("b"));
                assertSame(caller, inputs.get("a"));
                assertSame(view, inputs.get("b"));
                assertSame(unchanged, output.get("unchanged"));
                assertNotSame(caller, output.get("a"));
                assertNotSame(view, output.get("b"));
                assertEquals(DataType.FLOAT, output.get("a").dataType());
                assertFalse(output.get("b").isView());
                assertFalse(output.get("a").isAttached());
                assertFalse(output.get("b").isAttached());
                assertEquals(2, owned(session).size(), "Copies must be registered inside the helper");
                assertTrue(owns(session, output.get("a")));
                assertTrue(owns(session, output.get("b")));
                assertFalse(owns(session, unchanged));
                DataBuffer castBuffer = output.get("a").data();
                DataBuffer viewBuffer = output.get("b").data();
                session.closePooledResources();
                assertTrue(castBuffer.wasClosed());
                assertTrue(viewBuffer.wasClosed());
                assertTrue(owned(session).isEmpty());
                assertFalse(caller.data().wasClosed());
                assertFalse(root.data().wasClosed());
                assertFalse(unchanged.data().wasClosed());
            } finally {
                finish(session, copies);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void copiesOutliveTheCallersWorkspace(boolean training) throws Exception {
        try (SameDiff sd = graph();
             INDArray caller = Nd4j.ones(DataType.DOUBLE, 4);
             INDArray root = Nd4j.ones(DataType.FLOAT, 8)) {
            InferenceSession session = session(sd, training);
            List<INDArray> copies = new ArrayList<>();
            MemoryWorkspace previous = Nd4j.getMemoryManager().getCurrentWorkspace();
            MemoryWorkspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace(
                    WorkspaceConfiguration.builder().initialSize(1024 * 1024).build(),
                    "placeholder-materialization-" + training);
            try {
                try (MemoryWorkspace ignored = workspace.notifyScopeEntered()) {
                    Map<String, INDArray> output = materialize(session,
                            inputs(caller, root.get(NDArrayIndex.interval(0, 2, 8))));
                    copies.add(output.get("a"));
                    copies.add(output.get("b"));
                    assertSame(workspace, Nd4j.getMemoryManager().getCurrentWorkspace(),
                            "Materialization must restore the caller's active workspace");
                    for (INDArray copy : copies) {
                        assertFalse(copy.isAttached(), "A pinned handle may outlive this workspace");
                        assertTrue(owns(session, copy));
                    }
                }
                assertSame(previous, Nd4j.getMemoryManager().getCurrentWorkspace());
                for (INDArray copy : copies) {
                    assertEquals(4.0, copy.sumNumber().doubleValue(), 0.0);
                }
                DataBuffer castBuffer = copies.get(0).data();
                DataBuffer viewBuffer = copies.get(1).data();
                session.closePooledResources();
                assertTrue(castBuffer.wasClosed());
                assertTrue(viewBuffer.wasClosed());
                assertFalse(caller.data().wasClosed());
                assertFalse(root.data().wasClosed());
            } finally {
                try {
                    finish(session, copies);
                } finally {
                    Nd4j.getWorkspaceManager().destroyWorkspace(workspace);
                }
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void registrationPrecedesTheNextAllocation(boolean training) throws Exception {
        try (SameDiff sd = graph();
             INDArray caller = Nd4j.ones(DataType.DOUBLE, 4);
             INDArray firstCopy = Nd4j.ones(DataType.FLOAT, 4);
             INDArray secondCopy = Nd4j.ones(DataType.FLOAT, 4)) {
            InferenceSession session = session(sd, training);
            try {
                INDArray first = intercept(caller, "castTo", () -> firstCopy);
                INDArray second = intercept(caller, "castTo", () -> {
                    assertTrue(ownsUnchecked(session, firstCopy),
                            "First allocation must have an owner before the second allocation starts");
                    return secondCopy;
                });
                materialize(session, inputs(first, second));
                assertTrue(owns(session, firstCopy));
                assertTrue(owns(session, secondCopy));
                assertEquals(2, owned(session).size());
                assertFalse(owns(session, caller));
            } finally {
                session.closePooledResources();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void partialCastFailureClosesOnlyThisCallsCopies(boolean training) throws Exception {
        try (SameDiff sd = graph();
             INDArray caller = Nd4j.ones(DataType.DOUBLE, 4);
             INDArray olderCopy = Nd4j.ones(DataType.FLOAT, 4);
             INDArray newCopy = Nd4j.ones(DataType.FLOAT, 4)) {
            InferenceSession session = session(sd, training);
            try {
                materialize(session, Map.of("a", intercept(caller, "castTo", () -> olderCopy)));
                IllegalStateException allocationFailure = new IllegalStateException("second cast failed");
                INDArray first = intercept(caller, "castTo", () -> newCopy);
                INDArray second = intercept(caller, "castTo", () -> { throw allocationFailure; });
                DataBuffer newBuffer = newCopy.data();
                assertSame(allocationFailure, assertThrows(IllegalStateException.class,
                        () -> materialize(session, inputs(first, second))));
                assertTrue(newBuffer.wasClosed(), "No DSP borrower exists when casting itself fails");
                assertFalse(owns(session, newCopy));
                assertTrue(owns(session, olderCopy), "Do not reclaim earlier calls' potential borrowers");
                assertFalse(olderCopy.data().wasClosed());
                assertFalse(caller.data().wasClosed());
                assertEquals(1, owned(session).size());
            } finally {
                session.closePooledResources();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedPartialCleanupPreservesOwnerAndPrimaryException(boolean training) throws Exception {
        try (SameDiff sd = graph();
             INDArray caller = Nd4j.ones(DataType.DOUBLE, 4);
             INDArray copy = Nd4j.ones(DataType.FLOAT, 4)) {
            InferenceSession session = session(sd, training);
            AtomicBoolean rejectClose = new AtomicBoolean(true);
            IllegalStateException closeFailure = new IllegalStateException("close failed");
            INDArray ownedCopy = intercept(copy, "close", () -> {
                if (rejectClose.get()) throw closeFailure;
                copy.close();
                return null;
            });
            try {
                IllegalStateException allocationFailure = new IllegalStateException("second cast failed");
                INDArray first = intercept(caller, "castTo", () -> ownedCopy);
                INDArray second = intercept(caller, "castTo", () -> { throw allocationFailure; });
                Throwable thrown = assertThrows(IllegalStateException.class,
                        () -> materialize(session, inputs(first, second)));
                assertSame(allocationFailure, thrown);
                assertTrue(hasSuppressedFailure(thrown, closeFailure),
                        "Cleanup must not swallow its failure or replace the allocation failure");
                assertTrue(owns(session, ownedCopy), "Failed close must remain retryable at teardown");
                assertFalse(copy.data().wasClosed());
                assertFalse(caller.data().wasClosed());
                rejectClose.set(false);
                DataBuffer buffer = copy.data();
                session.closePooledResources();
                assertTrue(buffer.wasClosed());
                assertTrue(owned(session).isEmpty());
            } finally {
                rejectClose.set(false);
                session.closePooledResources();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void noOpCloseDoesNotLoseOwnership(boolean training) throws Exception {
        try (SameDiff sd = graph();
             INDArray caller = Nd4j.ones(DataType.DOUBLE, 4);
             INDArray copy = Nd4j.ones(DataType.FLOAT, 4)) {
            InferenceSession session = session(sd, training);
            AtomicBoolean noOp = new AtomicBoolean(true);
            INDArray ownedCopy = intercept(copy, "close", () -> {
                if (!noOp.get()) copy.close();
                return null;
            });
            try {
                materialize(session, Map.of("a", intercept(caller, "castTo", () -> ownedCopy)));
                assertTrue(owns(session, ownedCopy));
                DataBuffer buffer = copy.data();
                assertThrows(IllegalStateException.class, session::closePooledResources,
                        "Returning from close is not evidence that the storage was released");
                assertTrue(owns(session, ownedCopy));
                assertFalse(buffer.wasClosed());
                assertFalse(caller.data().wasClosed());
                noOp.set(false);
                session.closePooledResources();
                assertTrue(buffer.wasClosed());
                assertTrue(owned(session).isEmpty());
            } finally {
                noOp.set(false);
                session.closePooledResources();
            }
        }
    }

    private static SameDiff graph() {
        SameDiff sd = SameDiff.create();
        sd.placeHolder("a", DataType.FLOAT, -1);
        sd.placeHolder("b", DataType.FLOAT, -1);
        sd.placeHolder("unchanged", DataType.FLOAT, -1);
        return sd;
    }

    private static InferenceSession session(SameDiff sd, boolean training) {
        return training ? new TrainingSession(sd) : new InferenceSession(sd);
    }

    private static Map<String, INDArray> inputs(INDArray first, INDArray second) {
        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("a", first);
        inputs.put("b", second);
        return inputs;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, INDArray> materialize(InferenceSession session,
                                                    Map<String, INDArray> inputs) throws Exception {
        Method method = InferenceSession.class.getDeclaredMethod("castPlaceholderTypes", Map.class);
        method.setAccessible(true);
        try {
            return (Map<String, INDArray>) method.invoke(session, inputs);
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof RuntimeException) throw (RuntimeException) e.getCause();
            if (e.getCause() instanceof Error) throw (Error) e.getCause();
            throw e;
        }
    }

    @SuppressWarnings("unchecked")
    private static Collection<INDArray> owned(InferenceSession session) throws Exception {
        Field field = InferenceSession.class.getDeclaredField("pendingCastPlaceholderCloses");
        field.setAccessible(true);
        return new ArrayList<>((Collection<INDArray>) field.get(session));
    }

    private static boolean owns(InferenceSession session, INDArray array) throws Exception {
        // Never INDArray.equals(): it reads data and may be called after that data is closed.
        return owned(session).stream().anyMatch(candidate -> candidate == array);
    }

    private static boolean ownsUnchecked(InferenceSession session, INDArray array) {
        try {
            return owns(session, array);
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    /** Fault injection at the INDArray boundary without adding a production test hook. */
    private static INDArray intercept(INDArray delegate, String methodName, Supplier<?> action) {
        return (INDArray) Proxy.newProxyInstance(INDArray.class.getClassLoader(),
                new Class<?>[]{INDArray.class}, (proxy, method, args) -> {
                    if (method.getName().equals(methodName)) return action.get();
                    try {
                        return method.invoke(delegate, args);
                    } catch (InvocationTargetException e) {
                        throw e.getCause();
                    }
                });
    }

    private static boolean hasSuppressedFailure(Throwable primary, Throwable expected) {
        for (Throwable suppressed : primary.getSuppressed()) {
            for (Throwable cause = suppressed; cause != null; cause = cause.getCause()) {
                if (cause == expected) return true;
            }
        }
        return false;
    }

    private static void finish(InferenceSession session, List<INDArray> copies) {
        try {
            session.closePooledResources();
        } finally {
            // Also clean up against the pre-fix implementation, which forgets these owners.
            for (INDArray copy : copies) {
                if (!copy.wasClosed()) copy.close();
            }
        }
    }
}
