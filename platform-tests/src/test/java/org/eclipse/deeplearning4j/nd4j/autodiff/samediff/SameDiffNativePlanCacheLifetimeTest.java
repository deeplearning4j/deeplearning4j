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
 *  *  SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A graph that executes through a DSP plan owns a native plan cache: its plans with their
 * device workspaces (a 256MB cuBLAS workspace each, by default). A graph the caller drops
 * without close() must give that memory back once it is collected, like its arrays do.
 * SameDiffTests, which rarely closes its graphs, exhausted the 128GB of a GB10 this way.
 * Its executors must also leave the frozen executor count, which otherwise keeps
 * InferenceSession from clearing the TAD cache for the rest of the process.
 */
@Slf4j
public class SameDiffNativePlanCacheLifetimeTest {

    private static final int GRAPHS = 12;
    /** Device memory the graphs may keep once collected: measurement noise. */
    private static final long TOLERANCE_BYTES = 256L * 1024 * 1024;

    @Test
    void collectedGraphsReleaseTheirNativePlans() {
        assumeTrue(Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA,
                "device memory is the CUDA backend's");
        NativeOps ops = Nd4j.getNativeOps();

        // The first DSP execution sets up process-wide state (streams, library handles,
        // kernels) that no graph gives back; keep it out of the measurement.
        runGraphs(1, true);
        reclaim();
        long heldBefore = heldDeviceBytes(ops);
        int frozenBefore = DynamicShapePlanExecutor.frozenExecutorCount();
        runGraphs(GRAPHS, true);
        reclaim();
        long heldAfterClosed = heldDeviceBytes(ops);
        List<WeakReference<SameDiff>> dropped = runGraphs(GRAPHS, false);
        reclaim();
        long heldAfterDropped = heldDeviceBytes(ops);

        long keptByClosed = heldAfterClosed - heldBefore;
        long keptByDropped = heldAfterDropped - heldAfterClosed;
        long reachable = dropped.stream().filter(ref -> ref.get() != null).count();
        log.info("[PLAN_CACHE_LIFETIME] {} graphs: closed kept {}MB, dropped without close kept {}MB; "
                + "{} dropped graphs still reachable after collection; memory pool reserves {}MB unused",
                GRAPHS, mb(keptByClosed), mb(keptByDropped), reachable, mb(poolCachedBytes(ops)));
        assertEquals(0, reachable, "graphs dropped without close() must be collectable");
        assertEquals(frozenBefore, DynamicShapePlanExecutor.frozenExecutorCount(),
                "executors of collected graphs must leave the frozen executor count");
        assertTrue(keptByClosed <= TOLERANCE_BYTES, GRAPHS + " closed graphs kept "
                + mb(keptByClosed) + "MB of device memory");
        assertTrue(keptByDropped <= TOLERANCE_BYTES, GRAPHS + " graphs dropped without close() kept "
                + mb(keptByDropped) + "MB of device memory after collection");
    }

    /**
     * One-op graphs, each executed once through its DSP plan.
     *
     * @return weak references to the graphs, which the caller no longer holds
     */
    private static List<WeakReference<SameDiff>> runGraphs(int count, boolean close) {
        List<WeakReference<SameDiff>> graphs = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", Nd4j.rand(DataType.FLOAT, 3, 4));
            sd.expandDims(in, 0).eval();
            if (close) {
                sd.close();
            }
            graphs.add(new WeakReference<>(sd));
        }
        return graphs;
    }

    /**
     * Device memory in use, less what the memory pool keeps reserved but unused. Memory a
     * graph holds is in use; a trim does not return the pool's free fragments to the device
     * (the dropped graphs' twelve workspaces, alive at once, leave a few hundred MB of them),
     * and the pool serves later allocations from them.
     */
    private static long heldDeviceBytes(NativeOps ops) {
        long used = ops.getDeviceTotalMemory(0) - ops.getDeviceFreeMemory(0);
        return used - poolCachedBytes(ops);
    }

    private static long poolCachedBytes(NativeOps ops) {
        try (LongPointer used = new LongPointer(1); LongPointer reserved = new LongPointer(1)) {
            ops.getMemoryPoolStats(0, used, reserved);
            return reserved.get() - used.get();
        }
    }

    /** The test base's reclamation: collect, free what collection found, trim the pools. */
    private static void reclaim() {
        for (int i = 0; i < 3; i++) {
            System.gc();
        }
        Nd4j.getDeallocatorService().forceFlushAll();
        Nd4j.getExecutioner().commit();
        for (int d = 0; d < Nd4j.getAffinityManager().getNumberOfDevices(); d++) {
            Nd4j.getNativeOps().trimMemoryPool(d);
        }
    }

    private static long mb(long bytes) {
        return bytes / (1024 * 1024);
    }
}
