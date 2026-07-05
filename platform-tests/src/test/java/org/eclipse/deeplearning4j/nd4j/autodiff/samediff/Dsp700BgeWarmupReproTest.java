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
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SameDiffSerializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * In-tree repro for the kompile embedding-lane CUDA 700 (task #57):
 * bge-base-en-v1.5 DSP warmup at [32 x 512] under the production flag set
 * fails with "slot 579 (xw_plus_b) ... transformAny(...) cached kernel failed
 * cudaStreamSynchronize error code [700]" and poisons the CUDA context.
 *
 * Root cause chain: the warmup's dot_product_attention_v2 materializes
 * [32,12,512,512] fp32 scores (402MB per layer). When device memory is
 * already consumed (in production: the LFM2-extract model loaded and resident
 * FIRST, then bge loaded after), those allocations OOM and take
 * CudaMemoryPool::allocateFailover. On non-peer device pairs the managed
 * fallback used to prefetch on the DEFAULT stream — unordered against the
 * consuming kernels — so demand paging raced kernel access and surfaced as
 * error 700 at arbitrary downstream sync points.
 *
 * {@link #testBgeWarmupFailoverUnderMemoryPressure()} reproduces that
 * mechanism deterministically: it ballasts the device below one attention
 * scores allocation so the warmup MUST route through allocateFailover —
 * no second model required, pressure source is irrelevant to the bug.
 *
 * The model is an external artifact (~/.kompile/models/bge-base-en-v1.5) —
 * the test SKIPS when absent, so CI without the model is unaffected.
 *
 * Run with the production flags (they arrive as system properties):
 *   mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest='Dsp700BgeWarmupReproTest' \
 *     -Dnd4j.triton.graphCapture=true -Dnd4j.triton.compileAll=true \
 *     -Dnd4j.triton.sectionFusion=true -Dnd4j.triton.consolidatedArgTable=true \
 *     -Dnd4j.triton.argDirtyTracking=true -Dnd4j.triton.tf32=true \
 *     -Dnd4j.cublas.tf32=true -Dnd4j.cublas.captureWorkspace=1 \
 *     -Dnd4j.dsp.batchedGemm=true -Dnd4j.dsp.freezeMergeSegments=true \
 *     -Dnd4j.optimizer.enabled=true -Dnd4j.optimizer.fp16=true
 *
 * Model variant via -Dbge.model.file (default model.opt.sdz; plain model.sdz
 * gives the optimizer-on/off discriminator).
 */
@Slf4j
@Tag("manual-repro")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class Dsp700BgeWarmupReproTest {

    private static final String MODEL_DIR =
            System.getProperty("bge.model.dir",
                    System.getProperty("user.home") + "/.kompile/models/bge-base-en-v1.5");
    private static final String MODEL_FILE = System.getProperty("bge.model.file", "model.opt.sdz");
    private static final int BATCH = Integer.getInteger("bge.batch", 32);
    private static final int SEQ = Integer.getInteger("bge.seq", 512);
    // Bounded shape for the deterministic failover method — small enough that the
    // whole DSP plan fits on-device, so ONLY the ballast forces the failover path.
    private static final int P_BATCH = Integer.getInteger("bge.pressure.batch", 8);
    private static final int P_SEQ = Integer.getInteger("bge.pressure.seq", 256);
    // bge-base attention head count — sizes the scores tensor the warmup materializes.
    private static final int HEADS = 12;

    /**
     * The failover-concept repro (deterministic, bounded, runs on any box with the
     * model): a SMALL-shape plan that comfortably fits the device, then ballast
     * free device memory below one attention-scores allocation so the warmup's
     * attention temps are FORCED through CudaMemoryPool::allocateFailover
     * (OOM -> managed host-resident fallback), consumed by async GEMMs, and
     * freed through the event-deferred direct-free path. Green means the whole
     * OOM -> failover -> consume -> deferred-free lifecycle is ordered — the run
     * completes or fails with a clean OOM, never error 700. This is the
     * mechanism the production LFM-then-bge sequence triggers: any prior
     * resident model is just ballast with extra steps.
     */
    @Test
    @Order(1)
    public void testBgeWarmupFailoverUnderMemoryPressure() throws Exception {
        SameDiff sd = loadModelOrSkip();
        Map<String, INDArray> feed = buildFeed(sd, P_BATCH, P_SEQ);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        long scoresBytes = 4L * P_BATCH * HEADS * P_SEQ * P_SEQ;
        long targetFree = Math.min(256L * 1024 * 1024, scoresBytes / 2);

        // DEVICE-ONLY ballast (allocateBoth=false): INDArray ballast drags a host
        // primary along with every device chunk, which trips JavaCPP's
        // maxPhysicalBytes guard long before the device fills.
        List<OpaqueDataBuffer> ballast = new ArrayList<>();
        try {
            long free = nativeOps.getDeviceFreeMemory(device);
            log.info("device {} free before ballast: {} MB (target < {} MB, scores alloc = {} MB)",
                    device, free >> 20, targetFree >> 20, scoresBytes >> 20);
            // Adaptive chunking: big strides far from the target, small ones close to
            // it. If free memory stops dropping the ballast alloc itself failed over
            // (off-device) — further chunks change nothing, so stop and let the
            // precondition assert decide.
            while (free > targetFree + 8L * 1024 * 1024) {
                long chunkBytes = Math.max(8L * 1024 * 1024,
                        Math.min(256L * 1024 * 1024, free - targetFree - 4L * 1024 * 1024));
                ballast.add(OpaqueDataBuffer.allocateDataBuffer(chunkBytes / 4, DataType.FLOAT, false));
                long nextFree = nativeOps.getDeviceFreeMemory(device);
                if (nextFree >= free - chunkBytes / 2) {
                    log.info("ballast chunk did not reduce device free ({} MB -> {} MB) — stopping",
                            free >> 20, nextFree >> 20);
                    break;
                }
                free = nextFree;
            }
            log.info("device {} free after {} ballast chunks: {} MB", device, ballast.size(), free >> 20);
            // Failover engages via EITHER trigger: hard OOM (free < one scores alloc)
            // or the pool's proactive soft limit (usage above the threshold routes
            // allocations off-device — free memory FLOORS there because ballast
            // chunks themselves start failing over; observed ~208MB on a 24GB card).
            long total = nativeOps.getDeviceTotalMemory(device);
            boolean softLimitRegion = total > 0 && free < total / 50; // ≤2% free
            assertTrue(free < scoresBytes || softLimitRegion,
                    "ballast failed to push free memory (" + (free >> 20) + " MB) below one scores " +
                            "allocation (" + (scoresBytes >> 20) + " MB) or into the soft-limit region — " +
                            "failover path not exercised");

            execTwice(sd, feed, "under-pressure");
        } finally {
            for (OpaqueDataBuffer b : ballast) {
                try {
                    Nd4j.getNativeOps().dbClose(b);
                } catch (Throwable t) {
                    log.warn("ballast close failed: {}", t.toString());
                }
            }
        }
    }

    /**
     * Production-shape repro ([32 x 512], kompile embedding lane). CAPACITY NOTE:
     * this plan legitimately wants ~23GB of device slot buffers plus tens of GB
     * of host memory (host primaries + host-resident failover overflow) — run
     * with -Dtest.maxphysicalbytes=96g on a 24GB card, and expect a SLOW warmup
     * (overflow buffers run at PCIe speed). The bug this guards against poisons
     * the context with error 700 within the first two failover events; a clean
     * physical-memory-limit abort is a CAPACITY result, not the bug.
     */
    @Test
    @Order(2)
    public void testBgeWarmupShape32x512() throws Exception {
        SameDiff sd = loadModelOrSkip();
        Map<String, INDArray> feed = buildFeed(sd, BATCH, SEQ);
        execTwice(sd, feed, "clean-box");
    }

    private SameDiff loadModelOrSkip() throws Exception {
        File model = new File(MODEL_DIR, MODEL_FILE);
        assumeTrue(model.isFile(), "bge model not present: " + model);
        log.info("Loading {} ({} MB)", model, model.length() / (1024 * 1024));
        return SameDiffSerializer.load(model, false);
    }

    private Map<String, INDArray> buildFeed(SameDiff sd, int batch, int seq) {
        Map<String, INDArray> feed = new LinkedHashMap<>();
        for (String in : sd.inputs()) {
            DataType dt = sd.getVariable(in).dataType();
            // Token-style inputs: ids get a small nonzero id, masks get ones —
            // all-zero masks degenerate attention and hide nothing.
            INDArray arr;
            String lower = in.toLowerCase();
            if (lower.contains("mask")) {
                arr = Nd4j.ones(dt, batch, seq);
            } else if (lower.contains("type")) {
                arr = Nd4j.zeros(dt, batch, seq);
            } else {
                arr = Nd4j.valueArrayOf(new long[]{batch, seq}, 101, dt); // [CLS]-ish id
            }
            feed.put(in, arr);
            log.info("feed '{}' dtype={} shape=[{}, {}]", in, dt, batch, seq);
        }
        return feed;
    }

    /** Exec twice: warmup/compile+capture, then the replay-path exec. */
    private void execTwice(SameDiff sd, Map<String, INDArray> feed, String label) {
        List<String> outputs = sd.outputs();
        for (int i = 0; i < 2; i++) {
            long t0 = System.currentTimeMillis();
            Map<String, INDArray> out = sd.output(feed, outputs);
            long ms = System.currentTimeMillis() - t0;
            INDArray first = out.values().iterator().next();
            double mean = first.meanNumber().doubleValue();
            log.info("[{}] exec #{} OK in {} ms — out shape={} mean={}", label, i, ms,
                    java.util.Arrays.toString(first.shape()), mean);
            assertTrue(Double.isFinite(mean), "[" + label + "] exec #" + i + " produced non-finite mean");
        }
    }
}
