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

package org.eclipse.deeplearning4j.nd4j.linalg.framework.device;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspReplayTransferAnalytics;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.device.StubDeviceDescriptor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.framework.device.TransferDirection;
import org.nd4j.linalg.framework.device.TransferEvent;
import org.nd4j.linalg.framework.device.TransferReason;
import org.nd4j.linalg.framework.device.TransferSubsystem;
import org.nd4j.linalg.framework.stats.MemoryStats;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Measures REAL native memory impact of VLM page-reuse leak patterns and verifies fixes.
 *
 * <p>Uses {@link Pointer#physicalBytes()} for actual native allocation tracking,
 * {@link Nd4j#framework} subsystem APIs for off-heap/lifecycle metrics, and
 * {@link DeviceMemoryManager} constraint simulator for GPU memory modeling.
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=VlmPageReuseMemoryLeakTest \
 *     2>&1 | tee /tmp/vlm-page-reuse-leak.log
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class VlmPageReuseMemoryLeakTest {

    private DeviceMemoryManager memMgr;
    private boolean cacheWasEnabled;
    private double originalGrowthFactor;

    @BeforeEach
    void setUp() {
        memMgr = DeviceMemoryManager.getInstance();
        cacheWasEnabled = ArrayCacheMemoryMgr.isCacheEnabled();
        originalGrowthFactor = ArrayCacheMemoryMgr.getGrowthFactor().get();
        ArrayCacheMemoryMgr.setEnableCache(false);
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);
        ArrayCacheMemoryMgr.clearCacheState();
        // Stabilize: let GC settle before measuring
        System.gc();
        try { Thread.sleep(50); } catch (InterruptedException ignored) {}
    }

    @AfterEach
    void tearDown() {
        memMgr.clearStubTopology();
        ArrayCacheMemoryMgr.setEnableCache(cacheWasEnabled);
        ArrayCacheMemoryMgr.setGrowthFactor(originalGrowthFactor);
        ArrayCacheMemoryMgr.clearCacheState();
    }

    // =========================================================================
    // Test 1: Measure native memory leaked when cache stays enabled after DSP null
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("Leak 1 impact: cache enable persistence leaks real native memory")
    void testCacheEnableLeaksRealMemory() {
        // --- UNFIXED path: cache stays enabled ---
        long physBefore = Pointer.physicalBytes();
        long totalBytesBefore = Pointer.totalBytes();
        long liveCountBefore = Nd4j.framework.lifecycle().liveCount();

        ArrayCacheMemoryMgr.setEnableCache(true);
        // Simulate DSP null return WITHOUT the fix (no setEnableCache(false))

        ArrayCacheMemoryMgr mgr = new ArrayCacheMemoryMgr();
        int numArrays = 50;
        long expectedBytes = numArrays * 256L * 256 * DataType.FLOAT.width(); // ~12.5 MB

        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < numArrays; i++) {
            arrays.add(mgr.allocate(false, DataType.FLOAT, new long[]{256, 256}));
        }
        // Release: with cache ON, arrays go INTO cache (not freed)
        for (INDArray arr : arrays) {
            mgr.release(arr);
        }
        arrays.clear();

        long physAfterCached = Pointer.physicalBytes();
        long totalBytesAfterCached = Pointer.totalBytes();
        long cacheSize = ArrayCacheMemoryMgr.getCurrentCacheSize().get();
        long liveCountCached = Nd4j.framework.lifecycle().liveCount();

        long physLeaked = physAfterCached - physBefore;
        long totalBytesLeaked = totalBytesAfterCached - totalBytesBefore;
        long liveLeaked = liveCountCached - liveCountBefore;

        log.info("LEAK 1 — UNFIXED (cache ON after DSP null):");
        log.info("  Pointer.physicalBytes delta: +{} bytes (+{} MB)", physLeaked, physLeaked / (1024 * 1024));
        log.info("  Pointer.totalBytes delta:    +{} bytes (+{} MB)", totalBytesLeaked, totalBytesLeaked / (1024 * 1024));
        log.info("  ArrayCache size:             {} bytes ({} MB)", cacheSize, cacheSize / (1024 * 1024));
        log.info("  Live array count delta:      +{}", liveLeaked);
        log.info("  Expected leaked:             {} bytes ({} MB)", expectedBytes, expectedBytes / (1024 * 1024));

        // Cache holds the memory — this IS the leak
        assertTrue(cacheSize >= expectedBytes * 0.9,
                "Cache should hold ~" + expectedBytes / (1024*1024) + "MB, actual: " + cacheSize / (1024*1024) + "MB");

        // --- FIXED path: cache disabled after DSP null ---
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(false); // THE FIX
        System.gc();
        try { Thread.sleep(50); } catch (InterruptedException ignored) {}

        long physFixed = Pointer.physicalBytes();
        long totalBytesFixed = Pointer.totalBytes();
        long liveCountFixed = Nd4j.framework.lifecycle().liveCount();

        ArrayCacheMemoryMgr mgr2 = new ArrayCacheMemoryMgr();
        List<INDArray> arrays2 = new ArrayList<>();
        for (int i = 0; i < numArrays; i++) {
            arrays2.add(mgr2.allocate(false, DataType.FLOAT, new long[]{256, 256}));
        }
        // Release: with cache OFF, arrays freed immediately
        for (INDArray arr : arrays2) {
            mgr2.release(arr);
        }
        arrays2.clear();
        System.gc();
        try { Thread.sleep(50); } catch (InterruptedException ignored) {}

        long physAfterFixed = Pointer.physicalBytes();
        long totalBytesAfterFixed = Pointer.totalBytes();
        long cacheSizeFixed = ArrayCacheMemoryMgr.getCurrentCacheSize().get();
        long liveCountAfterFixed = Nd4j.framework.lifecycle().liveCount();

        long physDeltaFixed = physAfterFixed - physFixed;
        long totalBytesDeltaFixed = totalBytesAfterFixed - totalBytesFixed;

        log.info("LEAK 1 — FIXED (cache OFF after DSP null):");
        log.info("  Pointer.physicalBytes delta: +{} bytes (+{} KB)", physDeltaFixed, physDeltaFixed / 1024);
        log.info("  Pointer.totalBytes delta:    +{} bytes (+{} KB)", totalBytesDeltaFixed, totalBytesDeltaFixed / 1024);
        log.info("  ArrayCache size:             {} bytes", cacheSizeFixed);
        log.info("  Live array count delta:      +{}", liveCountAfterFixed - liveCountFixed);

        assertEquals(0, cacheSizeFixed, "With fix, cache should be empty");

        // The fix should save ~expectedBytes of native memory
        long memorySaved = cacheSize; // = what was leaked before
        log.info("IMPACT: Fix saves {} MB of native memory per execution cycle",
                memorySaved / (1024 * 1024));
    }

    // =========================================================================
    // Test 2: Measure monotonic cache growth across pages (before/after fix)
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("Leak 2 impact: cache grows across pages — real native memory growth measured")
    void testCacheGrowthAcrossPagesRealMemory() {
        ArrayCacheMemoryMgr mgr = new ArrayCacheMemoryMgr();
        int numPages = 5;
        int arraysPerPage = 20;
        long arrayBytes = 512L * 512 * DataType.FLOAT.width(); // ~1 MB each

        // --- UNFIXED: no reset between pages ---
        ArrayCacheMemoryMgr.setEnableCache(true);
        System.gc();
        try { Thread.sleep(50); } catch (InterruptedException ignored) {}

        long physBaseline = Pointer.physicalBytes();
        long totalBytesBaseline = Pointer.totalBytes();
        long liveBaseline = Nd4j.framework.lifecycle().liveCount();

        List<Long> physPerPage = new ArrayList<>();
        List<Long> totalBytesPerPage = new ArrayList<>();
        List<Long> cacheSizePerPage = new ArrayList<>();
        List<Long> livePerPage = new ArrayList<>();

        for (int page = 0; page < numPages; page++) {
            int dim = 512 + page; // different shape per page → cache miss
            List<INDArray> pageArrays = new ArrayList<>();
            for (int i = 0; i < arraysPerPage; i++) {
                pageArrays.add(mgr.allocate(false, DataType.FLOAT, new long[]{dim, 512}));
            }
            for (INDArray arr : pageArrays) {
                mgr.release(arr);
            }
            pageArrays.clear();

            // NO resetForNextPage — the bug

            physPerPage.add(Pointer.physicalBytes());
            totalBytesPerPage.add(Pointer.totalBytes());
            cacheSizePerPage.add(ArrayCacheMemoryMgr.getCurrentCacheSize().get());
            livePerPage.add(Nd4j.framework.lifecycle().liveCount());
        }

        long physGrowthUnfixed = physPerPage.get(numPages - 1) - physBaseline;
        long totalBytesGrowthUnfixed = totalBytesPerPage.get(numPages - 1) - totalBytesBaseline;
        long cacheGrowthUnfixed = cacheSizePerPage.get(numPages - 1);
        long liveGrowthUnfixed = livePerPage.get(numPages - 1) - liveBaseline;

        log.info("LEAK 2 — UNFIXED (no reset between pages):");
        for (int p = 0; p < numPages; p++) {
            log.info("  Page {}: physBytes=+{}MB, totalBytes=+{}MB, cache={}MB, live=+{}",
                    p,
                    (physPerPage.get(p) - physBaseline) / (1024 * 1024),
                    (totalBytesPerPage.get(p) - totalBytesBaseline) / (1024 * 1024),
                    cacheSizePerPage.get(p) / (1024 * 1024),
                    livePerPage.get(p) - liveBaseline);
        }
        log.info("  TOTAL native growth: {} MB (physicalBytes), {} MB (totalBytes)",
                physGrowthUnfixed / (1024 * 1024), totalBytesGrowthUnfixed / (1024 * 1024));
        log.info("  TOTAL cache size: {} MB", cacheGrowthUnfixed / (1024 * 1024));
        log.info("  TOTAL live array growth: +{}", liveGrowthUnfixed);

        // Cache should grow monotonically
        for (int i = 1; i < numPages; i++) {
            assertTrue(cacheSizePerPage.get(i) > cacheSizePerPage.get(i - 1),
                    "Cache grows across pages: page " + (i-1) + "=" +
                    cacheSizePerPage.get(i-1) / (1024*1024) + "MB → page " + i + "=" +
                    cacheSizePerPage.get(i) / (1024*1024) + "MB");
        }

        // --- FIXED: reset between pages ---
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(false);
        System.gc();
        try { Thread.sleep(50); } catch (InterruptedException ignored) {}

        long physBaselineFixed = Pointer.physicalBytes();
        long totalBytesBaselineFixed = Pointer.totalBytes();
        long liveBaselineFixed = Nd4j.framework.lifecycle().liveCount();

        List<Long> physPerPageFixed = new ArrayList<>();
        List<Long> cacheSizePerPageFixed = new ArrayList<>();

        for (int page = 0; page < numPages; page++) {
            ArrayCacheMemoryMgr.setEnableCache(true); // DSP enables
            int dim = 512 + page;
            List<INDArray> pageArrays = new ArrayList<>();
            for (int i = 0; i < arraysPerPage; i++) {
                pageArrays.add(mgr.allocate(false, DataType.FLOAT, new long[]{dim, 512}));
            }
            for (INDArray arr : pageArrays) {
                mgr.release(arr);
            }
            pageArrays.clear();

            // FIXED resetForNextPage: drain cache + deferred
            ArrayCacheMemoryMgr.closeDeferredBuffers(null);
            ArrayCacheMemoryMgr.clearCacheState();
            ArrayCacheMemoryMgr.setEnableCache(false);

            System.gc();

            physPerPageFixed.add(Pointer.physicalBytes());
            cacheSizePerPageFixed.add(ArrayCacheMemoryMgr.getCurrentCacheSize().get());
        }

        long physGrowthFixed = physPerPageFixed.get(numPages - 1) - physBaselineFixed;
        long liveGrowthFixed = Nd4j.framework.lifecycle().liveCount() - liveBaselineFixed;

        log.info("LEAK 2 — FIXED (reset between pages):");
        for (int p = 0; p < numPages; p++) {
            log.info("  Page {}: physBytes=+{}MB, cache={}MB",
                    p,
                    (physPerPageFixed.get(p) - physBaselineFixed) / (1024 * 1024),
                    cacheSizePerPageFixed.get(p) / (1024 * 1024));
        }
        log.info("  TOTAL native growth: {} MB", physGrowthFixed / (1024 * 1024));

        // Cache should be 0 after every page
        for (int p = 0; p < numPages; p++) {
            assertEquals(0, cacheSizePerPageFixed.get(p), "Cache 0 after page " + p);
        }

        log.info("IMPACT: Fix prevents {} MB of native memory accumulation across {} pages",
                cacheGrowthUnfixed / (1024 * 1024), numPages);
    }

    // =========================================================================
    // Test 3: Measure deferred buffer accumulation (real bytes)
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("Leak 3 impact: deferred close buffers leak real native memory")
    void testDeferredBuffersLeakRealMemory() {
        ArrayCacheMemoryMgr.setEnableCache(true);

        int numPages = 5;
        int deferredPerPage = 20;
        long bufferElements = 4096; // 4096 floats = 16 KB each
        long expectedBytesPerBuffer = bufferElements * DataType.FLOAT.width();
        long expectedTotalLeak = numPages * deferredPerPage * expectedBytesPerBuffer;

        // --- UNFIXED: deferred buffers accumulate ---
        System.gc();
        try { Thread.sleep(50); } catch (InterruptedException ignored) {}

        long physBaseline = Pointer.physicalBytes();
        long totalBytesBaseline = Pointer.totalBytes();
        long liveBaseline = Nd4j.framework.lifecycle().liveCount();

        List<Long> physPerPage = new ArrayList<>();
        List<Long> totalBytesPerPage = new ArrayList<>();
        List<Integer> deferredPerPageList = new ArrayList<>();

        for (int page = 0; page < numPages; page++) {
            for (int i = 0; i < deferredPerPage; i++) {
                DataBuffer buf = Nd4j.createBuffer(DataType.FLOAT, bufferElements, false);
                ArrayCacheMemoryMgr.getDeferredCloseBuffers().put(buf, Boolean.TRUE);
            }
            // NO closeDeferredBuffers — the bug

            physPerPage.add(Pointer.physicalBytes());
            totalBytesPerPage.add(Pointer.totalBytes());
            deferredPerPageList.add(ArrayCacheMemoryMgr.getDeferredCloseBuffers().size());
        }

        long physGrowthUnfixed = physPerPage.get(numPages - 1) - physBaseline;
        long totalBytesGrowthUnfixed = totalBytesPerPage.get(numPages - 1) - totalBytesBaseline;
        int totalDeferred = deferredPerPageList.get(numPages - 1);
        long liveGrowthUnfixed = Nd4j.framework.lifecycle().liveCount() - liveBaseline;
        long retainedDeferredBytes = ArrayCacheMemoryMgr.getDeferredCloseBuffers().keySet().stream()
                .mapToLong(buffer -> buffer.length() * buffer.dataType().width())
                .sum();

        log.info("LEAK 3 — UNFIXED (deferred buffers accumulate):");
        for (int p = 0; p < numPages; p++) {
            log.info("  Page {}: physBytes=+{}KB, totalBytes=+{}KB, deferred={}",
                    p,
                    (physPerPage.get(p) - physBaseline) / 1024,
                    (totalBytesPerPage.get(p) - totalBytesBaseline) / 1024,
                    deferredPerPageList.get(p));
        }
        log.info("  TOTAL: physBytes=+{}KB, totalBytes=+{}KB, deferred={}, live=+{}",
                physGrowthUnfixed / 1024, totalBytesGrowthUnfixed / 1024, totalDeferred, liveGrowthUnfixed);
        log.info("  Expected leak: {} KB ({} buffers × {} KB each)",
                expectedTotalLeak / 1024, numPages * deferredPerPage, expectedBytesPerBuffer / 1024);

        // Deferred buffers and their native byte footprint should accumulate. Pointer.physicalBytes()
        // is only diagnostic here: a warmed native allocator may satisfy these buffers from its pool
        // without increasing process-reserved bytes.
        assertEquals(numPages * deferredPerPage, totalDeferred);
        assertEquals(expectedTotalLeak, retainedDeferredBytes,
                "Deferred buffers should retain the expected native byte footprint");
        assertTrue(liveGrowthUnfixed >= numPages * deferredPerPage,
                "Deferred buffers should remain live until the page-boundary drain");

        // --- FIXED: drain deferred at each page boundary ---
        ArrayCacheMemoryMgr.clearCacheState();
        System.gc();
        try { Thread.sleep(50); } catch (InterruptedException ignored) {}

        long physBaselineFixed = Pointer.physicalBytes();
        long totalBytesBaselineFixed = Pointer.totalBytes();

        List<Long> physPerPageFixed = new ArrayList<>();
        List<Long> totalBytesPerPageFixed = new ArrayList<>();
        long totalFreedBytes = 0;
        int totalFreedBuffers = 0;

        for (int page = 0; page < numPages; page++) {
            for (int i = 0; i < deferredPerPage; i++) {
                DataBuffer buf = Nd4j.createBuffer(DataType.FLOAT, bufferElements, false);
                ArrayCacheMemoryMgr.getDeferredCloseBuffers().put(buf, Boolean.TRUE);
            }

            // FIXED: drain at page boundary
            long[] stats = ArrayCacheMemoryMgr.closeDeferredBuffers(null);
            totalFreedBuffers += (int) stats[0];
            totalFreedBytes += stats[1];
            ArrayCacheMemoryMgr.clearCacheState();

            System.gc();
            physPerPageFixed.add(Pointer.physicalBytes());
            totalBytesPerPageFixed.add(Pointer.totalBytes());
        }

        long physGrowthFixed = physPerPageFixed.get(numPages - 1) - physBaselineFixed;
        long totalBytesGrowthFixed = totalBytesPerPageFixed.get(numPages - 1) - totalBytesBaselineFixed;

        log.info("LEAK 3 — FIXED (drain deferred at page boundary):");
        for (int p = 0; p < numPages; p++) {
            log.info("  Page {}: physBytes=+{}KB, totalBytes=+{}KB",
                    p,
                    (physPerPageFixed.get(p) - physBaselineFixed) / 1024,
                    (totalBytesPerPageFixed.get(p) - totalBytesBaselineFixed) / 1024);
        }
        log.info("  Total freed: {} buffers, {} KB", totalFreedBuffers, totalFreedBytes / 1024);
        log.info("  Residual native growth: physBytes=+{}KB, totalBytes=+{}KB",
                physGrowthFixed / 1024, totalBytesGrowthFixed / 1024);

        assertEquals(numPages * deferredPerPage, totalFreedBuffers,
                "All deferred buffers should be freed");
        assertEquals(expectedTotalLeak, totalFreedBytes,
                "The page-boundary drains should reclaim the full deferred byte footprint");

        log.info("IMPACT: Fix reclaims ~{} KB of native memory per page ({} buffers × {} KB)",
                deferredPerPage * expectedBytesPerBuffer / 1024,
                deferredPerPage, expectedBytesPerBuffer / 1024);
    }

    // =========================================================================
    // Test 4: Full VLM page lifecycle — compare total native footprint
    // =========================================================================

    @Test
    @Order(4)
    @DisplayName("Full lifecycle: total native memory footprint with all leaks vs all fixes")
    void testFullLifecycleNativeMemoryComparison() {
        ArrayCacheMemoryMgr mgr = new ArrayCacheMemoryMgr();
        int numPages = 5;
        int opsPerPage = 30;
        int viewsPerPage = 10;
        long opArrayBytes = 128L * 128 * DataType.FLOAT.width(); // 64 KB each
        long viewBufferBytes = 64L * 64 * DataType.FLOAT.width(); // 16 KB each

        // ===================== UNFIXED PATH =====================
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(false);
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException ignored) {}

        long physBaselineUnfixed = Pointer.physicalBytes();
        long totalBytesBaselineUnfixed = Pointer.totalBytes();
        long liveBaselineUnfixed = Nd4j.framework.lifecycle().liveCount();

        List<Long> physUnfixed = new ArrayList<>();
        List<Long> totalBytesUnfixed = new ArrayList<>();
        List<Long> cacheUnfixed = new ArrayList<>();
        List<Integer> deferredUnfixed = new ArrayList<>();
        List<Long> liveUnfixed = new ArrayList<>();

        for (int page = 0; page < numPages; page++) {
            // BUG: setEnableCache(true) not undone on DSP null
            ArrayCacheMemoryMgr.setEnableCache(true);

            // Simulate ops
            List<INDArray> pageArrays = new ArrayList<>();
            for (int op = 0; op < opsPerPage; op++) {
                pageArrays.add(mgr.allocate(false, DataType.FLOAT, new long[]{128, 128}));
            }
            for (INDArray arr : pageArrays) {
                mgr.release(arr);
            }
            pageArrays.clear();

            // Simulate views → deferred
            for (int v = 0; v < viewsPerPage; v++) {
                DataBuffer buf = Nd4j.createBuffer(DataType.FLOAT, 64 * 64, false);
                ArrayCacheMemoryMgr.getDeferredCloseBuffers().put(buf, Boolean.TRUE);
            }

            // BUG: resetForNextPage doesn't drain cache/deferred
            // (we just don't call the cleanup)

            physUnfixed.add(Pointer.physicalBytes());
            totalBytesUnfixed.add(Pointer.totalBytes());
            cacheUnfixed.add(ArrayCacheMemoryMgr.getCurrentCacheSize().get());
            deferredUnfixed.add(ArrayCacheMemoryMgr.getDeferredCloseBuffers().size());
            liveUnfixed.add(Nd4j.framework.lifecycle().liveCount());
        }

        long physGrowthUnfixed = physUnfixed.get(numPages - 1) - physBaselineUnfixed;
        long totalBytesGrowthUnfixed = totalBytesUnfixed.get(numPages - 1) - totalBytesBaselineUnfixed;
        long cacheUnfixedFinal = cacheUnfixed.get(numPages - 1);
        int deferredUnfixedFinal = deferredUnfixed.get(numPages - 1);
        long liveGrowthUnfixed = liveUnfixed.get(numPages - 1) - liveBaselineUnfixed;

        // ===================== FIXED PATH =====================
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(false);
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException ignored) {}

        long physBaselineFixed = Pointer.physicalBytes();
        long totalBytesBaselineFixed = Pointer.totalBytes();
        long liveBaselineFixed = Nd4j.framework.lifecycle().liveCount();

        List<Long> physFixed = new ArrayList<>();
        List<Long> totalBytesFixed = new ArrayList<>();
        List<Long> cacheFixed = new ArrayList<>();
        List<Integer> deferredFixed = new ArrayList<>();
        List<Long> liveFixed = new ArrayList<>();

        for (int page = 0; page < numPages; page++) {
            // FIX 1: DSP enables cache
            ArrayCacheMemoryMgr.setEnableCache(true);

            // Simulate ops
            List<INDArray> pageArrays = new ArrayList<>();
            for (int op = 0; op < opsPerPage; op++) {
                pageArrays.add(mgr.allocate(false, DataType.FLOAT, new long[]{128, 128}));
            }
            for (INDArray arr : pageArrays) {
                mgr.release(arr);
            }
            pageArrays.clear();

            // Simulate views → deferred
            for (int v = 0; v < viewsPerPage; v++) {
                DataBuffer buf = Nd4j.createBuffer(DataType.FLOAT, 64 * 64, false);
                ArrayCacheMemoryMgr.getDeferredCloseBuffers().put(buf, Boolean.TRUE);
            }

            // FIX 2+3: resetForNextPage drains everything
            ArrayCacheMemoryMgr.closeDeferredBuffers(null);
            ArrayCacheMemoryMgr.clearCacheState();
            ArrayCacheMemoryMgr.setEnableCache(false);

            System.gc();
            physFixed.add(Pointer.physicalBytes());
            totalBytesFixed.add(Pointer.totalBytes());
            cacheFixed.add(ArrayCacheMemoryMgr.getCurrentCacheSize().get());
            deferredFixed.add(ArrayCacheMemoryMgr.getDeferredCloseBuffers().size());
            liveFixed.add(Nd4j.framework.lifecycle().liveCount());
        }

        long physGrowthFixed = physFixed.get(numPages - 1) - physBaselineFixed;
        long totalBytesGrowthFixed = totalBytesFixed.get(numPages - 1) - totalBytesBaselineFixed;
        long liveGrowthFixed = liveFixed.get(numPages - 1) - liveBaselineFixed;

        // ===================== COMPARISON =====================
        log.info("========== NATIVE MEMORY IMPACT COMPARISON ==========");
        log.info("  {} pages × {} ops × {} KB + {} views × {} KB per page",
                numPages, opsPerPage, opArrayBytes / 1024, viewsPerPage, viewBufferBytes / 1024);
        log.info("");
        log.info("  UNFIXED (per page):");
        for (int p = 0; p < numPages; p++) {
            log.info("    Page {}: physBytes=+{}MB, totalBytes=+{}MB, cache={}KB, deferred={}, live=+{}",
                    p,
                    (physUnfixed.get(p) - physBaselineUnfixed) / (1024 * 1024),
                    (totalBytesUnfixed.get(p) - totalBytesBaselineUnfixed) / (1024 * 1024),
                    cacheUnfixed.get(p) / 1024,
                    deferredUnfixed.get(p),
                    liveUnfixed.get(p) - liveBaselineUnfixed);
        }
        log.info("  UNFIXED TOTAL: physBytes=+{}MB, totalBytes=+{}MB, cache={}KB, deferred={}, live=+{}",
                physGrowthUnfixed / (1024 * 1024),
                totalBytesGrowthUnfixed / (1024 * 1024),
                cacheUnfixedFinal / 1024,
                deferredUnfixedFinal,
                liveGrowthUnfixed);
        log.info("");
        log.info("  FIXED (per page):");
        for (int p = 0; p < numPages; p++) {
            log.info("    Page {}: physBytes=+{}MB, totalBytes=+{}MB, cache={}KB, deferred={}, live=+{}",
                    p,
                    (physFixed.get(p) - physBaselineFixed) / (1024 * 1024),
                    (totalBytesFixed.get(p) - totalBytesBaselineFixed) / (1024 * 1024),
                    cacheFixed.get(p) / 1024,
                    deferredFixed.get(p),
                    liveFixed.get(p) - liveBaselineFixed);
        }
        log.info("  FIXED TOTAL: physBytes=+{}MB, totalBytes=+{}MB, live=+{}",
                physGrowthFixed / (1024 * 1024),
                totalBytesGrowthFixed / (1024 * 1024),
                liveGrowthFixed);
        log.info("");

        long memorySaved = totalBytesGrowthUnfixed - totalBytesGrowthFixed;
        log.info("  MEMORY SAVED BY FIX: {} MB (Pointer.totalBytes)",
                memorySaved / (1024 * 1024));
        log.info("  Cache eliminated: {} KB", cacheUnfixedFinal / 1024);
        log.info("  Deferred eliminated: {} buffers", deferredUnfixedFinal);
        log.info("  Live array leak eliminated: {} arrays", liveGrowthUnfixed - liveGrowthFixed);
        log.info("=====================================================");

        // Verify fixes
        for (int p = 0; p < numPages; p++) {
            assertEquals(0, cacheFixed.get(p), "Cache 0 after page " + p);
            assertEquals(0, deferredFixed.get(p), "Deferred 0 after page " + p);
        }

        // Unfixed should have accumulated significantly
        assertTrue(cacheUnfixedFinal > 0, "Unfixed path has cached arrays");
        assertTrue(deferredUnfixedFinal > 0, "Unfixed path has deferred buffers");
        assertTrue(totalBytesGrowthUnfixed > totalBytesGrowthFixed,
                "Unfixed native growth (" + totalBytesGrowthUnfixed / (1024*1024) + "MB) should exceed fixed (" +
                totalBytesGrowthFixed / (1024*1024) + "MB)");
    }

    // =========================================================================
    // Test 5: Constraint sim — progressive exhaustion vs stable (with GPU model)
    // =========================================================================

    @Test
    @Order(5)
    @DisplayName("Constraint sim: progressive GPU exhaustion unfixed vs stable fixed")
    void testConstraintSimCompareUnfixedVsFixed() {
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .deviceName("Stub GPU 24GB")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(6L * 1024 * 1024 * 1024)
                .isDefault(true).addPeerDevice(1)
                .build();
        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .deviceName("Stub GPU overflow")
                .totalMemory(24L * 1024 * 1024 * 1024)
                .availableMemory(20L * 1024 * 1024 * 1024)
                .addPeerDevice(0)
                .build();
        memMgr.configureStubTopology(Arrays.asList(gpu0, gpu1));
        memMgr.setMemoryPressureThreshold(0.9);

        TransferSubsystem transferSub = new TransferSubsystem();
        transferSub.setEnabled(true);

        long perPageUsage = 500L * 1024 * 1024;   // 500 MB per page
        long unfixedLeakFraction = 500L * 1024 * 1024; // 500 MB leaked (0% freed — worst case)
        int numPages = 10;

        // --- UNFIXED: 300MB leaked per page, accumulating ---
        DspReplayTransferAnalytics analyticsUnfixed = new DspReplayTransferAnalytics(transferSub, memMgr);
        gpu0.setAvailableMemory(6L * 1024 * 1024 * 1024);
        memMgr.setSimulatedFreeMemory(0, gpu0.getAvailableMemory());
        List<Long> freeUnfixed = new ArrayList<>();
        int reroutesUnfixed = 0;

        for (int page = 0; page < numPages; page++) {
            freeUnfixed.add(gpu0.getAvailableMemory());

            analyticsUnfixed.beginStep(page, page * 1000L);

            // Simulate page execution: consume memory
            gpu0.consumeMemory(perPageUsage);
            // BUG: only 40% freed → 300MB leaked per page
            gpu0.setAvailableMemory(gpu0.getAvailableMemory() + perPageUsage - unfixedLeakFraction);
            memMgr.setSimulatedFreeMemory(0, gpu0.getAvailableMemory());

            // Check pressure for NEXT page's capture buffer (1 GB needed)
            DeviceDescriptor target = analyticsUnfixed.checkMemoryPressureForTransfer(
                    gpu0, 1L * 1024 * 1024 * 1024, TransferReason.CAPTURE_BUFFER_COPY);
            if (!target.getDeviceId().equals(gpu0.getDeviceId())) reroutesUnfixed++;

            analyticsUnfixed.recordTransfer(TransferEvent.builder()
                    .variableName("page_" + page).direction(TransferDirection.D2D)
                    .reason(TransferReason.CAPTURE_BUFFER_COPY)
                    .bytes(perPageUsage).durationNanos(1000)
                    .sourceDeviceId(0).destDeviceId(target.getDeviceIndex())
                    .build());
            analyticsUnfixed.endStep();
        }

        // --- FIXED: 100% recovery ---
        DspReplayTransferAnalytics analyticsFixed = new DspReplayTransferAnalytics(transferSub, memMgr);
        gpu0.setAvailableMemory(6L * 1024 * 1024 * 1024);
        List<Long> freeFixed = new ArrayList<>();
        int reroutesFixed = 0;

        for (int page = 0; page < numPages; page++) {
            freeFixed.add(gpu0.getAvailableMemory());
            memMgr.setSimulatedFreeMemory(0, gpu0.getAvailableMemory());

            analyticsFixed.beginStep(page, page * 1000L);
            gpu0.consumeMemory(perPageUsage);
            // FIX: 100% freed
            gpu0.setAvailableMemory(gpu0.getAvailableMemory() + perPageUsage);
            memMgr.setSimulatedFreeMemory(0, gpu0.getAvailableMemory());

            DeviceDescriptor target = analyticsFixed.checkMemoryPressureForTransfer(
                    gpu0, 1L * 1024 * 1024 * 1024, TransferReason.CAPTURE_BUFFER_COPY);
            if (!target.getDeviceId().equals(gpu0.getDeviceId())) reroutesFixed++;

            analyticsFixed.recordTransfer(TransferEvent.builder()
                    .variableName("page_" + page).direction(TransferDirection.D2D)
                    .reason(TransferReason.CAPTURE_BUFFER_COPY)
                    .bytes(perPageUsage).durationNanos(1000)
                    .sourceDeviceId(0).destDeviceId(target.getDeviceIndex())
                    .build());
            analyticsFixed.endStep();
        }

        log.info("========== GPU MEMORY CONSTRAINT COMPARISON ==========");
        log.info("  {} pages, {} MB per page, 24GB GPU with 6GB free", numPages, perPageUsage / (1024*1024));
        log.info("");
        log.info("  UNFIXED (500MB leaked/page):");
        for (int p = 0; p < numPages; p++) {
            log.info("    Page {}: free={}MB", p, freeUnfixed.get(p) / (1024*1024));
        }
        long totalGpuLeak = freeUnfixed.get(0) - freeUnfixed.get(numPages - 1);
        int pagesUntilOom = (int) (freeUnfixed.get(0) / unfixedLeakFraction);
        log.info("  Total GPU leak: {}MB, reroutes: {}, est pages to OOM: {}",
                totalGpuLeak / (1024*1024), reroutesUnfixed, pagesUntilOom);
        log.info("");
        log.info("  FIXED (100% recovery):");
        for (int p = 0; p < numPages; p++) {
            log.info("    Page {}: free={}MB", p, freeFixed.get(p) / (1024*1024));
        }
        log.info("  Total GPU leak: 0MB, reroutes: {}", reroutesFixed);
        log.info("======================================================");

        // Unfixed: GPU free decreases significantly
        long unfixedFreeLoss = freeUnfixed.get(0) - freeUnfixed.get(numPages - 1);
        assertTrue(freeUnfixed.get(numPages - 1) < freeUnfixed.get(0),
                "Unfixed: GPU free should decrease");
        assertTrue(unfixedFreeLoss > 2L * 1024 * 1024 * 1024,
                "Unfixed: should lose >2GB of GPU memory across " + numPages + " pages, lost: " +
                unfixedFreeLoss / (1024*1024) + "MB");

        // Fixed: GPU free stable
        assertEquals(freeFixed.get(0), freeFixed.get(numPages - 1),
                "Fixed: GPU free should be stable");
        assertEquals(0, reroutesFixed, "Fixed: no rerouting needed");

        // Impact: how many more pages can we run with the fix?
        log.info("IMPACT: Fix enables unlimited pages. Without fix, OOM at ~{} pages ({} MB leaked/page from {} MB free)",
                pagesUntilOom, unfixedLeakFraction / (1024*1024), freeUnfixed.get(0) / (1024*1024));
    }

    // =========================================================================
    // Test 6: Growth factor scoping (baseline — not a leak)
    // =========================================================================

    @Test
    @Order(6)
    @DisplayName("Baseline: growth factor scoping is correct — no memory impact")
    void testGrowthFactorScopingWorks() {
        assertEquals(1.0, ArrayCacheMemoryMgr.effectiveGrowthFactor(), 0.001);

        long physBefore = Pointer.physicalBytes();
        long totalBytesBefore = Pointer.totalBytes();
        long liveBefore = Nd4j.framework.lifecycle().liveCount();

        try (AutoCloseable scope = ArrayCacheMemoryMgr.withGrowthFactor(1.1)) {
            assertEquals(1.1, ArrayCacheMemoryMgr.effectiveGrowthFactor(), 0.001);
        } catch (Exception e) {
            fail(e.getMessage());
        }

        assertEquals(1.0, ArrayCacheMemoryMgr.effectiveGrowthFactor(), 0.001);

        long physAfter = Pointer.physicalBytes();
        long totalBytesAfter = Pointer.totalBytes();
        long liveAfter = Nd4j.framework.lifecycle().liveCount();

        log.info("Growth factor scoping: physBytes delta={}KB, totalBytes delta={}KB, live delta={}",
                (physAfter - physBefore) / 1024,
                (totalBytesAfter - totalBytesBefore) / 1024,
                liveAfter - liveBefore);
    }

    // =========================================================================
    // Test 7: A component reset must not sweep another executor's retained inputs
    // =========================================================================

    @Test
    @Order(7)
    @DisplayName("Multi-model reset preserves frozen decoder inputs until the shared cache sweep")
    void testComponentResetDefersSharedCacheCleanup() {
        DynamicShapePlanExecutor disposableExecutor =
                new DynamicShapePlanExecutor(SameDiff.create(), new ArrayCacheMemoryMgr());
        INDArray retainedDecoderMask = Nd4j.zeros(DataType.LONG, 1, 2305);
        INDArray disposableIntermediate = Nd4j.zeros(DataType.FLOAT, 8, 8);
        DataBuffer retainedBuffer = retainedDecoderMask.data();
        DataBuffer disposableBuffer = disposableIntermediate.data();

        try {
            // Reproduce the page-boundary state: both buffers were released by session memory
            // managers, but the frozen decoder still strongly owns its external input array.
            ArrayCacheMemoryMgr.getDeferredCloseBuffers().put(retainedBuffer, Boolean.TRUE);
            ArrayCacheMemoryMgr.getDeferredCloseBuffers().put(disposableBuffer, Boolean.TRUE);

            // Resetting the vision/embed component must not sweep the thread-wide cache. The VLM
            // page owner will do that once all disposable executors have been reset.
            disposableExecutor.resetForNextPage(false);
            assertFalse(retainedBuffer.wasClosed(),
                    "A component reset must not close another executor's frozen external input");
            assertFalse(disposableBuffer.wasClosed(),
                    "Shared cleanup must remain deferred until the multi-model page boundary");

            IdentityHashMap<DataBuffer, Boolean> survivingBuffers = new IdentityHashMap<>();
            survivingBuffers.put(retainedBuffer, Boolean.TRUE);
            long[] cleanup = ArrayCacheMemoryMgr.closeDeferredBuffers(survivingBuffers);
            ArrayCacheMemoryMgr.clearCacheState();
            ArrayCacheMemoryMgr.setEnableCache(false);

            assertEquals(1, cleanup[0], "Only the unprotected component buffer should be reclaimed");
            assertTrue(disposableBuffer.wasClosed(),
                    "The page-level sweep must reclaim buffers from reset executors");
            assertFalse(retainedBuffer.wasClosed(),
                    "The page-level sweep must preserve the frozen decoder input");

            // This is the exact operation and shape from the multi-page OCR failure.
            assertDoesNotThrow(() -> retainedDecoderMask.assign(0));
            assertEquals(0L, retainedDecoderMask.sumNumber().longValue());
        } finally {
            disposableExecutor.close();
            if (!retainedDecoderMask.wasClosed()) {
                retainedDecoderMask.close();
            }
            if (!disposableIntermediate.wasClosed()) {
                disposableIntermediate.close();
            }
        }
    }

    // =========================================================================
    // Test 8: Frozen-plan reuse must reject arrays whose DataBuffer was closed
    // =========================================================================

    @Test
    @Order(8)
    @DisplayName("Frozen decoder reuse rejects a closed retained attention mask")
    void testClosedFrozenExternalInputIsNotReusable() {
        INDArray retainedDecoderMask = Nd4j.zeros(DataType.LONG, 1, 2305);
        DataBuffer retainedBuffer = retainedDecoderMask.data();

        try {
            INDArray[] frozenInputs = new INDArray[]{retainedDecoderMask};
            assertTrue(DynamicShapePlanExecutor.areExternalInputsReusable(frozenInputs),
                    "A live frozen attention mask should be reusable");

            // Reproduce the exact stale-owner state from the dogfood runtime: the executor still
            // holds the INDArray object, but cleanup has already closed its DataBuffer.
            retainedBuffer.close();

            assertFalse(DynamicShapePlanExecutor.areExternalInputsReusable(frozenInputs),
                    "Frozen-plan reuse must fail closed before assign(0) touches a dead DataBuffer");
            assertFalse(DynamicShapePlanExecutor.areExternalInputsReusable(
                            new INDArray[]{retainedDecoderMask, null}),
                    "A partially populated snapshot cannot preserve pointer stability");
        } finally {
            if (!retainedDecoderMask.wasClosed() && retainedDecoderMask.data() != null
                    && !retainedDecoderMask.data().wasClosed()) {
                retainedDecoderMask.close();
            }
        }
    }

    // =========================================================================
    // Test 9: Inactive frozen plans retain their inputs across active-plan cleanup
    // =========================================================================

    @Test
    @Order(9)
    @DisplayName("Prefill cleanup preserves the inactive decode plan attention mask")
    void testInactivePinnedPlanInputsAreProtectedDuringCacheCleanup() throws Exception {
        ArrayCacheMemoryMgr memoryManager = new ArrayCacheMemoryMgr();
        DynamicShapePlanExecutor executor =
                new DynamicShapePlanExecutor(SameDiff.create(), memoryManager);
        INDArray retainedDecoderMask = Nd4j.zeros(DataType.LONG, 1, 2305);
        INDArray activePrefillInput = Nd4j.zeros(DataType.FLOAT, 1, 1536, 4);
        INDArray disposablePrefillIntermediate = Nd4j.zeros(DataType.FLOAT, 8, 8);
        DataBuffer retainedBuffer = retainedDecoderMask.data();
        DataBuffer disposableBuffer = disposablePrefillIntermediate.data();

        try {
            // Model the decoder's frozen A/B plan switch without requiring a native backend:
            // handle 0xdec owns the decode mask and handle 0xpre owns the active prefill input.
            java.lang.reflect.Method retainForPlan = DynamicShapePlanExecutor.class
                    .getDeclaredMethod("retainExternalInputsForPlan", long.class, INDArray[].class);
            retainForPlan.setAccessible(true);
            retainForPlan.invoke(executor, 0xdecL, (Object) new INDArray[]{retainedDecoderMask});
            retainForPlan.invoke(executor, 0xfeeL, (Object) new INDArray[]{activePrefillInput});

            assertTrue(executor.isRetainedExternalInput(retainedDecoderMask),
                    "Pending cast cleanup must recognize the inactive decode-plan input by identity");
            assertTrue(executor.isRetainedExternalInput(activePrefillInput),
                    "The active prefill-plan input must also remain retained");
            assertFalse(executor.isRetainedExternalInput(disposablePrefillIntermediate),
                    "Unowned intermediates must remain eligible for cleanup");

            ArrayCacheMemoryMgr.setEnableCache(true);
            memoryManager.release(retainedDecoderMask);
            memoryManager.release(disposablePrefillIntermediate);

            IdentityHashMap<DataBuffer, Boolean> protectedBuffers = new IdentityHashMap<>();
            executor.collectRetainedExternalInputBuffers(protectedBuffers);
            memoryManager.close(protectedBuffers);

            assertTrue(protectedBuffers.containsKey(retainedBuffer),
                    "Inactive decode-plan inputs must be part of the session cleanup protection set");
            assertFalse(retainedBuffer.wasClosed(),
                    "Active prefill cleanup must not close the inactive decode attention mask");
            assertTrue(disposableBuffer.wasClosed(),
                    "Unowned prefill intermediates should still be reclaimed");

            // Exact operation and shape from the post-rebuild dogfood failure.
            assertDoesNotThrow(() -> retainedDecoderMask.assign(0));
        } finally {
            executor.close();
            if (!retainedDecoderMask.wasClosed()) retainedDecoderMask.close();
            if (!activePrefillInput.wasClosed()) activePrefillInput.close();
            if (!disposablePrefillIntermediate.wasClosed()) disposablePrefillIntermediate.close();
        }
    }
}
