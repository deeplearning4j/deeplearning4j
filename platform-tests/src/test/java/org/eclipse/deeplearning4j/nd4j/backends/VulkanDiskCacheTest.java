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
package org.eclipse.deeplearning4j.nd4j.backends;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * ADR 0115 — Vulkan SPIR-V + pipeline disk cache tests.
 *
 * Tier 1: MLIR→SPIR-V lowering results persisted as spv_<16hex>.spv + .meta
 * (VulkanSpirvDiskCache, mirrors the Triton kernel disk cache).
 * Tier 2: the driver VkPipelineCache blob persisted per physical device as
 * vkpc_<16hex>.bin (VulkanDeviceContext::savePipelineCacheBlob).
 *
 * Cache behavior is observed through the VulkanConfig counters surfaced on
 * {@link Environment} (native sd_printf output cannot be intercepted from
 * Java — the TritonConfig residency-warn-counter pattern).
 *
 * <h3>Cross-JVM phase harness</h3>
 * The product contract is cross-process reuse, and the in-process eager
 * pipeline cache (device-context lifetime) legitimately shields the disk
 * tiers within one JVM. {@link #testDiskCachePhase()} therefore runs one
 * phase per JVM, sharing a fixed directory wired through the surefire
 * env-var mapping (which this also validates):
 *
 * <pre>
 *   DIR=/tmp/vk-disk-cache-phases; rm -rf $DIR
 *   for PHASE in cold warm corrupt bypass; do
 *     EXTRA=""; [ "$PHASE" = bypass ] && EXTRA="-Dnd4j.environment.vulkanAlwaysCompile=true"
 *     mvn test -Ptest-vulkan -Dtest=VulkanDiskCacheTest#testDiskCachePhase* \
 *       -Dvulkan.cache.test.phase=$PHASE -Dvulkan.cache.test.dir=$DIR \
 *       -Dnd4j.environment.vulkanSpirvCacheDir=$DIR/spirv \
 *       -Dnd4j.environment.vulkanPipelineCacheDir=$DIR/pipe $EXTRA
 *   done
 * </pre>
 *
 * Without {@code -Dvulkan.cache.test.phase} the phase test is skipped, so a
 * plain suite run stays green.
 *
 * <p>{@link #testDspReplayPathPopulatesCache()} exercises the DSP
 * VULKAN_REPLAY path (per-plan pipeline caches → true in-JVM disk hits). It
 * currently self-skips while the Vulkan input-staging frontier
 * ("internal segment input is not device-authoritative") is in flight; it
 * arms itself automatically once that lands.</p>
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("Vulkan SPIR-V + pipeline disk cache (ADR 0115)")
public class VulkanDiskCacheTest {

    private static final String VULKAN_BINDINGS_CLASS = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";
    private static final int SPIRV_MAGIC = 0x07230203;
    private static final String WIP_STAGING_ERROR = "device-authoritative";

    private static Object nativeOps;
    private static boolean vulkanDevicePresent = false;
    private static boolean mlirEnabled = false;

    @BeforeAll
    static void setup() {
        try {
            Class<?> bindingsClass = Class.forName(VULKAN_BINDINGS_CLASS);
            nativeOps = bindingsClass.getDeclaredConstructor().newInstance();
            int count = (int) bindingsClass.getMethod("getAvailableDevices").invoke(nativeOps);
            vulkanDevicePresent = (count > 0);
            log.info("Vulkan device count: {}", count);
            try {
                Object value = bindingsClass.getField("HAVE_MLIR").get(null);
                mlirEnabled = value instanceof Number && ((Number) value).intValue() == 1;
            } catch (NoSuchFieldException noGeneratedConstant) {
                mlirEnabled = false;
            }
            log.info("Vulkan MLIR enabled: {}", mlirEnabled);
        } catch (ClassNotFoundException e) {
            log.warn("Vulkan NativeOps not on the test classpath — run with -Ptest-vulkan");
            nativeOps = null;
        } catch (Exception e) {
            throw new IllegalStateException("Vulkan NativeOps loaded but initialization failed", e);
        }
    }

    private static void requireVulkanMlir() {
        assumeTrue(nativeOps != null,
                "Vulkan NativeOps (" + VULKAN_BINDINGS_CLASS + ") not on classpath — run with -Ptest-vulkan");
        assumeTrue(vulkanDevicePresent,
                "getAvailableDevices()==0 — no Vulkan device present; lavapipe not installed?");
        assumeTrue(mlirEnabled,
                "HAVE_MLIR=0 in this chip build — no SPIR-V JIT, nothing to disk-cache. Skipping.");
        try {
            Object result = nativeOps.getClass().getMethod("setDevice", int.class).invoke(nativeOps, 0);
            assertEquals(1, ((Number) result).intValue(), "setDevice(0) must succeed");
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("setDevice unavailable on Vulkan NativeOps", e);
        }
    }

    // ── workloads ────────────────────────────────────────────────────────────

    /**
     * Eager-path workload: zeros + assign compile real SPIR-V kernels through
     * the device-context VulkanPipelineCache (proven independent of the DSP
     * staging frontier). Deterministic → identical MLIR → identical Tier-1
     * keys across JVMs.
     */
    private static void eagerWorkload() {
        float[] data = new float[16 * 16];
        for (int i = 0; i < data.length; i++) {
            data[i] = ((i % 13) - 6) * 0.11f;
        }
        INDArray x = Nd4j.zeros(DataType.FLOAT, 16, 16);
        x.assign(Nd4j.create(data, new long[]{16, 16}));
        Nd4j.getExecutioner().commit();

        INDArray y = Nd4j.zeros(DataType.FLOAT, 4, 64);
        float[] data2 = new float[4 * 64];
        for (int i = 0; i < data2.length; i++) {
            data2[i] = ((i % 7) - 3) * 0.17f;
        }
        y.assign(Nd4j.create(data2, new long[]{4, 64}));
        Nd4j.getExecutioner().commit();
    }

    /** DSP VULKAN_REPLAY workload — the T3.1 placeholder-only matmul chain. */
    private float[] dspReplayWorkload() {
        SameDiff sd = SameDiff.create();
        try {
            SDVariable a = sd.placeHolder("a", DataType.FLOAT, 4, 8);
            SDVariable b = sd.placeHolder("b", DataType.FLOAT, 8, 4);
            SDVariable c = sd.placeHolder("c", DataType.FLOAT, 4, 2);
            SDVariable first = sd.mmul("first", a, b);
            sd.mmul("out", first, c);

            sd.getSessions().clear();
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            float[] bData = new float[8 * 4];
            for (int i = 0; i < 4; i++) {
                bData[i * 4 + i] = 1.0f;
            }
            float[] cData = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f};

            INDArray aInput = Nd4j.zeros(DataType.FLOAT, 4, 8);
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("a", aInput);
            inputs.put("b", Nd4j.create(bData, new long[]{8, 4}));
            inputs.put("c", Nd4j.create(cData, new long[]{4, 2}));

            float[] last = null;
            for (int step = 0; step < 24; step++) {
                float[] aData = new float[4 * 8];
                for (int i = 0; i < aData.length; i++) {
                    aData[i] = step + i * 0.125f;
                }
                aInput.assign(Nd4j.create(aData, new long[]{4, 8}));
                INDArray actual = sd.output(inputs, "out").get("out");
                last = actual.toFloatVector();
            }
            return last;
        } finally {
            sd.close();
        }
    }

    // ── file helpers ─────────────────────────────────────────────────────────

    private static List<Path> listFiles(Path dir, String suffix) throws IOException {
        if (!Files.isDirectory(dir)) return List.of();
        try (Stream<Path> stream = Files.list(dir)) {
            return stream.filter(p -> p.getFileName().toString().endsWith(suffix))
                    .sorted().collect(Collectors.toList());
        }
    }

    private static int readLeInt(Path file, int offset) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file.toFile(), "r")) {
            raf.seek(offset);
            byte[] buf = new byte[4];
            raf.readFully(buf);
            return ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getInt();
        }
    }

    private static void corruptMagic(Path spv) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(spv.toFile(), "rw")) {
            raf.seek(0);
            raf.writeInt(0xDEADBEEF);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Environment setters round-trip through the native VulkanConfig")
    void testConfigRoundTrip() {
        assumeTrue(nativeOps != null, "requires -Ptest-vulkan classpath");
        assumeTrue(vulkanDevicePresent, "requires a Vulkan device for backend selection");
        Environment env = Nd4j.getEnvironment();

        String prevDir = env.vulkanSpirvCacheDir();
        boolean prevEnabled = env.vulkanSpirvCacheEnabled();
        long prevMax = env.vulkanPipelineCacheMaxBytes();
        try {
            env.setVulkanSpirvCacheDir("/tmp/vk-roundtrip-test");
            assertEquals("/tmp/vk-roundtrip-test", env.vulkanSpirvCacheDir(),
                    "spirvCacheDir must round-trip through the native VulkanConfig — "
                            + "if this returns the interface default, the backend Environment "
                            + "is not delegating (bindings not regenerated?)");
            env.setVulkanSpirvCacheEnabled(false);
            assertFalse(env.vulkanSpirvCacheEnabled(), "enabled flag must round-trip");
            env.setVulkanPipelineCacheMaxBytes(1234567L);
            assertEquals(1234567L, env.vulkanPipelineCacheMaxBytes(), "maxBytes must round-trip");
        } finally {
            env.setVulkanSpirvCacheDir(prevDir);
            env.setVulkanSpirvCacheEnabled(prevEnabled);
            env.setVulkanPipelineCacheMaxBytes(prevMax);
        }
    }

    /**
     * One phase per JVM (see class doc for the runner loop):
     * cold    — empty dir: misses, then stores; .spv/.meta pairs with valid magic.
     * warm    — fresh JVM, same dir: Tier-1 disk hits, zero stores; Tier-2 blob
     *           (written by the cold JVM) present, valid, and loaded at init.
     * corrupt — all .spv magics flipped beforehand: misses, JIT fallback,
     *           entries re-stored and healed.
     * bypass  — ND4J_VULKAN_ALWAYS_COMPILE=true (via the property→env-var
     *           wiring): no reads, no writes, counters stay zero.
     */
    @Test
    @DisplayName("Cross-JVM phases: cold store / warm hit / corruption heal / alwaysCompile bypass")
    void testDiskCachePhase() throws Exception {
        String phase = System.getProperty("vulkan.cache.test.phase", "");
        assumeTrue(!phase.isEmpty(),
                "phase-driven cross-JVM test — run with -Dvulkan.cache.test.phase=cold|warm|corrupt|bypass "
                        + "and the nd4j.environment.vulkan*Dir properties (see class javadoc)");
        requireVulkanMlir();

        String dirProp = System.getProperty("vulkan.cache.test.dir", "");
        assertFalse(dirProp.isEmpty(), "-Dvulkan.cache.test.dir is required for phase runs");
        Path spirvDir = Paths.get(dirProp, "spirv");
        Path pipeDir = Paths.get(dirProp, "pipe");

        Environment env = Nd4j.getEnvironment();
        // The dirs must have arrived through the surefire env-var wiring so the
        // native config saw them BEFORE the first device context initialized.
        assertEquals(spirvDir.toString(), env.vulkanSpirvCacheDir(),
                "ND4J_VULKAN_SPIRV_CACHE_DIR must reach the native VulkanConfig via the pom env-var wiring");
        assertEquals(pipeDir.toString(), env.vulkanPipelineCacheDir(),
                "ND4J_VULKAN_PIPELINE_CACHE_DIR must reach the native VulkanConfig via the pom env-var wiring");

        switch (phase) {
            case "cold": {
                assertTrue(listFiles(spirvDir, ".spv").isEmpty(),
                        "cold phase requires an empty cache dir — wipe " + dirProp + " first");
                env.clearVulkanCacheCounters();
                eagerWorkload();
                assertTrue(env.vulkanSpirvDiskMisses() > 0, "cold compile must miss first");
                assertTrue(env.vulkanSpirvDiskStores() > 0, "cold compile must store SPIR-V entries");
                List<Path> spvFiles = listFiles(spirvDir, ".spv");
                List<Path> metaFiles = listFiles(spirvDir, ".meta");
                assertFalse(spvFiles.isEmpty(), "spv_<hash>.spv files must exist in " + spirvDir);
                assertEquals(spvFiles.size(), metaFiles.size(), "every .spv needs a .meta sidecar");
                for (Path spv : spvFiles) {
                    assertEquals(SPIRV_MAGIC, readLeInt(spv, 0), "SPIR-V magic in " + spv);
                }
                log.info("cold: stores={} files={}", env.vulkanSpirvDiskStores(), spvFiles.size());
                break;
            }
            case "warm": {
                assertFalse(listFiles(spirvDir, ".spv").isEmpty(),
                        "warm phase requires the cold phase to have populated " + spirvDir);
                // The Tier-2 blob is loaded when the device context initializes —
                // during backend init, BEFORE this test body runs. Read the
                // counter before clearing or the increment is wiped.
                long blobLoadsAtInit = env.vulkanPipelineBlobLoads();
                env.clearVulkanCacheCounters();
                eagerWorkload();
                assertTrue(env.vulkanSpirvDiskHits() > 0,
                        "fresh JVM with identical kernels must hit the Tier-1 disk cache (hits="
                                + env.vulkanSpirvDiskHits() + ", misses=" + env.vulkanSpirvDiskMisses()
                                + ") — key instability across processes if this fails");
                assertEquals(0, env.vulkanSpirvDiskStores(),
                        "a fully warm run must not re-store entries");

                // Tier 2: the cold JVM persisted the driver blob; this JVM must
                // have loaded it when the device context initialized.
                List<Path> blobs = listFiles(pipeDir, ".bin");
                assertFalse(blobs.isEmpty(), "Tier-2 vkpc_<hash>.bin must exist in " + pipeDir);
                for (Path blob : blobs) {
                    assertTrue(Files.size(blob) >= 32, "blob needs the 32-byte header: " + blob);
                    assertEquals(1, readLeInt(blob, 4),
                            "VK_PIPELINE_CACHE_HEADER_VERSION_ONE expected in " + blob);
                }
                assertTrue(blobLoadsAtInit > 0,
                        "device context must load the persisted Tier-2 blob at init (counter read "
                                + "before clearing; was " + blobLoadsAtInit + ")");
                log.info("warm: hits={} blobLoadsAtInit={}", env.vulkanSpirvDiskHits(), blobLoadsAtInit);
                break;
            }
            case "corrupt": {
                List<Path> spvFiles = listFiles(spirvDir, ".spv");
                assertFalse(spvFiles.isEmpty(), "corrupt phase requires populated cache dir");
                for (Path spv : spvFiles) corruptMagic(spv);
                env.clearVulkanCacheCounters();
                eagerWorkload();
                assertTrue(env.vulkanSpirvDiskMisses() > 0, "corrupted .spv must be treated as a miss");
                assertTrue(env.vulkanSpirvDiskStores() > 0, "corrupted entries must be overwritten");
                for (Path spv : listFiles(spirvDir, ".spv")) {
                    assertEquals(SPIRV_MAGIC, readLeInt(spv, 0), "entry must be healed: " + spv);
                }
                log.info("corrupt: misses={} restores={}", env.vulkanSpirvDiskMisses(), env.vulkanSpirvDiskStores());
                break;
            }
            case "bypass": {
                assertTrue(env.vulkanAlwaysCompile(),
                        "bypass phase must run with -Dnd4j.environment.vulkanAlwaysCompile=true "
                                + "(validates the property→env-var→native chain)");
                env.clearVulkanCacheCounters();
                eagerWorkload();
                assertEquals(0, env.vulkanSpirvDiskHits(), "alwaysCompile must bypass reads");
                assertEquals(0, env.vulkanSpirvDiskMisses(), "alwaysCompile must not even attempt reads");
                assertEquals(0, env.vulkanSpirvDiskStores(), "alwaysCompile must skip writes");
                break;
            }
            default:
                throw new IllegalArgumentException("Unknown phase: " + phase);
        }
    }

    /**
     * DSP VULKAN_REPLAY coverage: per-plan pipeline caches give true in-JVM
     * fresh-instance disk hits. Self-skips while the Vulkan input-staging
     * frontier is in flight; arms automatically once it lands.
     */
    @Test
    @DisplayName("DSP replay path populates and reuses the Tier-1 cache across plan lifetimes")
    void testDspReplayPathPopulatesCache() throws Exception {
        requireVulkanMlir();
        Environment env = Nd4j.getEnvironment();

        Path spirvDir = Files.createTempDirectory("vk-spirv-dsp-test");
        String prevSpirvDir = env.vulkanSpirvCacheDir();
        try {
            env.setVulkanSpirvCacheDir(spirvDir.toString());
            env.clearVulkanCacheCounters();

            float[] first;
            try {
                first = dspReplayWorkload();
            } catch (RuntimeException e) {
                String msg = String.valueOf(e.getMessage())
                        + (e.getCause() != null ? " / " + e.getCause().getMessage() : "");
                assumeTrue(!msg.contains(WIP_STAGING_ERROR),
                        "Vulkan DSP input staging is an in-flight frontier "
                                + "('" + WIP_STAGING_ERROR + "') — skipping until it lands. Error: " + msg);
                throw e;
            }

            assertTrue(env.vulkanSpirvDiskStores() > 0,
                    "DSP capture must persist segment kernels");
            env.clearVulkanCacheCounters();
            float[] second = dspReplayWorkload();
            assertEquals(first.length, second.length);
            for (int i = 0; i < first.length; i++) {
                assertTrue(Math.abs(first[i] - second[i]) <= 1e-4f,
                        "warm-start result mismatch at " + i);
            }
            assertTrue(env.vulkanSpirvDiskHits() > 0,
                    "fresh plan with identical MLIR must hit the Tier-1 disk cache");
        } finally {
            env.setVulkanSpirvCacheDir(prevSpirvDir);
        }
    }
}
