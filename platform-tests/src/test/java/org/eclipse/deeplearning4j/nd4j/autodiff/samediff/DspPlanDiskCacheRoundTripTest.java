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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.execution.DspPlanDiskCache;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.tags.TagNames;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Store→load round trip for the DSP plan disk cache (ADR 0093).
 *
 * Regression guard for the fingerprint-truncation bug: {@code buildInfo()} is
 * a MULTI-LINE string, the {@code .meta} sidecar is line-oriented, and the
 * reader parses a single line back. Before the fix, the raw multi-line value
 * was embedded on write, read back truncated to its first line
 * ("Build Info:"), and the equality check against the full in-memory string
 * could never pass — every disk read was silently rejected as a
 * "native build fingerprint mismatch" and the plan cache never hit, on any
 * backend. The fix canonicalizes the fingerprint to one line at the source
 * ({@code DspPlanDiskCache.getNativeBuildFingerprint()}).
 */
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP plan disk cache store→load round trip (fingerprint regression)")
public class DspPlanDiskCacheRoundTripTest {

    // DynamicShapePlan.DSP_MAGIC ("DSP1") + current version, little-endian.
    private static final int DSP_MAGIC = 0x44535031;
    private static final int DSP_VERSION = 5;

    /** Minimal byte payload that passes DynamicShapePlan.isValidSerializedPlan(). */
    private static byte[] fakePlanBytes() {
        ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(DSP_MAGIC);
        buf.putInt(DSP_VERSION);
        for (int i = 8; i < 64; i++) {
            buf.put((byte) (i * 7));
        }
        return buf.array();
    }

    @Test
    @DisplayName("store() then tryLoadByHash() must hit, and the .meta fingerprint must be single-line")
    void testStoreLoadRoundTrip() throws Exception {
        // Force backend init: getNativeBuildFingerprint() reads buildInfo() via
        // NativeOpsHolder, which needs the backend's native.ops wiring loaded.
        org.nd4j.linalg.factory.Nd4j.getEnvironment();

        Path dir = Files.createTempDirectory("dsp-plan-cache-roundtrip");
        String prevDir = System.getProperty(ND4JSystemProperties.DSP_PLAN_CACHE_DISK_DIR);
        System.setProperty(ND4JSystemProperties.DSP_PLAN_CACHE_DISK_DIR, dir.toString());
        try {
            byte[] plan = fakePlanBytes();
            assertTrue(DynamicShapePlan.isValidSerializedPlan(plan), "test payload must be a valid DSP1 header");
            long hash = DynamicShapePlan.computeStructureHash(plan);

            DspPlanDiskCache.store(hash, plan, 3, 2, 1, "out");

            List<Path> metas;
            try (Stream<Path> s = Files.list(dir)) {
                metas = s.filter(p -> p.getFileName().toString().endsWith(".meta"))
                        .collect(Collectors.toList());
            }
            assertEquals(1, metas.size(), "store() must write exactly one .meta sidecar");

            // The fingerprint value must live entirely on its own line — a
            // multi-line value is exactly the bug this test guards against.
            List<String> lines = Files.readAllLines(metas.get(0));
            String fingerprintLine = lines.stream()
                    .filter(l -> l.startsWith("nativeBuildFingerprint="))
                    .findFirst().orElse(null);
            assertNotNull(fingerprintLine, ".meta must contain a nativeBuildFingerprint line");
            String value = fingerprintLine.substring("nativeBuildFingerprint=".length()).trim();
            assertFalse(value.isEmpty(), "fingerprint value must not be empty");
            assertFalse(value.equals("Build Info:"),
                    "fingerprint is truncated to the first line of a multi-line buildInfo() — "
                            + "the canonicalization fix is missing, and every cache read will be "
                            + "rejected as a fingerprint mismatch");

            byte[] loaded = DspPlanDiskCache.tryLoadByHash(hash);
            assertNotNull(loaded,
                    "tryLoadByHash must hit immediately after store — a null here means the "
                            + "fingerprint comparison rejected our own freshly written entry "
                            + "(the multi-line truncation bug)");
            assertArrayEquals(plan, loaded, "loaded bytes must equal stored bytes");
        } finally {
            if (prevDir == null) {
                System.clearProperty(ND4JSystemProperties.DSP_PLAN_CACHE_DISK_DIR);
            } else {
                System.setProperty(ND4JSystemProperties.DSP_PLAN_CACHE_DISK_DIR, prevDir);
            }
        }
    }
}
