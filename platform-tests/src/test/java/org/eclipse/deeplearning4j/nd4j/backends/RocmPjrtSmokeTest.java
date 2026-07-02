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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Smoke tests for running ND4J on AMD GPUs via the ROCm PJRT plugin
 * (xla_rocm_plugin.so from the jax-rocm7-pjrt wheel). See ADR 0102.
 *
 * ARCHITECTURE: AMD is NOT a separate backend. The ROCm plugin is just another
 * PJRT plugin loaded by the SAME native PjrtClientManager that drives the TPU
 * path — the uniform GetPjrtApi() C ABI. You point the manager at the plugin
 * with PJRT_PLUGIN_LIBRARY_PATH (or ROCM_PJRT_PATH / -Drocm.pjrt.path). So this
 * tier lives on the nd4j-tpu (PJRT) module; run it with -Ptest-rocm.
 *
 * The plugin is fetched WITHOUT python (libnd4j/scripts/fetch-pjrt-plugin.sh
 * rocm → curl+jq+unzip of the wheel). Two coverage levels:
 *   - Anywhere (incl. hosted CI, no AMD GPU): the plugin .so exists and exports
 *     the GetPjrtApi entry point (validated by reading the ELF dynamic-symbol
 *     table in pure Java — no toolchain needed).
 *   - AMD/ROCm host only: the plugin dlopens in-process (needs ROCm 7 runtime
 *     libs: libamdhip64.so.7, librocblas.so.5, libMIOpen.so.1, …). Skips
 *     elsewhere via an assumption.
 *
 * No AMD hardware here means these skip cleanly — safe in unfiltered runs.
 */
@Slf4j
@Tag(TagNames.ROCM)
@Tag(TagNames.AMD_GPU)
@Tag(TagNames.BACKEND_DISCOVERY)
@DisplayName("ROCm PJRT smoke tests (ND4J on AMD via xla_rocm_plugin.so)")
public class RocmPjrtSmokeTest {

    private static String resolvePluginPath() {
        List<String> candidates = new ArrayList<>();
        candidates.add(System.getProperty("rocm.pjrt.path"));
        candidates.add(System.getProperty("pjrt.plugin.library.path"));
        candidates.add(System.getenv("PJRT_PLUGIN_LIBRARY_PATH"));
        candidates.add(System.getenv("ROCM_PJRT_PATH"));
        for (String c : candidates) {
            if (c == null || c.isEmpty() || c.contains("${")) continue;
            File f = new File(c);
            if (f.isFile()) return f.getAbsolutePath();
            if (f.isDirectory()) {
                File direct = new File(f, "xla_rocm_plugin.so");
                if (direct.isFile()) return direct.getAbsolutePath();
                File[] hits = f.listFiles((d, n) -> n.endsWith(".so") && n.contains("rocm"));
                if (hits != null && hits.length > 0) return hits[0].getAbsolutePath();
            }
        }
        return null;
    }

    @Test
    @DisplayName("ROCm PJRT plugin exists and exports GetPjrtApi")
    public void pluginExistsAndExportsGetPjrtApi() throws IOException {
        String path = resolvePluginPath();
        assumeTrue(path != null,
                "No ROCm PJRT plugin found — set PJRT_PLUGIN_LIBRARY_PATH / ROCM_PJRT_PATH / "
                        + "-Drocm.pjrt.path (fetch it with libnd4j/scripts/fetch-pjrt-plugin.sh rocm). Skipping.");

        File so = new File(path);
        assertTrue(so.isFile(), "resolved ROCm plugin path is not a file: " + path);
        log.info("ROCm PJRT plugin: {} ({} MB)", path, so.length() / (1024 * 1024));

        assertTrue(elfExportsSymbol(so, "GetPjrtApi"),
                "xla_rocm_plugin.so does not export GetPjrtApi — PjrtClientManager could not dlsym it. "
                        + "Wrong/corrupt wheel?");
        log.info("Verified: exports GetPjrtApi (dlsym-resolvable by PjrtClientManager)");
    }

    @Test
    @DisplayName("ROCm PJRT plugin loads in-process (AMD/ROCm host only)")
    public void pluginLoadsInProcess() {
        String path = resolvePluginPath();
        assumeTrue(path != null, "No ROCm PJRT plugin path set — skipping.");
        // Needs ROCm 7 runtime libraries present; skips on any non-AMD host.
        boolean rocmPresent = new File("/opt/rocm").isDirectory()
                || System.getenv("ROCM_PATH") != null;
        assumeTrue(rocmPresent,
                "ROCm runtime not detected (no /opt/rocm, no ROCM_PATH) — the AMD plugin needs "
                        + "libamdhip64.so.7 / librocblas.so.5 / libMIOpen.so.1 etc. Skipping in-process load.");

        final String lib = path;
        assertDoesNotThrow(() -> System.load(lib),
                "Failed to dlopen the ROCm PJRT plugin — check ROCm 7 install and LD_LIBRARY_PATH");
        log.info("Successfully loaded ROCm PJRT plugin in-process: {}", lib);
    }

    /**
     * Minimal ELF64 dynamic-symbol reader — returns true if {@code symbol} appears
     * in the .dynsym string table. Enough to confirm the GetPjrtApi entry point is
     * exported without any external toolchain (readelf/nm) on the runner. Parses the
     * ELF header → section headers → .dynstr, and scans it for the symbol name.
     */
    private static boolean elfExportsSymbol(File soFile, String symbol) throws IOException {
        try (RandomAccessFile f = new RandomAccessFile(soFile, "r")) {
            byte[] ident = new byte[16];
            f.readFully(ident);
            if (ident[0] != 0x7F || ident[1] != 'E' || ident[2] != 'L' || ident[3] != 'F')
                return false;
            if (ident[4] != 2) return false;              // ELFCLASS64 only (our plugins are x86_64)
            boolean le = ident[5] == 1;                   // little-endian

            f.seek(40);
            long eShoff = readLong(f, le);                // section header table offset
            f.seek(58);
            int eShentsize = readShort(f, le);
            int eShnum = readShort(f, le);
            int eShstrndx = readShort(f, le);

            // Read the section-header string table to find ".dynstr".
            long shstrHdr = eShoff + (long) eShstrndx * eShentsize;
            f.seek(shstrHdr + 24);
            long shstrOff = readLong(f, le);
            f.seek(shstrHdr + 32);
            long shstrSize = readLong(f, le);
            byte[] shstr = new byte[(int) shstrSize];
            f.seek(shstrOff);
            f.readFully(shstr);

            for (int i = 0; i < eShnum; i++) {
                long hdr = eShoff + (long) i * eShentsize;
                f.seek(hdr);
                int nameOff = readInt(f, le);
                int type = readInt(f, le);
                if (type != 3) continue;                  // SHT_STRTAB
                if (!cstrAt(shstr, nameOff).equals(".dynstr")) continue;
                f.seek(hdr + 24);
                long off = readLong(f, le);
                f.seek(hdr + 32);
                long size = readLong(f, le);
                byte[] dynstr = new byte[(int) size];
                f.seek(off);
                f.readFully(dynstr);
                return new String(dynstr, java.nio.charset.StandardCharsets.ISO_8859_1).contains(symbol);
            }
        }
        return false;
    }

    private static String cstrAt(byte[] buf, int off) {
        int end = off;
        while (end < buf.length && buf[end] != 0) end++;
        return new String(buf, off, end - off, java.nio.charset.StandardCharsets.ISO_8859_1);
    }

    private static int readShort(RandomAccessFile f, boolean le) throws IOException {
        int a = f.read(), b = f.read();
        return le ? (a | (b << 8)) : (b | (a << 8));
    }

    private static int readInt(RandomAccessFile f, boolean le) throws IOException {
        int a = f.read(), b = f.read(), c = f.read(), d = f.read();
        return le ? (a | (b << 8) | (c << 16) | (d << 24)) : (d | (c << 8) | (b << 16) | (a << 24));
    }

    private static long readLong(RandomAccessFile f, boolean le) throws IOException {
        long v = 0;
        byte[] x = new byte[8];
        f.readFully(x);
        for (int i = 0; i < 8; i++) {
            int shift = le ? i : (7 - i);
            v |= (long) (x[i] & 0xFF) << (8 * shift);
        }
        return v;
    }
}
