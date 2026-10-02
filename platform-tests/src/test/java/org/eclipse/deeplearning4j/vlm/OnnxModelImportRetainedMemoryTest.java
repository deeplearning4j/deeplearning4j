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
package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.deallocation.DeallocatableReference;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A model import must free what it does not return. The first run after a native rebuild (the
 * build fingerprint invalidates every cached graph) or after a download imports each ONNX model,
 * saves it, optimizes a copy and saves that; whatever those steps allocate and drop instead of
 * closing waits for garbage collection, which a mostly idle Java heap may not run for a long
 * time. That showed up as a process holding ~4GB more native memory than one that loaded the
 * same models from the cache.
 *
 * <p>The ONNX files are hard-linked into a temporary directory, so the import runs the
 * uncached path and writes its caches there, leaving the shared model cache alone.</p>
 */
@Slf4j
public class OnnxModelImportRetainedMemoryTest {

    /** Native memory the import may leave to garbage collection: measurement noise. */
    private static final long IMPORT_GARBAGE_TOLERANCE_BYTES = 64L * 1024 * 1024;

    @Test
    void importLeavesNothingToCollection(@TempDir Path cacheDir) throws Exception {
        VLMModelDownloader.VLMModel[] models = {
                VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER,
                VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER,
                VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS};
        String[] onnxPaths = new String[models.length];
        for (int i = 0; i < models.length; i++) {
            File onnx = VLMModelDownloader.download(models[i]).getModelFile();
            Path link = cacheDir.resolve(onnx.getName());
            Files.createLink(link, onnx.toPath());
            onnxPaths[i] = link.toString();
        }

        collect();
        String before = footprint();
        SameDiff[] imported = OnnxModelCache.importAllWithCache(onnxPaths);
        try {
            Nd4j.getExecutioner().commit();
            String afterImport = footprint();

            // What the import left to garbage collection, by deallocator and bytes still held
            Map<Long, DeallocatableReference> references = Nd4j.getDeallocatorService().getReferenceMap();
            Map<Long, String> labels = new HashMap<>();
            Map<Long, Long> bytes = new HashMap<>();
            for (Map.Entry<Long, DeallocatableReference> entry : references.entrySet()) {
                Deallocator deallocator = entry.getValue().getDeallocator();
                long held = entry.getValue().getBytes();
                labels.put(entry.getKey(), (deallocator == null ? "none" : deallocator.getClass().getSimpleName())
                        + ":" + held);
                bytes.put(entry.getKey(), held);
            }
            collect();
            String afterCollection = footprint();
            // Freed memory the device pools and the allocator still hold is not the import's
            SameDiffMemoryUtils.trimAllDevicePools();
            String afterPoolTrim = footprint();
            Pointer.trimMemory();
            String afterTrim = footprint();
            Map<String, Integer> reclaimed = new TreeMap<>();
            long reclaimedBytes = 0;
            for (Map.Entry<Long, String> entry : labels.entrySet()) {
                if (!references.containsKey(entry.getKey())) {
                    reclaimed.merge(entry.getValue(), 1, Integer::sum);
                    reclaimedBytes += bytes.get(entry.getKey());
                }
            }
            log.info("[IMPORT] before: {}", before);
            log.info("[IMPORT] after import: {}", afterImport);
            log.info("[IMPORT] after collection: {}", afterCollection);
            log.info("[IMPORT] after pool trim: {}", afterPoolTrim);
            log.info("[IMPORT] after malloc trim: {}", afterTrim);
            log.info("[IMPORT] left to collection: {}MB in {} buffers (deallocator:bytes=count): {}",
                    mb(reclaimedBytes), reclaimed.values().stream().mapToInt(Integer::intValue).sum(), reclaimed);

            assertTrue(reclaimedBytes <= IMPORT_GARBAGE_TOLERANCE_BYTES,
                    "importing " + models.length + " models left " + mb(reclaimedBytes)
                            + "MB of native buffers to garbage collection: " + reclaimed);
        } finally {
            for (SameDiff sd : imported) {
                sd.close();
            }
        }
    }

    /** Collect, then free everything collection found. */
    private static void collect() {
        for (int i = 0; i < 3; i++) {
            System.gc();
        }
        Nd4j.getDeallocatorService().forceFlushAll();
    }

    /** Process RSS, the committed and used Java heap, and resident memory by mapping kind. */
    private static String footprint() {
        Runtime runtime = Runtime.getRuntime();
        return "rss=" + mb(Pointer.physicalBytes()) + "MB heapCommitted=" + mb(runtime.totalMemory())
                + "MB heapUsed=" + mb(runtime.totalMemory() - runtime.freeMemory()) + "MB " + residentByMapping();
    }

    /**
     * Resident kB of /proc/self/smaps by mapping kind (anonymous, brk heap, NVIDIA device files,
     * other files), with the largest anonymous mappings; empty where smaps does not exist.
     */
    private static String residentByMapping() {
        Path smaps = Path.of("/proc/self/smaps");
        if (!Files.isReadable(smaps)) {
            return "";
        }
        Map<String, Long> byKind = new TreeMap<>();
        List<Long> anonymous = new ArrayList<>();
        try {
            String kind = null;
            for (String line : Files.readAllLines(smaps)) {
                String[] fields = line.trim().split("\\s+");
                if (fields.length >= 5 && fields[0].contains("-") && !line.endsWith(":")
                        && fields[0].matches("[0-9a-f]+-[0-9a-f]+")) {
                    String path = fields.length >= 6 ? fields[5] : "";
                    kind = path.isEmpty() ? "anon" : path.startsWith("/dev/nvidia") ? "nvidia"
                            : path.startsWith("[") ? path : "file";
                } else if (kind != null && fields.length >= 2 && fields[0].equals("Rss:")) {
                    long kb = Long.parseLong(fields[1]);
                    byKind.merge(kind, kb, Long::sum);
                    if (kind.equals("anon")) {
                        anonymous.add(kb);
                    }
                }
            }
        } catch (IOException e) {
            return "smaps unreadable: " + e.getMessage();
        }
        anonymous.sort(Collections.reverseOrder());
        StringBuilder out = new StringBuilder("residentMB{");
        byKind.forEach((k, kb) -> out.append(k).append('=').append(kb / 1024).append(' '));
        out.append("largestAnonMB=");
        for (int i = 0; i < Math.min(6, anonymous.size()); i++) {
            out.append(i == 0 ? "" : ",").append(anonymous.get(i) / 1024);
        }
        return out.append('}').toString();
    }

    private static long mb(long bytes) {
        return bytes / (1024 * 1024);
    }
}
