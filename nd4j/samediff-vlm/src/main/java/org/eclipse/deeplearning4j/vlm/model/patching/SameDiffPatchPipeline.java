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

package org.eclipse.deeplearning4j.vlm.model.patching;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A pipeline that holds an ordered list of {@link SameDiffGraphPatch} instances
 * and applies them to a SameDiff graph.
 *
 * <p>The pipeline supports two modes of operation:</p>
 * <ul>
 *   <li><b>Auto-detect mode</b> ({@link #applyMatching(SameDiff)}): Only applies patches
 *       whose {@link SameDiffGraphPatch#appliesTo(SameDiff)} returns true.</li>
 *   <li><b>Force mode</b> ({@link #applyAll(SameDiff)}): Applies all patches regardless
 *       of the auto-detection result.</li>
 * </ul>
 *
 * <p>Patches are applied in the order they were added to the pipeline.</p>
 *
 * Adam Gibson
 */
@Slf4j
public class SameDiffPatchPipeline {

    private final List<SameDiffGraphPatch> patches;

    /**
     * Create an empty pipeline.
     */
    public SameDiffPatchPipeline() {
        this.patches = new ArrayList<>();
    }

    /**
     * Create a pipeline with the given patches.
     *
     * @param patches the patches to include
     */
    public SameDiffPatchPipeline(List<SameDiffGraphPatch> patches) {
        this.patches = new ArrayList<>(patches);
    }

    /**
     * Create a pipeline with the given patches.
     *
     * @param patches the patches to include
     * @return the pipeline
     */
    public static SameDiffPatchPipeline of(SameDiffGraphPatch... patches) {
        return new SameDiffPatchPipeline(Arrays.asList(patches));
    }

    /**
     * Add a patch to the pipeline.
     *
     * @param patch the patch to add
     * @return this pipeline for chaining
     */
    public SameDiffPatchPipeline addPatch(SameDiffGraphPatch patch) {
        patches.add(patch);
        return this;
    }

    /**
     * Get an unmodifiable view of the patches in this pipeline.
     *
     * @return the patches
     */
    public List<SameDiffGraphPatch> getPatches() {
        return Collections.unmodifiableList(patches);
    }

    /**
     * Apply only the patches that match the given graph (auto-detect mode).
     * Each patch's {@link SameDiffGraphPatch#appliesTo(SameDiff)} is checked first.
     *
     * @param graph the SameDiff graph to patch
     * @return the list of patch names that were successfully applied
     */
    public List<String> applyMatching(SameDiff graph) {
        List<String> applied = new ArrayList<>();
        for (SameDiffGraphPatch patch : patches) {
            try {
                if (patch.appliesTo(graph)) {
                    log.info("Patch '{}' matches graph, applying: {}", patch.name(), patch.description());
                    boolean success = patch.apply(graph);
                    if (success) {
                        applied.add(patch.name());
                        log.info("Patch '{}' applied successfully", patch.name());
                    } else {
                        log.warn("Patch '{}' returned false (not applied)", patch.name());
                    }
                } else {
                    log.debug("Patch '{}' does not apply to this graph, skipping", patch.name());
                }
            } catch (Exception e) {
                log.error("Patch '{}' failed with exception: {}", patch.name(), e.getMessage(), e);
            }
        }

        if (applied.isEmpty()) {
            log.info("No patches were applied to the graph");
        } else {
            log.info("Applied {} patch(es): {}", applied.size(), applied);
        }
        return applied;
    }

    /**
     * Apply all patches in the pipeline regardless of auto-detection.
     *
     * @param graph the SameDiff graph to patch
     * @return the list of patch names that were successfully applied
     */
    public List<String> applyAll(SameDiff graph) {
        List<String> applied = new ArrayList<>();
        for (SameDiffGraphPatch patch : patches) {
            try {
                log.info("Force-applying patch '{}': {}", patch.name(), patch.description());
                boolean success = patch.apply(graph);
                if (success) {
                    applied.add(patch.name());
                    log.info("Patch '{}' applied successfully", patch.name());
                } else {
                    log.warn("Patch '{}' returned false (not applied)", patch.name());
                }
            } catch (Exception e) {
                log.error("Patch '{}' failed with exception: {}", patch.name(), e.getMessage(), e);
            }
        }

        if (applied.isEmpty()) {
            log.info("No patches were applied to the graph");
        } else {
            log.info("Applied {} patch(es): {}", applied.size(), applied);
        }
        return applied;
    }

    /**
     * Check if this pipeline has any patches.
     *
     * @return true if the pipeline is empty
     */
    public boolean isEmpty() {
        return patches.isEmpty();
    }

    /**
     * Get the number of patches in this pipeline.
     *
     * @return the number of patches
     */
    public int size() {
        return patches.size();
    }
}
