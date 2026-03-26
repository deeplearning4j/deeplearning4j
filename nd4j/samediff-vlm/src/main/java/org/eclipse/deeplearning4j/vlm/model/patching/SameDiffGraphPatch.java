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

import org.nd4j.autodiff.samediff.SameDiff;

/**
 * A patch that can be applied to a SameDiff graph after model import.
 *
 * <p>Patches can fix incorrect shapes, replace constants, rename variables,
 * and perform other graph transformations. They are designed to be composable
 * and applied through a {@link SameDiffPatchPipeline}.</p>
 *
 * <p>Typical use cases:</p>
 * <ul>
 *   <li>Fixing incorrect constant initializer shapes in ONNX models</li>
 *   <li>Replacing constant values with correct data</li>
 *   <li>Reshaping tensors that were exported with wrong dimensions</li>
 * </ul>
 *
 * Adam Gibson
 */
public interface SameDiffGraphPatch {

    /**
     * Human-readable name for this patch.
     *
     * @return the patch name
     */
    String name();

    /**
     * Description of what this patch does.
     *
     * @return the patch description
     */
    String description();

    /**
     * Check if this patch should be applied to the given graph.
     * Can inspect variable names, shapes, op types, etc.
     *
     * @param graph the SameDiff graph to inspect
     * @return true if this patch is applicable to the graph
     */
    boolean appliesTo(SameDiff graph);

    /**
     * Apply the patch to the graph. Returns true if the patch was applied successfully.
     *
     * @param graph the SameDiff graph to patch
     * @return true if the patch was applied successfully
     */
    boolean apply(SameDiff graph);
}
