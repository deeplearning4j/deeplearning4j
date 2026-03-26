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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * A specific patch for SmolDocling vision encoder that fixes the position_ids
 * constant from shape [1,32] to [1,1024].
 *
 * <p>The SmolDocling ONNX vision encoder exports position_ids with shape [1,32]
 * which is incorrect -- it should be [1,1024] to support the full sequence length.
 * The values should be arange(0, 1024) cast to INT64.</p>
 *
 * <p>Auto-detection checks for:</p>
 * <ul>
 *   <li>A constant variable named "position_ids" (with or without ":0" suffix)</li>
 *   <li>The constant has shape [1, 32]</li>
 * </ul>
 *
 * <p>The target sequence length can be configured (default: 1024).</p>
 *
 * Adam Gibson
 */
@Slf4j
public class SmolDoclingPositionIdsPatch implements SameDiffGraphPatch {

    /** Default target sequence length for position_ids. */
    public static final int DEFAULT_TARGET_LENGTH = 1024;

    /** Default incorrect source length in the ONNX export. */
    public static final int DEFAULT_SOURCE_LENGTH = 32;

    private static final String POSITION_IDS_NAME = "position_ids";

    private final int targetLength;
    private final int sourceLength;

    /**
     * Create a patch with default settings (fix [1,32] to [1,1024]).
     */
    public SmolDoclingPositionIdsPatch() {
        this(DEFAULT_TARGET_LENGTH, DEFAULT_SOURCE_LENGTH);
    }

    /**
     * Create a patch with a custom target length.
     *
     * @param targetLength the desired position_ids sequence length
     * @param sourceLength the expected incorrect source length to detect
     */
    public SmolDoclingPositionIdsPatch(int targetLength, int sourceLength) {
        this.targetLength = targetLength;
        this.sourceLength = sourceLength;
    }

    @Override
    public String name() {
        return "smoldocling-position-ids";
    }

    @Override
    public String description() {
        return "Fix SmolDocling vision encoder position_ids from [1," + sourceLength +
                "] to [1," + targetLength + "] with arange values";
    }

    @Override
    public boolean appliesTo(SameDiff graph) {
        SDVariable var = graph.getVariable(POSITION_IDS_NAME);
        if (var == null) {
            return false;
        }
        if (!var.isConstant()) {
            return false;
        }

        INDArray arr = var.getArr();
        if (arr == null) {
            return false;
        }

        long[] shape = arr.shape();
        // Check for the known incorrect shape [1, sourceLength]
        return shape.length == 2 && shape[0] == 1 && shape[1] == sourceLength;
    }

    @Override
    public boolean apply(SameDiff graph) {
        SDVariable var = graph.getVariable(POSITION_IDS_NAME);
        if (var == null || !var.isConstant()) {
            log.warn("Variable '{}' not found or not a constant", POSITION_IDS_NAME);
            return false;
        }

        INDArray oldArr = var.getArr();
        if (oldArr == null) {
            log.warn("Variable '{}' has no array data", POSITION_IDS_NAME);
            return false;
        }

        log.info("Fixing SmolDocling position_ids: shape {} -> [1, {}]",
                Arrays.toString(oldArr.shape()), targetLength);

        // Create arange [0, 1, 2, ..., targetLength-1] reshaped to [1, targetLength]
        INDArray newPositionIds = Nd4j.arange(targetLength)
                .reshape(1, targetLength)
                .castTo(DataType.INT64);

        graph.setArrayForVariable(POSITION_IDS_NAME, newPositionIds);

        log.info("SmolDocling position_ids fixed: shape [1, {}], dtype {}",
                targetLength, newPositionIds.dataType());
        return true;
    }
}
