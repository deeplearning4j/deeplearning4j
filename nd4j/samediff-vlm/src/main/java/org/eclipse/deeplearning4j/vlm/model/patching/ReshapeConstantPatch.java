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
import java.util.function.Function;

/**
 * A patch that reshapes an existing constant to a new shape, optionally
 * regenerating its data using a provided function.
 *
 * <p>Two modes of operation:</p>
 * <ul>
 *   <li><b>Reshape only</b>: Keeps the existing data but changes the shape.
 *       The total element count must remain the same.</li>
 *   <li><b>Regenerate</b>: Creates entirely new data for the new shape using
 *       a user-supplied function (e.g., arange, zeros, ones).</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * // Reshape and regenerate with arange
 * SameDiffGraphPatch patch = ReshapeConstantPatch.withRegeneration(
 *     "position_ids",
 *     new long[]{1, 32},   // expected current shape
 *     new long[]{1, 1024}, // desired new shape
 *     (shape) -> Nd4j.arange(shape[1]).reshape(shape).castTo(DataType.INT64)
 * );
 *
 * // Simple reshape (same element count)
 * SameDiffGraphPatch patch = ReshapeConstantPatch.reshapeOnly(
 *     "my_constant",
 *     new long[]{4, 8},    // expected current shape
 *     new long[]{2, 16}    // desired new shape (same element count)
 * );
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
public class ReshapeConstantPatch implements SameDiffGraphPatch {

    private final String variableName;
    private final long[] expectedCurrentShape;
    private final long[] newShape;
    private final Function<long[], INDArray> dataGenerator;

    /**
     * Create a reshape constant patch.
     *
     * @param variableName the variable name to reshape
     * @param expectedCurrentShape the expected current shape (for auto-detection), or null to skip shape check
     * @param newShape the desired new shape
     * @param dataGenerator function that takes the new shape and returns new data, or null for reshape-only
     */
    public ReshapeConstantPatch(String variableName, long[] expectedCurrentShape,
                                long[] newShape, Function<long[], INDArray> dataGenerator) {
        this.variableName = variableName;
        this.expectedCurrentShape = expectedCurrentShape;
        this.newShape = newShape;
        this.dataGenerator = dataGenerator;
    }

    /**
     * Create a patch that reshapes a constant and regenerates its data.
     *
     * @param variableName the variable name
     * @param expectedCurrentShape the expected current shape for detection
     * @param newShape the desired new shape
     * @param dataGenerator function to generate new data for the given shape
     * @return the patch
     */
    public static ReshapeConstantPatch withRegeneration(String variableName,
                                                        long[] expectedCurrentShape,
                                                        long[] newShape,
                                                        Function<long[], INDArray> dataGenerator) {
        return new ReshapeConstantPatch(variableName, expectedCurrentShape, newShape, dataGenerator);
    }

    /**
     * Create a patch that only reshapes a constant (same element count required).
     *
     * @param variableName the variable name
     * @param expectedCurrentShape the expected current shape for detection
     * @param newShape the desired new shape (must have same total elements)
     * @return the patch
     */
    public static ReshapeConstantPatch reshapeOnly(String variableName,
                                                   long[] expectedCurrentShape,
                                                   long[] newShape) {
        return new ReshapeConstantPatch(variableName, expectedCurrentShape, newShape, null);
    }

    @Override
    public String name() {
        return "reshape-constant-" + variableName;
    }

    @Override
    public String description() {
        return "Reshape constant '" + variableName + "' from " +
                Arrays.toString(expectedCurrentShape) + " to " + Arrays.toString(newShape) +
                (dataGenerator != null ? " (with data regeneration)" : " (reshape only)");
    }

    @Override
    public boolean appliesTo(SameDiff graph) {
        SDVariable var = graph.getVariable(variableName);
        if (var == null || !var.isConstant()) {
            return false;
        }

        // If we have an expected shape, check it matches
        if (expectedCurrentShape != null) {
            INDArray arr = var.getArr();
            if (arr == null) {
                return false;
            }
            return Arrays.equals(arr.shape(), expectedCurrentShape);
        }

        return true;
    }

    @Override
    public boolean apply(SameDiff graph) {
        SDVariable var = graph.getVariable(variableName);
        if (var == null || !var.isConstant()) {
            log.warn("Variable '{}' not found or not a constant", variableName);
            return false;
        }

        INDArray oldArr = var.getArr();
        if (oldArr == null) {
            log.warn("Variable '{}' has no array data", variableName);
            return false;
        }

        log.info("Reshaping constant '{}': {} -> {}",
                variableName, Arrays.toString(oldArr.shape()), Arrays.toString(newShape));

        INDArray newArr;
        if (dataGenerator != null) {
            // Generate entirely new data
            newArr = dataGenerator.apply(newShape);
            log.info("Generated new data for '{}' with shape {} and dtype {}",
                    variableName, Arrays.toString(newArr.shape()), newArr.dataType());
        } else {
            // Reshape only - element count must match
            long oldElements = oldArr.length();
            long newElements = 1;
            for (long dim : newShape) {
                newElements *= dim;
            }

            if (oldElements != newElements) {
                log.error("Cannot reshape '{}' from {} ({} elements) to {} ({} elements) - element count mismatch",
                        variableName, Arrays.toString(oldArr.shape()), oldElements,
                        Arrays.toString(newShape), newElements);
                return false;
            }

            newArr = oldArr.reshape(newShape);
        }

        graph.setArrayForVariable(variableName, newArr);
        return true;
    }
}
