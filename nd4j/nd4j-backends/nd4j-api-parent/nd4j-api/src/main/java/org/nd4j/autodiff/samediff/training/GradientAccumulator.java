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

package org.nd4j.autodiff.samediff.training;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * Accumulates gradients over multiple micro-batches for gradient accumulation training.
 *
 * <p>Gradient accumulation is useful when the desired batch size doesn't fit in memory.
 * Instead of processing the full batch at once, gradients are accumulated over multiple
 * smaller micro-batches, then averaged and applied.</p>
 *
 * <p>This is particularly important for mixed precision training where memory constraints
 * may require smaller batch sizes to avoid running out of memory.</p>
 *
 * <p>Example usage:</p>
 * <pre>
 * GradientAccumulator accumulator = new GradientAccumulator(4); // 4 accumulation steps
 *
 * for (int i = 0; i &lt; numMicroBatches; i++) {
 *     Map&lt;String, INDArray&gt; gradients = computeGradients(microBatch);
 *     accumulator.accumulate(gradients);
 *
 *     if (accumulator.isReady()) {
 *         Map&lt;String, INDArray&gt; avgGradients = accumulator.getAndReset();
 *         applyUpdates(avgGradients);
 *     }
 * }
 * </pre>
 *
 * @author Adam Gibson
 */
@Slf4j
public class GradientAccumulator {

    /**
     * Number of accumulation steps before applying updates.
     */
    @Getter
    private final int accumulationSteps;

    /**
     * Current step within the accumulation cycle.
     */
    @Getter
    private int currentStep;

    /**
     * Accumulated gradients for each variable.
     */
    private final Map<String, INDArray> accumulatedGradients;

    /**
     * Create a gradient accumulator with the specified number of accumulation steps.
     *
     * @param accumulationSteps Number of micro-batches to accumulate before updating.
     *                          Must be >= 1. A value of 1 means no accumulation.
     */
    public GradientAccumulator(int accumulationSteps) {
        if (accumulationSteps < 1) {
            throw new IllegalArgumentException("accumulationSteps must be >= 1, got: " + accumulationSteps);
        }
        this.accumulationSteps = accumulationSteps;
        this.currentStep = 0;
        this.accumulatedGradients = new HashMap<>();
    }

    /**
     * Accumulate gradients from a micro-batch.
     *
     * @param varName  The variable name
     * @param gradient The gradient for this variable from the current micro-batch
     */
    public void accumulate(String varName, INDArray gradient) {
        if (accumulationSteps == 1) {
            // No accumulation needed - just store directly
            accumulatedGradients.put(varName, gradient.dup());
            return;
        }

        INDArray accumulated = accumulatedGradients.get(varName);
        if (accumulated == null) {
            // First accumulation - store a copy in FP32 for numerical stability
            if (gradient.dataType() == DataType.FLOAT || gradient.dataType() == DataType.DOUBLE) {
                accumulated = gradient.dup();
            } else {
                // Cast to FP32 for accumulation to maintain precision
                accumulated = gradient.castTo(DataType.FLOAT);
            }
            accumulatedGradients.put(varName, accumulated);
        } else {
            // Add to existing accumulated gradients
            if (gradient.dataType() != accumulated.dataType()) {
                accumulated.addi(gradient.castTo(accumulated.dataType()));
            } else {
                accumulated.addi(gradient);
            }
        }
    }

    /**
     * Accumulate all gradients from a micro-batch.
     *
     * @param gradients Map of variable names to their gradients
     */
    public void accumulate(Map<String, INDArray> gradients) {
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            accumulate(entry.getKey(), entry.getValue());
        }
    }

    /**
     * Increment the step counter. Call this after processing a micro-batch.
     */
    public void step() {
        currentStep++;
    }

    /**
     * Check if we've accumulated enough steps and should apply updates.
     *
     * @return true if currentStep >= accumulationSteps
     */
    public boolean isReady() {
        return currentStep >= accumulationSteps;
    }

    /**
     * Get the accumulated (averaged) gradients and reset the accumulator.
     * Call this when isReady() returns true.
     * The returned arrays are owned by the caller - they will not be closed by this class.
     *
     * @return Map of variable names to averaged gradients
     */
    public Map<String, INDArray> getAndReset() {
        Map<String, INDArray> result = new HashMap<>();

        for (Map.Entry<String, INDArray> entry : accumulatedGradients.entrySet()) {
            INDArray accumulated = entry.getValue();
            if (accumulationSteps > 1) {
                // Average the gradients
                accumulated.divi(accumulationSteps);
            }
            result.put(entry.getKey(), accumulated);
        }

        // Clear the map without closing arrays - they are now owned by the caller
        accumulatedGradients.clear();
        currentStep = 0;
        return result;
    }

    /**
     * Get the averaged gradient for a specific variable and remove it from accumulator.
     *
     * @param varName The variable name
     * @return The averaged gradient, or null if not found
     */
    public INDArray getAndRemove(String varName) {
        INDArray accumulated = accumulatedGradients.remove(varName);
        if (accumulated != null && accumulationSteps > 1) {
            accumulated.divi(accumulationSteps);
        }
        return accumulated;
    }

    /**
     * Reset the accumulator, clearing all accumulated gradients and resetting step counter.
     */
    public void reset() {
        // Close existing arrays to free memory
        for (INDArray arr : accumulatedGradients.values()) {
            if (arr != null && !arr.wasClosed()) {
                arr.close();
            }
        }
        accumulatedGradients.clear();
        currentStep = 0;
    }

    /**
     * Check if gradient accumulation is enabled (accumulationSteps > 1).
     *
     * @return true if accumulation is enabled
     */
    public boolean isEnabled() {
        return accumulationSteps > 1;
    }

    /**
     * Get the number of variables currently being accumulated.
     *
     * @return Number of variables with accumulated gradients
     */
    public int getNumVariables() {
        return accumulatedGradients.size();
    }

    /**
     * Check if gradients have been accumulated for a specific variable.
     *
     * @param varName The variable name
     * @return true if gradients exist for this variable
     */
    public boolean hasGradient(String varName) {
        return accumulatedGradients.containsKey(varName);
    }
}
