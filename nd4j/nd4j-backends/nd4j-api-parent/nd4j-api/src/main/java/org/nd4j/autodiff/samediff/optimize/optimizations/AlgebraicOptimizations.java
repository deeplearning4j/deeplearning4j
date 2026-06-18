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

package org.nd4j.autodiff.samediff.optimize.optimizations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.DivOp;

import java.util.List;

/**
 * Algebraic simplification optimizations inspired by Luminal's e-graph rewrites.
 * These optimizations eliminate identity operations and simplify expressions.
 *
 * Supported simplifications:
 * - x + 0 -> x (AddZero)
 * - x - 0 -> x (SubtractZero)
 * - x * 1 -> x (MultiplyOne)
 * - x * 0 -> 0 (MultiplyZero)
 * - x - x -> 0 (SubtractSelf)
 * - x / 1 -> x (DivideOne)
 * - x / x -> 1 (DivideSelf)
 */
@Slf4j
public class AlgebraicOptimizations extends BaseOptimizerSet {

    /**
     * Check if an array is a scalar constant with a specific value.
     */
    private static boolean isScalarConstant(SameDiff sd, ArrayHolder constantArrays,
                                            ArrayHolder variablesArrays, String varName, double value) {
        SDVariable sdVar = sd.getVariable(varName);
        if (sdVar == null) {
            return false;
        }

        // Must be a constant - check variable type explicitly
        if (sdVar.getVariableType() != VariableType.CONSTANT) {
            return false;
        }

        // Get the array from the constantArrays parameter (the reliable way during optimization)
        INDArray arr = constantArrays.getArray(varName);
        if (arr == null) {
            return false;
        }

        // Check scalar or single element arrays (shape [1] is also considered scalar-like)
        if (arr.isScalar() || arr.length() == 1) {
            return Math.abs(arr.getDouble(0) - value) < 1e-10;
        }

        // Check if all elements are the same value (for broadcast constants like [1,1,1])
        // Only check for small arrays to avoid performance issues
        if (arr.length() <= 1000) {
            double[] data = arr.toDoubleVector();
            for (double d : data) {
                if (Math.abs(d - value) > 1e-10) return false;
            }
            return true;
        }

        return false;
    }

    /**
     * Replace all uses of oldVar with newVar and remove the op that produced oldVar.
     */
    private static void replaceWithInput(SameDiff sd, OptimizationHelper helper,
                                         SameDiffOp op, String oldOutput, String newInput) {
        // Replace all uses of the output with the input
        OptimizationUtils.replaceOpInputsWith(sd, helper, oldOutput, newInput);

        // If the old output was a registered graph output, update the output list
        // to point to the replacement. Otherwise outputSingle() returns null because
        // the orphaned variable has no producing op.
        List<String> graphOutputs = sd.outputs();
        if (graphOutputs != null) {
            for (int i = 0; i < graphOutputs.size(); i++) {
                if (graphOutputs.get(i).equals(oldOutput)) {
                    graphOutputs.set(i, newInput);
                }
            }
        }

        // Remove the op
        OptimizationUtils.removeOp(sd, helper, op.getName());

        // Remove the old output variable
        OptimizationUtils.removeVariable(sd, helper, oldOutput);
    }

    /**
     * Simplifies x + 0 -> x
     * Also handles 0 + x -> x
     */
    public static class AddZero implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof AddOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String outputVar = outputs.get(0);

            // Check if either input is constant 0
            String nonZeroInput = null;
            if (isScalarConstant(sd, constantArrays, variablesArrays, inputs.get(0), 0.0)) {
                nonZeroInput = inputs.get(1);
            } else if (isScalarConstant(sd, constantArrays, variablesArrays, inputs.get(1), 0.0)) {
                nonZeroInput = inputs.get(0);
            }

            if (nonZeroInput == null) {
                return false;
            }

            log.debug("Applying x + 0 -> x optimization: {} + 0 -> {}", nonZeroInput, nonZeroInput);
            replaceWithInput(sd, helper, op, outputVar, nonZeroInput);
            return true;
        }
    }

    /**
     * Simplifies x - 0 -> x
     * Note: 0 - x is NOT simplified (that would be -x, not x)
     */
    public static class SubtractZero implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof SubOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String outputVar = outputs.get(0);

            // Only check if second input (subtrahend) is constant 0
            // x - 0 = x, but 0 - x = -x (not x)
            if (!isScalarConstant(sd, constantArrays, variablesArrays, inputs.get(1), 0.0)) {
                return false;
            }

            String minuend = inputs.get(0);
            log.debug("Applying x - 0 -> x optimization: {} - 0 -> {}", minuend, minuend);
            replaceWithInput(sd, helper, op, outputVar, minuend);
            return true;
        }
    }

    /**
     * Simplifies x * 1 -> x
     * Also handles 1 * x -> x
     */
    public static class MultiplyOne implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof MulOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String outputVar = outputs.get(0);

            // Check if either input is constant 1
            String nonOneInput = null;
            if (isScalarConstant(sd, constantArrays, variablesArrays, inputs.get(0), 1.0)) {
                nonOneInput = inputs.get(1);
            } else if (isScalarConstant(sd, constantArrays, variablesArrays, inputs.get(1), 1.0)) {
                nonOneInput = inputs.get(0);
            }

            if (nonOneInput == null) {
                return false;
            }

            log.debug("Applying x * 1 -> x optimization: {} * 1 -> {}", nonOneInput, nonOneInput);
            replaceWithInput(sd, helper, op, outputVar, nonOneInput);
            return true;
        }
    }

    /**
     * Simplifies x * 0 -> 0
     * Also handles 0 * x -> 0
     */
    public static class MultiplyZero implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof MulOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String outputVar = outputs.get(0);

            // Check if either input is constant 0
            String zeroInput = null;
            if (isScalarConstant(sd, constantArrays, variablesArrays, inputs.get(0), 0.0)) {
                zeroInput = inputs.get(0);
            } else if (isScalarConstant(sd, constantArrays, variablesArrays, inputs.get(1), 0.0)) {
                zeroInput = inputs.get(1);
            }

            if (zeroInput == null) {
                return false;
            }

            log.debug("Applying x * 0 -> 0 optimization: replacing {} with {}", outputVar, zeroInput);
            replaceWithInput(sd, helper, op, outputVar, zeroInput);
            return true;
        }
    }

    /**
     * Simplifies x - x -> 0
     * Detects when both inputs to subtraction are the same variable.
     */
    public static class SubtractSelf implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof SubOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            // Check if both inputs are the same variable
            if (!inputs.get(0).equals(inputs.get(1))) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String outputVar = outputs.get(0);

            log.debug("Applying x - x -> 0 optimization for variable: {}", inputs.get(0));

            // Get the shape from the input to create a zero constant with the same shape
            SDVariable inputVar = sd.getVariable(inputs.get(0));
            if (inputVar == null) return false;

            try {
                // Create a zero constant with the same datatype
                SDVariable zero = sd.constant("zero_" + System.nanoTime(),
                    org.nd4j.linalg.factory.Nd4j.zeros(inputVar.dataType(), 1));

                // Replace uses
                OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, zero.name());
                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeVariable(sd, helper, outputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to apply x - x -> 0: {}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * Simplifies x / 1 -> x
     */
    public static class DivideOne implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof DivOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String outputVar = outputs.get(0);

            // Check if the divisor (second input) is constant 1
            if (!isScalarConstant(sd, constantArrays, variablesArrays, inputs.get(1), 1.0)) {
                return false;
            }

            String dividend = inputs.get(0);
            log.debug("Applying x / 1 -> x optimization: {} / 1 -> {}", dividend, dividend);
            replaceWithInput(sd, helper, op, outputVar, dividend);
            return true;
        }
    }

    /**
     * Simplifies x / x -> 1.
     * Detects when both inputs to division are the same variable.
     */
    public static class DivideSelf implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof DivOp)) {
                return false;
            }

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                return false;
            }

            if (!inputs.get(0).equals(inputs.get(1))) {
                return false;
            }

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) {
                return false;
            }
            String outputVar = outputs.get(0);

            SDVariable inputVar = sd.getVariable(inputs.get(0));
            if (inputVar == null) return false;

            log.debug("Applying x / x -> 1 optimization for variable: {}", inputs.get(0));

            try {
                SDVariable one = sd.constant("one_" + System.nanoTime(),
                    org.nd4j.linalg.factory.Nd4j.ones(inputVar.dataType(), 1));

                OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, one.name());

                // Update graph outputs before removal
                List<String> graphOutputs = sd.outputs();
                if (graphOutputs != null) {
                    for (int i = 0; i < graphOutputs.size(); i++) {
                        if (graphOutputs.get(i).equals(outputVar)) {
                            graphOutputs.set(i, one.name());
                        }
                    }
                }

                OptimizationUtils.removeOp(sd, helper, op.getName());
                OptimizationUtils.removeVariable(sd, helper, outputVar);

                return true;
            } catch (Exception e) {
                log.warn("Failed to apply x / x -> 1: {}", e.getMessage());
                return false;
            }
        }
    }
}
