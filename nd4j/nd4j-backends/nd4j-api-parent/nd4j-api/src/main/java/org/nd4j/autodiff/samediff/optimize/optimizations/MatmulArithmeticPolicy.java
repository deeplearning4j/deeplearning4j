/* SPDX-License-Identifier: Apache-2.0 */
package org.nd4j.autodiff.samediff.optimize.optimizations;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Set;

/** Admission rules for rewrites that cannot reproduce an explicit matmul recurrence. */
final class MatmulArithmeticPolicy {
    private MatmulArithmeticPolicy() {}

    static boolean isExplicit(DifferentialFunction function) {
        if (!(function instanceof DynamicCustomOp) || !"matmul".equals(function.opName())) return false;
        DynamicCustomOp op = (DynamicCustomOp) function;
        // Unknown nonzero policies must also survive to validation, never become legacy math.
        return op.numIArguments() > 3 && op.getIArgument(3) != 0;
    }

    /**
     * Lossy whole-graph weight conversion must preserve the operands of explicit math,
     * including constants behind casts, reshapes and other producers. Shared ancestors
     * remain unchanged; unrelated legacy branches are still eligible for quantization.
     */
    static Set<String> protectedInputs(SameDiff sd) {
        Set<String> protectedVariables = new HashSet<>();
        ArrayDeque<String> pending = new ArrayDeque<>();
        for (SameDiffOp op : sd.getOps().values()) {
            if (isExplicit(op.getOp()) && op.getInputsToOp() != null) pending.addAll(op.getInputsToOp());
        }
        while (!pending.isEmpty()) {
            String name = pending.removeFirst();
            if (!protectedVariables.add(name)) continue;
            Variable variable = sd.getVariables().get(name);
            if (variable == null || variable.getOutputOfOp() == null) continue;
            SameDiffOp producer = sd.getOps().get(variable.getOutputOfOp());
            if (producer != null && producer.getInputsToOp() != null) pending.addAll(producer.getInputsToOp());
        }
        return protectedVariables;
    }
}
