/* SPDX-License-Identifier: Apache-2.0 */
package org.nd4j.autodiff.samediff.optimize.optimizations;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Device storage layout for the constant weights of explicit (SERIAL_FMA) matmuls.
 *
 * <p>GPU SERIAL_FMA kernels (native and Triton) give each lane one output column
 * and walk K in ascending order. A weight stored [N, K] and read through a
 * transposing view is K-contiguous, so every lane streams a different row; stored
 * [K, N] c-order, the lanes of one K step read one contiguous segment. On GB10 the
 * row-per-lane form reaches about half the DRAM bandwidth of the coalesced one
 * (ADR 0122).</p>
 *
 * <p>The rewrite materializes the transposed constant once, in place of the
 * original, and binds the matmuls to it directly. Element values and each
 * output's K order are unchanged, so SERIAL_FMA results are bit-identical; only
 * storage order differs, and memory does not grow because the original is
 * replaced (it must have no other consumers). CPU SERIAL_FMA walks K per output
 * and keeps the K-contiguous storage, so the rewrite is CUDA-only.</p>
 */
@Slf4j
public class SerialMatmulLayoutOptimizations extends BaseOptimizerSet {

    protected static final boolean isCudaBackend;

    static {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        isCudaBackend = "CUDA".equalsIgnoreCase(backend);
    }

    /**
     * matmul_serial(X, permute(W, 1, 0)) with constant rank-2 W consumed only by that
     * transpose → matmul_serial(X, W') where W' holds permute(W, 1, 0) in c-order.
     */
    public static class TransposedConstantWeightToKn implements Optimizer {
        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Mmul.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!isCudaBackend || !isUntransposedExplicitMatmul(op)) return false;
            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 2) return false;

            String viewName = inputs.get(1);
            Variable view = sd.getVariables().get(viewName);
            if (view == null || view.getOutputOfOp() == null) return false;
            SameDiffOp transposeOp = sd.getOps().get(view.getOutputOfOp());
            if (transposeOp == null || !swapsRank2Axes(transposeOp)) return false;
            // Every consumer of the transposed view must be an explicit matmul reading it as B;
            // the view then disappears, so no other reader may observe it.
            List<String> viewUsers = view.getInputsForOp();
            if (viewUsers == null || viewUsers.isEmpty()) return false;
            for (String user : viewUsers) {
                SameDiffOp userOp = sd.getOps().get(user);
                if (userOp == null || !isUntransposedExplicitMatmul(userOp)) return false;
                List<String> userInputs = userOp.getInputsToOp();
                if (userInputs == null || userInputs.size() != 2
                        || !viewName.equals(userInputs.get(1)) || viewName.equals(userInputs.get(0))) return false;
            }

            String weightName = transposeOp.getInputsToOp().get(0);
            Variable weight = sd.getVariables().get(weightName);
            if (weight == null || weight.getVariable().getVariableType() != VariableType.CONSTANT) return false;
            List<String> weightUsers = weight.getInputsForOp();
            if (weightUsers == null || weightUsers.size() != 1 || !weightUsers.get(0).equals(transposeOp.getName()))
                return false;
            INDArray stored = constantArrays.getArray(weightName);
            if (stored == null || stored.rank() != 2) return false;

            INDArray kn = stored.transpose().dup('c');
            constantArrays.removeArray(weightName);
            constantArrays.setArray(weightName, kn);
            weight.getVariable().setShape(kn.shape());

            List<String> consumers = new ArrayList<>(viewUsers);
            weightUsers.clear();
            for (String user : consumers) {
                sd.getOps().get(user).getInputsToOp().set(1, weightName);
                weightUsers.add(user);
            }
            view.getInputsForOp().clear();
            OptimizationUtils.removeOp(sd, helper, transposeOp.getName());
            OptimizationUtils.removeVariable(sd, helper, viewName);
            log.debug("SerialMatmulLayout: stored {} as [K,N] {} for {} SERIAL_FMA matmul(s)",
                    weightName, java.util.Arrays.toString(kn.shape()), consumers.size());
            return true;
        }

        private static boolean isUntransposedExplicitMatmul(SameDiffOp op) {
            if (!(op.getOp() instanceof Mmul) || !MatmulArithmeticPolicy.isExplicit(op.getOp())) return false;
            long[] iArgs = ((Mmul) op.getOp()).iArgs();
            for (int i = 0; i < Math.min(3, iArgs.length); i++) if (iArgs[i] != 0) return false;
            return true;
        }

        /** A single-input permute/transpose of a rank-2 variable that exchanges its two axes. */
        private static boolean swapsRank2Axes(SameDiffOp op) {
            DifferentialFunction function = op.getOp();
            if (!(function instanceof DynamicCustomOp)) return false;
            String name = function.opName();
            if (!"permute".equals(name) && !"transpose".equals(name)) return false;
            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 1) return false;
            long[] dims = ((DynamicCustomOp) function).iArgs();
            if (dims.length == 0) return "transpose".equals(name);  // no permutation: reverse the axes
            return dims.length == 2 && dims[0] == 1 && dims[1] == 0;
        }
    }
}
