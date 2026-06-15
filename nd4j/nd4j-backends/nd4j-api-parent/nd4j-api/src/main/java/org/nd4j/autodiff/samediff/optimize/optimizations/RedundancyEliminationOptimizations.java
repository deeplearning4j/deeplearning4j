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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.Gather;
import org.nd4j.linalg.api.ops.impl.shape.Slice;
import org.nd4j.linalg.api.ops.impl.shape.StridedSlice;

import java.util.List;
import java.util.Set;

/**
 * Redundancy elimination optimizations for no-op shape manipulation.
 *
 * <p>These passes remove shape ops that produce the same result as their input,
 * which is common in ONNX-imported models due to framework-level canonicalization.</p>
 *
 * <ul>
 *   <li><b>Single-input concat</b>: {@code concat(x, axis)} with one input is identity</li>
 *   <li><b>Full-extent slice</b>: {@code slice(x, [0,...], shape(x))} covering all elements</li>
 *   <li><b>Identity gather</b>: {@code gather(x, [0,1,...,n-1], axis)} with contiguous indices</li>
 * </ul>
 */
@Slf4j
public class RedundancyEliminationOptimizations extends BaseOptimizerSet {

    /**
     * Remove concat ops that have only a single data input.
     * {@code concat(x, axis)} with one tensor is just {@code x}.
     *
     * <p>Common in ONNX models after other optimizations eliminate operands.</p>
     */
    public static class RemoveSingleInputConcat implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Concat.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Concat)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() != 1) return false;

            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String singleInput = inputs.get(0);
            String outputVar = outputs.get(0);

            log.debug("Removing single-input concat: concat({}) -> {}", singleInput, singleInput);

            OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, singleInput);

            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(outputVar)) {
                        graphOutputs.set(i, singleInput);
                    }
                }
            }

            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputVar);
            return true;
        }
    }

    /**
     * Remove slice ops that extract the entire input tensor.
     *
     * <p>A slice starting at [0, 0, ...] with sizes matching the input shape
     * is a no-op. Common in ONNX models that insert explicit slices for
     * shape compatibility.</p>
     */
    public static class RemoveFullExtentSlice implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Slice.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Slice)) return false;
            Slice slice = (Slice) op.getOp();

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.isEmpty()) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String inputVar = inputs.get(0);
            String outputVar = outputs.get(0);

            // Slice stores begin and size in iArguments:
            // iArgs = [begin0, begin1, ..., size0, size1, ...]
            long[] iArgs = slice.iArgs();
            if (iArgs == null || iArgs.length == 0) return false;

            // Must be even length (begin + size pairs)
            if (iArgs.length % 2 != 0) return false;
            int rank = iArgs.length / 2;

            // Get input shape
            SDVariable inputSdVar = sd.getVariable(inputVar);
            if (inputSdVar == null) return false;
            long[] inputShape = inputSdVar.getShape();
            if (inputShape == null || inputShape.length != rank) return false;

            // Check if all begins are 0 and all sizes match input shape
            for (int i = 0; i < rank; i++) {
                long begin = iArgs[i];
                long size = iArgs[rank + i];
                if (begin != 0) return false;
                if (inputShape[i] < 0) return false; // Dynamic dim — can't verify
                if (size != inputShape[i] && size != -1) return false; // -1 means "all remaining"
            }

            log.debug("Removing full-extent slice: slice({}) -> {}", inputVar, inputVar);

            OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, inputVar);

            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(outputVar)) {
                        graphOutputs.set(i, inputVar);
                    }
                }
            }

            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputVar);
            return true;
        }
    }

    /**
     * Remove gather ops that select all elements in order (identity gather).
     *
     * <p>{@code gather(x, [0, 1, 2, ..., n-1], axis)} is equivalent to {@code x}
     * when the indices cover the entire axis dimension in order.</p>
     */
    public static class RemoveIdentityGather implements Optimizer {

        @Override
        public Set<Class<? extends DifferentialFunction>> getApplicableOpTypes() {
            return Set.of(Gather.class);
        }

        @Override
        public boolean checkAndApply(SameDiff sd, OptimizationHelper helper, SameDiffOp op,
                                     ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            if (!(op.getOp() instanceof Gather)) return false;

            List<String> inputs = op.getInputsToOp();
            if (inputs == null || inputs.size() < 2) return false;
            List<String> outputs = op.getOutputsOfOp();
            if (outputs == null || outputs.isEmpty()) return false;

            String dataInput = inputs.get(0);
            String indicesInput = inputs.get(1);
            String outputVar = outputs.get(0);

            // Indices must be a constant
            SDVariable indicesSdVar = sd.getVariable(indicesInput);
            if (indicesSdVar == null || indicesSdVar.getVariableType() != VariableType.CONSTANT) return false;

            INDArray indicesArr = constantArrays.getArray(indicesInput);
            if (indicesArr == null) return false;

            // Must be 1D
            if (indicesArr.rank() > 1) return false;

            // Get the axis from iArguments
            Gather gather = (Gather) op.getOp();
            long[] iArgs = gather.iArgs();
            int axis = (iArgs != null && iArgs.length > 0) ? (int) iArgs[0] : 0;

            // Get the input shape to check against
            SDVariable dataSdVar = sd.getVariable(dataInput);
            if (dataSdVar == null) return false;
            long[] inputShape = dataSdVar.getShape();
            if (inputShape == null) return false;

            // Normalize negative axis
            if (axis < 0) axis += inputShape.length;
            if (axis < 0 || axis >= inputShape.length) return false;

            long axisDim = inputShape[axis];
            if (axisDim < 0) return false; // Dynamic dimension

            // Check if indices are [0, 1, 2, ..., axisDim-1]
            if (indicesArr.length() != axisDim) return false;
            for (long i = 0; i < axisDim; i++) {
                if (indicesArr.getLong(i) != i) return false;
            }

            log.debug("Removing identity gather: gather({}, [0..{}], axis={}) -> {}",
                    dataInput, axisDim - 1, axis, dataInput);

            OptimizationUtils.replaceOpInputsWith(sd, helper, outputVar, dataInput);

            List<String> graphOutputs = sd.outputs();
            if (graphOutputs != null) {
                for (int i = 0; i < graphOutputs.size(); i++) {
                    if (graphOutputs.get(i).equals(outputVar)) {
                        graphOutputs.set(i, dataInput);
                    }
                }
            }

            OptimizationUtils.removeOp(sd, helper, op.getName());
            OptimizationUtils.removeVariable(sd, helper, outputVar);
            return true;
        }
    }
}
