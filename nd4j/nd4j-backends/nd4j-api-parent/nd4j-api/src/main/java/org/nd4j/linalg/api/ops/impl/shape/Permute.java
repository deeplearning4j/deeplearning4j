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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.primitives.Ints;
import org.nd4j.shade.guava.primitives.Longs;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@Slf4j
public class Permute extends Transpose {

    private long[] reverseDims;

    /**
     * Normalize negative permutation indices to positive equivalents given the rank.
     * For example, -1 with rank=3 -> 2, -2 with rank=3 -> 1, etc.
     */
    private static long[] normalizePermuteDims(long[] permuteDims, int rank) {
        if (permuteDims == null || rank <= 0) return permuteDims;
        boolean hasNegative = false;
        for (long p : permuteDims) {
            if (p < 0) { hasNegative = true; break; }
        }
        if (!hasNegative) return permuteDims;
        long[] normalized = new long[permuteDims.length];
        for (int i = 0; i < permuteDims.length; i++) {
            normalized[i] = permuteDims[i] < 0 ? permuteDims[i] + rank : permuteDims[i];
        }
        return normalized;
    }

    public Permute(SameDiff sameDiff, SDVariable i_v, long... permuteDims) {
        super(sameDiff, i_v);
        // Normalize negative indices if rank is known from the variable shape
        long[] shape = i_v.getShape();
        if (shape != null && shape.length > 0) {
            permuteDims = normalizePermuteDims(permuteDims, shape.length);
        }
        this.permuteDims = permuteDims;
        this.reverseDims = new long[permuteDims.length];
        for (int i = 0; i < reverseDims.length; i++) {
            reverseDims[i] = ArrayUtils.indexOf(permuteDims, i);
        }
        addIArgument(permuteDims);
    }

    public Permute(INDArray input, INDArray result, long... permuteDims) {
        super(input, result);
        permuteDims = normalizePermuteDims(permuteDims, input.rank());
        this.permuteDims = permuteDims;
        this.reverseDims = new long[permuteDims.length];
        for (int i = 0; i < reverseDims.length; i++) {
            reverseDims[i] = ArrayUtils.indexOf(permuteDims, i);
        }
        addIArgument(permuteDims);
    }

    public Permute(SameDiff sd, SDVariable input, SDVariable permuteDims) {
        super(sd, input, permuteDims);
    }

    public Permute(INDArray input, long... permuteDims){
        super(input, null);
        permuteDims = normalizePermuteDims(permuteDims, input.rank());
        this.permuteDims = permuteDims;
        addIArgument(permuteDims);
    }

    public Permute() {
    }

    @Override
    public String opName() {
        return "permute";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret;
        log.debug("[Permute.doDiff] args().length={}, permuteDims={}, reverseDims={}", args().length, java.util.Arrays.toString(permuteDims), java.util.Arrays.toString(reverseDims));
        if(args().length == 1) {
            //Static dimensions
            if(reverseDims == null) {
                log.debug("[Permute.doDiff] ERROR: reverseDims is null! Recomputing from iArguments...");
                if(!iArguments.isEmpty()) {
                    long[] dims = Longs.toArray(iArguments);
                    this.reverseDims = new long[dims.length];
                    for (int i = 0; i < reverseDims.length; i++) {
                        reverseDims[i] = ArrayUtils.indexOf(dims, i);
                    }
                    log.debug("[Permute.doDiff] Recomputed reverseDims={}", java.util.Arrays.toString(reverseDims));
                }
            }
            ret = sameDiff.permute(i_v.get(0), reverseDims);
        } else {
            //Dynamic dimensions
            ret = sameDiff.permute(i_v.get(0), sameDiff.invertPermutation(arg(1)));
        }
        return Collections.singletonList(ret);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if(!iArguments.isEmpty()) {
            this.reverseDims = Longs.toArray(iArguments);
            this.permuteDims = Longs.toArray(iArguments);
            for (int i = 0; i < reverseDims.length; i++) {
                reverseDims[i] = ArrayUtils.indexOf(permuteDims, i);
            }
        }
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {

    }

    @Override
    public boolean initializeOutputs(OpContext ctx) {
        // DSP allocates its own output buffers. The view created below has permuted
        // (non-standard) strides which DynamicShapePlanCompiler captures via
        // getShapeInfoForVar(), corrupting the native plan's buffer layout.
        // Only skip for SameDiff graph execution (sameDiff != null), not direct imperative ops.
        if (sameDiff != null && org.nd4j.autodiff.samediff.internal.InferenceSession.isDynamicShapePlanEnabled()) {
            return super.initializeOutputs(ctx);
        }

        configureFromArguments();

        List<INDArray> inputs = inputArguments();
        if (inputs == null || inputs.isEmpty()) {
            return super.initializeOutputs(ctx);
        }

        INDArray input = inputs.get(0);
        if (input == null || input.isEmpty()) {
            return super.initializeOutputs(ctx);
        }

        int rank = input.rank();

        // Determine permutation: from iArgs, permuteDims field, or default reverse
        long[] perm = null;
        if (!iArguments.isEmpty()) {
            perm = Longs.toArray(iArguments);
        } else if (permuteDims != null && permuteDims.length > 0) {
            perm = permuteDims;
        }

        if (perm == null) {
            // Plain transpose (reverse dims) — fall back to C++
            return super.initializeOutputs(ctx);
        }

        // Handle rank mismatch: adapt permutation to actual input rank.
        // Mirrors the C++ logic in permute.cpp CUSTOM_OP_IMPL and DECLARE_SHAPE_FN.
        if (perm.length != rank) {
            int permSize = perm.length;
            int extraDims = rank - permSize;

            if (extraDims > 0) {
                // Input has more dimensions than perm expects (e.g., expand_dims added leading 1s).
                int leadingOnes = 0;
                for (int i = 0; i < rank && leadingOnes < extraDims; i++) {
                    if (input.size(i) == 1) {
                        leadingOnes++;
                    } else {
                        break;
                    }
                }

                if (leadingOnes >= extraDims) {
                    long[] adapted = new long[rank];
                    for (int i = 0; i < extraDims; i++) {
                        adapted[i] = i;  // identity for extra leading size-1 dims
                    }
                    for (int i = 0; i < permSize; i++) {
                        adapted[extraDims + i] = perm[i] + extraDims;  // shift original indices
                    }
                    perm = adapted;
                } else {
                    // Not enough leading 1s — fall back to identity
                    perm = new long[rank];
                    for (int i = 0; i < rank; i++) perm[i] = i;
                }
            } else {
                // Perm larger than input rank — filter to valid indices
                java.util.List<Long> valid = new java.util.ArrayList<>();
                for (long p : perm) {
                    if (p < rank) valid.add(p);
                }
                if (valid.size() == rank) {
                    perm = new long[rank];
                    for (int i = 0; i < rank; i++) perm[i] = valid.get(i);
                } else {
                    perm = new long[rank];
                    for (int i = 0; i < rank; i++) perm[i] = i;
                }
            }
        }

        // Validate permutation
        for (long p : perm) {
            if (p < 0 || p >= rank) {
                return super.initializeOutputs(ctx);
            }
        }

        // Update iArguments to the adapted permutation so C++ sees consistent args
        iArguments.clear();
        for (long p : perm) {
            iArguments.add(p);
        }

        // Build output shape and strides by permuting input's strides
        long[] outShape = new long[rank];
        long[] outStrides = new long[rank];
        for (int i = 0; i < rank; i++) {
            int srcAxis = (int) perm[i];
            outShape[i] = input.size(srcAxis);
            outStrides[i] = input.stride(srcAxis);
        }

        // Create view sharing the input's data buffer with permuted strides
        INDArray view = Nd4j.create(input.data(), outShape, outStrides, input.offset(), input.ordering());
        addOutputArgument(view);
        return false;  // tells framework: output already set, don't allocate
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return super.calculateOutputDataTypes(dataTypes);
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " + opName());
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }
}
