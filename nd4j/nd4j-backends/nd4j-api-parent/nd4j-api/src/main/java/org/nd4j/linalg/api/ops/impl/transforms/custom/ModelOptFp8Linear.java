/* SPDX-License-Identifier: Apache-2.0 */
package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * ModelOpt FP8 inference linear: X[...,K] times FLOAT8 E4M3 W[N,K] transposed.
 * X is FLOAT, HALF or BFLOAT16. The third input is weightScale and the fourth
 * is inputScale: both positive finite FLOAT rank-zero dequantization scales.
 * Activations are quantized using saturated E4M3 round-to-nearest-even of
 * X/inputScale. Products of (quantized X * inputScale) and (W * weightScale)
 * accumulate in FP32. Output [...,N] has X's dtype, or FLOAT when requested.
 * Native validation owns shapes, scale values and output alias rejection.
 */
public class ModelOptFp8Linear extends DynamicCustomOp {
    public ModelOptFp8Linear() {
        // Serialization restores inputs and IArgs, which are the source of truth.
    }

    public ModelOptFp8Linear(SameDiff sd, SDVariable x, SDVariable w,
                            SDVariable scale, SDVariable secondScale, boolean floatOutput) {
        super(null, sd, new SDVariable[]{x, w, scale, secondScale});
        addIArgument(floatOutput ? 1 : 0);
    }

    public ModelOptFp8Linear(INDArray x, INDArray w, INDArray scale,
                            INDArray secondScale, boolean floatOutput) {
        super(null, new INDArray[]{x, w, scale, secondScale}, null);
        addIArgument(floatOutput ? 1 : 0);
    }

    @Override
    public String opName() {
        return "modelopt_fp8_linear";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkArgument(inputDataTypes != null && inputDataTypes.size() == 4,
                "ModelOptFp8Linear requires four input dtypes");
        DataType x = inputDataTypes.get(0);
        Preconditions.checkArgument(x == DataType.FLOAT || x == DataType.HALF || x == DataType.BFLOAT16,
                "ModelOptFp8Linear activation must be FLOAT, HALF or BFLOAT16");
        Preconditions.checkArgument(inputDataTypes.get(1) == DataType.FLOAT8
                        && inputDataTypes.get(2) == DataType.FLOAT && inputDataTypes.get(3) == DataType.FLOAT,
                "ModelOptFp8Linear requires FLOAT8 weights and FLOAT scales");
        Preconditions.checkArgument(iArguments.size() == 1 && (iArguments.get(0) == 0 || iArguments.get(0) == 1),
                "ModelOptFp8Linear requires one floatOutput IArg (0 or 1)");
        return Collections.singletonList(iArguments.get(0) == 1 ? DataType.FLOAT : x);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        throw new UnsupportedOperationException("ModelOptFp8Linear is an inference-only quantized op");
    }
}
