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
 * ModelOpt NVFP4 inference linear: X[...,K] times packed W[N,K/2] transposed.
 * X is FLOAT, HALF or BFLOAT16; W is UBYTE with the even K element in the low
 * nibble. Block scales are FLOAT8 E4M3 [N,K/16], and the global scale is a
 * positive finite FLOAT rank-zero scalar. K must be divisible by 16.
 * Dequantization multiplies block/global scales in FP32, then E2M1 in FP32,
 * then rounds the weight to X's dtype before FP32 accumulation.
 * The output is [...,N], with X's dtype unless floatOutput requests FLOAT.
 * Native validation owns shapes, scale values and output alias rejection.
 */
public class ModelOptNvfp4Linear extends DynamicCustomOp {
    public ModelOptNvfp4Linear() {
        // Serialization restores inputs and IArgs, which are the source of truth.
    }

    public ModelOptNvfp4Linear(SameDiff sd, SDVariable x, SDVariable w,
                              SDVariable scale, SDVariable secondScale, boolean floatOutput) {
        super(null, sd, new SDVariable[]{x, w, scale, secondScale});
        addIArgument(floatOutput ? 1 : 0);
    }

    public ModelOptNvfp4Linear(INDArray x, INDArray w, INDArray scale,
                              INDArray secondScale, boolean floatOutput) {
        super(null, new INDArray[]{x, w, scale, secondScale}, null);
        addIArgument(floatOutput ? 1 : 0);
    }

    @Override
    public String opName() {
        return "modelopt_nvfp4_linear";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkArgument(inputDataTypes != null && inputDataTypes.size() == 4,
                "ModelOptNvfp4Linear requires four input dtypes");
        DataType x = inputDataTypes.get(0);
        Preconditions.checkArgument(x == DataType.FLOAT || x == DataType.HALF || x == DataType.BFLOAT16,
                "ModelOptNvfp4Linear activation must be FLOAT, HALF or BFLOAT16");
        Preconditions.checkArgument(inputDataTypes.get(1) == DataType.UBYTE
                        && inputDataTypes.get(2) == DataType.FLOAT8 && inputDataTypes.get(3) == DataType.FLOAT,
                "ModelOptNvfp4Linear requires UBYTE weights, FLOAT8 blocks and FLOAT global scale");
        Preconditions.checkArgument(iArguments.size() == 1 && (iArguments.get(0) == 0 || iArguments.get(0) == 1),
                "ModelOptNvfp4Linear requires one floatOutput IArg (0 or 1)");
        return Collections.singletonList(iArguments.get(0) == 1 ? DataType.FLOAT : x);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        throw new UnsupportedOperationException("ModelOptNvfp4Linear is an inference-only quantized op");
    }
}
