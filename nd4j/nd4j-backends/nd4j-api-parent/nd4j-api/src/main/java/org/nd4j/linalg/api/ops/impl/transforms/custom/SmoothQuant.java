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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * SmoothQuant W8A8 Quantized Matrix Multiplication.
 * <p>
 * Implements the SmoothQuant algorithm which migrates quantization difficulty
 * from activations to weights by applying a per-channel smoothing factor:
 * <pre>
 *   Y = deq( quant(X * diag(s)^{-1}) @ quant(diag(s) * W) )
 * </pre>
 * where {@code s} is a per-channel smoothing factor that balances the dynamic
 * ranges of activations and weights.
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: X (FLOAT) - input activations [batch, in_features]</li>
 *   <li>1: W_quantized (INT8) - pre-quantized smoothed weights [out_features, in_features]</li>
 *   <li>2: smooth_scale (FLOAT) - per-channel smoothing factors [in_features]</li>
 *   <li>3: act_scale (FLOAT) - activation quantization scale (scalar or per-channel)</li>
 *   <li>4: weight_scale (FLOAT) - weight quantization scale (per-channel) [out_features]</li>
 *   <li>5: bias (optional) [out_features]</li>
 * </ul>
 * <p>
 * Output: Y (FLOAT) - dequantized output [batch, out_features]
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: transpose_weight (0=no, 1=yes, default: 0)</li>
 * </ul>
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class SmoothQuant extends DynamicCustomOp {

    @Getter private boolean transposeWeight = false;

    /**
     * SameDiff constructor with required inputs (no bias).
     *
     * @param sameDiff    the SameDiff instance
     * @param x           input activations [batch, in_features]
     * @param wQuantized  pre-quantized smoothed weights [out_features, in_features]
     * @param smoothScale per-channel smoothing factors [in_features]
     * @param actScale    activation quantization scale
     * @param weightScale weight quantization scale [out_features]
     */
    public SmoothQuant(SameDiff sameDiff, SDVariable x, SDVariable wQuantized,
                       SDVariable smoothScale, SDVariable actScale, SDVariable weightScale) {
        super(null, sameDiff, new SDVariable[]{x, wQuantized, smoothScale, actScale, weightScale}, false);
        addIArgument(transposeWeight ? 1L : 0L);
    }

    /**
     * SameDiff constructor with bias.
     */
    public SmoothQuant(SameDiff sameDiff, SDVariable x, SDVariable wQuantized,
                       SDVariable smoothScale, SDVariable actScale, SDVariable weightScale,
                       SDVariable bias) {
        super(null, sameDiff,
                new SDVariable[]{x, wQuantized, smoothScale, actScale, weightScale, bias}, false);
        addIArgument(transposeWeight ? 1L : 0L);
    }

    /**
     * SameDiff constructor with all options.
     */
    public SmoothQuant(SameDiff sameDiff, SDVariable x, SDVariable wQuantized,
                       SDVariable smoothScale, SDVariable actScale, SDVariable weightScale,
                       SDVariable bias, boolean transposeWeight) {
        super(null, sameDiff,
                bias != null ?
                        new SDVariable[]{x, wQuantized, smoothScale, actScale, weightScale, bias} :
                        new SDVariable[]{x, wQuantized, smoothScale, actScale, weightScale},
                false);
        this.transposeWeight = transposeWeight;
        addIArgument(transposeWeight ? 1L : 0L);
    }

    /**
     * INDArray constructor.
     */
    public SmoothQuant(INDArray x, INDArray wQuantized, INDArray smoothScale,
                       INDArray actScale, INDArray weightScale, INDArray bias,
                       INDArray output, boolean transposeWeight) {
        super(null,
                bias != null ?
                        new INDArray[]{x, wQuantized, smoothScale, actScale, weightScale, bias} :
                        new INDArray[]{x, wQuantized, smoothScale, actScale, weightScale},
                output != null ? new INDArray[]{output} : null);
        this.transposeWeight = transposeWeight;
        addIArgument(transposeWeight ? 1L : 0L);
    }

    public SmoothQuant(SameDiff sd, SDVariable input, SDVariable smoothScale) {
        super(null, sd, new SDVariable[]{input, smoothScale}, false);
        addIArgument(transposeWeight ? 1L : 0L);
    }

    public SmoothQuant(INDArray input, INDArray smoothScale) {
        super(new INDArray[]{input, smoothScale}, null);
        addIArgument(transposeWeight ? 1L : 0L);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.transposeWeight = iArguments.get(0) != 0;
    }

    @Override
    public String opName() {
        return "smooth_quant";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 5,
                "Expected at least 5 input data types for smooth_quant, got %s", inputDataTypes);
        // Output is always FLOAT (dequantized result)
        return Collections.singletonList(DataType.FLOAT);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
