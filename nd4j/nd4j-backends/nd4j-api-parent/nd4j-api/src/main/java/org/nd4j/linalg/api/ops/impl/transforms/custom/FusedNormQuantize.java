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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Fused normalization and quantization.
 * <p>
 * Applies normalization (RMSNorm or LayerNorm) followed by quantization
 * (INT8 or FP8) in a single fused kernel:
 * <pre>
 *   normalized = norm(input, weight, bias)
 *   quantized  = quantize(normalized)
 *   scale      = compute_scale(normalized)
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: input [B, H]</li>
 *   <li>1: weight [H] - normalization weight/gamma</li>
 *   <li>2: bias [H] (optional) - normalization bias/beta</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: normType (0=RMSNorm, 1=LayerNorm)</li>
 *   <li>1: quantType (0=INT8, 1=FP8)</li>
 * </ul>
 * <p>
 * Float arguments:
 * <ul>
 *   <li>0: epsilon (default 1e-5)</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: quantized [B, H] INT8</li>
 *   <li>1: scale [B] FLOAT32</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class FusedNormQuantize extends DynamicCustomOp {

    public static final int NORM_RMSNORM = 0;
    public static final int NORM_LAYERNORM = 1;
    public static final int QUANT_INT8 = 0;
    public static final int QUANT_FP8 = 1;

    @Getter private int normType = NORM_RMSNORM;
    @Getter private int quantType = QUANT_INT8;
    @Getter private float epsilon = 1e-5f;

    /**
     * SameDiff constructor with required inputs.
     */
    public FusedNormQuantize(SameDiff sameDiff, SDVariable input, SDVariable weight) {
        super(null, sameDiff, new SDVariable[]{input, weight}, false);
        addIArgument((long) normType, (long) quantType);
        addTArgument((double) epsilon);
    }

    /**
     * SameDiff constructor with bias.
     */
    public FusedNormQuantize(SameDiff sameDiff, SDVariable input, SDVariable weight, SDVariable bias) {
        super(null, sameDiff, new SDVariable[]{input, weight, bias}, false);
        addIArgument((long) normType, (long) quantType);
        addTArgument((double) epsilon);
    }

    /**
     * SameDiff constructor with double epsilon.
     */
    public FusedNormQuantize(SameDiff sameDiff, SDVariable input, SDVariable weight, SDVariable bias,
                             int normType, int quantType, double epsilon) {
        this(sameDiff, input, weight, bias, normType, quantType, (float) epsilon);
    }

    /**
     * Full SameDiff constructor with all options.
     */
    public FusedNormQuantize(SameDiff sameDiff, SDVariable input, SDVariable weight, SDVariable bias,
                             int normType, int quantType, float epsilon) {
        super(null, sameDiff, bias != null ?
                new SDVariable[]{input, weight, bias} :
                new SDVariable[]{input, weight}, false);
        this.normType = normType;
        this.quantType = quantType;
        this.epsilon = epsilon;
        addIArgument((long) normType, (long) quantType);
        addTArgument((double) epsilon);
    }

    /**
     * INDArray constructor.
     */
    public FusedNormQuantize(INDArray input, INDArray weight, INDArray bias,
                             INDArray quantized, INDArray scale,
                             int normType, int quantType, float epsilon) {
        super(null, bias != null ?
                new INDArray[]{input, weight, bias} :
                new INDArray[]{input, weight},
                quantized != null ? new INDArray[]{quantized, scale} : null);
        this.normType = normType;
        this.quantType = quantType;
        this.epsilon = epsilon;
        addIArgument((long) normType, (long) quantType);
        addTArgument((double) epsilon);
    }

    public FusedNormQuantize(SameDiff sd, SDVariable input, SDVariable gamma, double epsilon, int quantType) {
        super(null, sd, new SDVariable[]{input, gamma}, false);
        this.normType = NORM_RMSNORM;
        this.quantType = quantType;
        this.epsilon = (float) epsilon;
        addIArgument((long) this.normType, (long) this.quantType);
        addTArgument(epsilon);
    }

    public FusedNormQuantize(INDArray input, INDArray gamma, double epsilon, int quantType) {
        super(new INDArray[]{input, gamma}, null);
        this.normType = NORM_RMSNORM;
        this.quantType = quantType;
        this.epsilon = (float) epsilon;
        addIArgument((long) this.normType, (long) this.quantType);
        addTArgument(epsilon);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.normType = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.quantType = iArguments.get(1).intValue();
        if (tArguments.size() > 0) this.epsilon = tArguments.get(0).floatValue();
    }

    @Override
    public String opName() {
        return "fused_norm_quantize";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Arrays.asList(DataType.INT8, DataType.FLOAT);
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
