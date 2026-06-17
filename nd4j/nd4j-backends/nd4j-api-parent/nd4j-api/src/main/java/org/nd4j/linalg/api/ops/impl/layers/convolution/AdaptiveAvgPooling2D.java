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

package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Adaptive Average Pooling 2D
 *
 * Automatically computes kernel size and stride to produce output
 * of the specified spatial dimensions. Common in vision models.
 *
 * @author Eclipse Deeplearning4j
 */
@NoArgsConstructor
@Getter
public class AdaptiveAvgPooling2D extends DynamicCustomOp {

    private int outputHeight;
    private int outputWidth;
    private boolean isNCHW = true;

    /**
     * Create an adaptive average pooling 2D operation.
     *
     * @param input Input tensor [batch, channels, height, width] (NCHW) or [batch, height, width, channels] (NHWC)
     * @param outputHeight Target output height
     * @param outputWidth Target output width
     * @param isNCHW True for NCHW format, false for NHWC
     */
    public AdaptiveAvgPooling2D(@NonNull INDArray input, int outputHeight, int outputWidth, boolean isNCHW) {
        super(new INDArray[]{input}, null);
        this.outputHeight = outputHeight;
        this.outputWidth = outputWidth;
        this.isNCHW = isNCHW;
        addArgs();
    }

    /**
     * Create an adaptive average pooling 2D operation with default NCHW format.
     *
     * @param input Input tensor [batch, channels, height, width]
     * @param outputHeight Target output height
     * @param outputWidth Target output width
     */
    public AdaptiveAvgPooling2D(@NonNull INDArray input, int outputHeight, int outputWidth) {
        this(input, outputHeight, outputWidth, true);
    }

    /**
     * Create an adaptive average pooling 2D for SameDiff.
     *
     * @param sd SameDiff instance
     * @param input Input tensor
     * @param outputHeight Target output height
     * @param outputWidth Target output width
     * @param isNCHW True for NCHW format, false for NHWC
     */
    public AdaptiveAvgPooling2D(@NonNull SameDiff sd, @NonNull SDVariable input,
                                int outputHeight, int outputWidth, boolean isNCHW) {
        super(null, sd, new SDVariable[]{input}, false);
        this.outputHeight = outputHeight;
        this.outputWidth = outputWidth;
        this.isNCHW = isNCHW;
        addArgs();
    }

    /**
     * Create an adaptive average pooling 2D for SameDiff with default NCHW format.
     *
     * @param sd SameDiff instance
     * @param input Input tensor
     * @param outputHeight Target output height
     * @param outputWidth Target output width
     */
    public AdaptiveAvgPooling2D(@NonNull SameDiff sd, @NonNull SDVariable input,
                                int outputHeight, int outputWidth) {
        this(sd, input, outputHeight, outputWidth, true);
    }

    private void addArgs() {
        addIArgument(outputHeight, outputWidth, isNCHW ? 1 : 0);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.outputHeight = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.outputWidth = iArguments.get(1).intValue();
        }
        if (iArguments.size() > 2) {
            this.isNCHW = iArguments.get(2) != 0;
        }
    }

    @Override
    public String opName() {
        return "adaptive_avgpool2d";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        return Collections.singletonList(
                new AdaptiveAvgPooling2DBp(sameDiff, arg(0), gradient.get(0),
                        outputHeight, outputWidth, isNCHW).outputVariable());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1,
                "Expected 1 input data type for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public String onnxName() {
        return "AdaptiveAveragePool";
    }
}
