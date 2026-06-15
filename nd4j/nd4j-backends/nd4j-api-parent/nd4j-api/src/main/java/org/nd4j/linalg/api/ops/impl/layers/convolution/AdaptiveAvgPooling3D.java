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
 * Adaptive Average Pooling 3D
 *
 * Automatically computes kernel size and stride to produce output
 * of the specified spatial dimensions.
 *
 * @author Eclipse Deeplearning4j
 */
@NoArgsConstructor
@Getter
public class AdaptiveAvgPooling3D extends DynamicCustomOp {

    private int outputDepth;
    private int outputHeight;
    private int outputWidth;
    private boolean isNCDHW = true;

    /**
     * Create an adaptive average pooling 3D operation.
     *
     * @param input Input tensor [batch, channels, depth, height, width] (NCDHW) or [batch, depth, height, width, channels] (NDHWC)
     * @param outputDepth Target output depth
     * @param outputHeight Target output height
     * @param outputWidth Target output width
     * @param isNCDHW True for NCDHW format, false for NDHWC
     */
    public AdaptiveAvgPooling3D(@NonNull INDArray input, int outputDepth, int outputHeight,
                                int outputWidth, boolean isNCDHW) {
        super(new INDArray[]{input}, null);
        this.outputDepth = outputDepth;
        this.outputHeight = outputHeight;
        this.outputWidth = outputWidth;
        this.isNCDHW = isNCDHW;
        addArgs();
    }

    /**
     * Create an adaptive average pooling 3D operation with default NCDHW format.
     *
     * @param input Input tensor [batch, channels, depth, height, width]
     * @param outputDepth Target output depth
     * @param outputHeight Target output height
     * @param outputWidth Target output width
     */
    public AdaptiveAvgPooling3D(@NonNull INDArray input, int outputDepth, int outputHeight, int outputWidth) {
        this(input, outputDepth, outputHeight, outputWidth, true);
    }

    /**
     * Create an adaptive average pooling 3D for SameDiff.
     *
     * @param sd SameDiff instance
     * @param input Input tensor
     * @param outputDepth Target output depth
     * @param outputHeight Target output height
     * @param outputWidth Target output width
     * @param isNCDHW True for NCDHW format, false for NDHWC
     */
    public AdaptiveAvgPooling3D(@NonNull SameDiff sd, @NonNull SDVariable input,
                                int outputDepth, int outputHeight, int outputWidth,
                                boolean isNCDHW) {
        super(null, sd, new SDVariable[]{input}, false);
        this.outputDepth = outputDepth;
        this.outputHeight = outputHeight;
        this.outputWidth = outputWidth;
        this.isNCDHW = isNCDHW;
        addArgs();
    }

    /**
     * Create an adaptive average pooling 3D for SameDiff with default NCDHW format.
     *
     * @param sd SameDiff instance
     * @param input Input tensor
     * @param outputDepth Target output depth
     * @param outputHeight Target output height
     * @param outputWidth Target output width
     */
    public AdaptiveAvgPooling3D(@NonNull SameDiff sd, @NonNull SDVariable input,
                                int outputDepth, int outputHeight, int outputWidth) {
        this(sd, input, outputDepth, outputHeight, outputWidth, true);
    }

    private void addArgs() {
        addIArgument(outputDepth, outputHeight, outputWidth, isNCDHW ? 1 : 0);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.outputDepth = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.outputHeight = iArguments.get(1).intValue();
        }
        if (iArguments.size() > 2) {
            this.outputWidth = iArguments.get(2).intValue();
        }
        if (iArguments.size() > 3) {
            this.isNCDHW = iArguments.get(3) != 0;
        }
    }

    @Override
    public String opName() {
        return "adaptive_avgpool3d";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1,
                "Expected 1 input data type for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public String onnxName() {
        return "AdaptiveAveragePool3D";
    }
}
