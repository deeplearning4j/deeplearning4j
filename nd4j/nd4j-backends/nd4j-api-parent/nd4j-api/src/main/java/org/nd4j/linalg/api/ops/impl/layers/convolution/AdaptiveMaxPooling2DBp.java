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
 * Adaptive Max Pooling 2D - Backward Pass
 *
 * @author Eclipse Deeplearning4j
 */
@NoArgsConstructor
@Getter
public class AdaptiveMaxPooling2DBp extends DynamicCustomOp {

    private int outputHeight;
    private int outputWidth;
    private boolean isNCHW = true;

    /**
     * Create backward pass for adaptive max pooling 2D.
     *
     * @param input Original input tensor
     * @param gradOut Gradient from upstream
     * @param outputHeight Target output height (from forward pass)
     * @param outputWidth Target output width (from forward pass)
     * @param isNCHW True for NCHW format, false for NHWC
     */
    public AdaptiveMaxPooling2DBp(@NonNull INDArray input, @NonNull INDArray gradOut,
                                  int outputHeight, int outputWidth, boolean isNCHW) {
        super(new INDArray[]{input, gradOut}, null);
        this.outputHeight = outputHeight;
        this.outputWidth = outputWidth;
        this.isNCHW = isNCHW;
        addArgs();
    }

    /**
     * Create backward pass for adaptive max pooling 2D for SameDiff.
     *
     * @param sd SameDiff instance
     * @param input Original input tensor
     * @param gradOut Gradient from upstream
     * @param outputHeight Target output height
     * @param outputWidth Target output width
     * @param isNCHW True for NCHW format
     */
    public AdaptiveMaxPooling2DBp(@NonNull SameDiff sd, @NonNull SDVariable input,
                                  @NonNull SDVariable gradOut, int outputHeight,
                                  int outputWidth, boolean isNCHW) {
        super(null, sd, new SDVariable[]{input, gradOut}, false);
        this.outputHeight = outputHeight;
        this.outputWidth = outputWidth;
        this.isNCHW = isNCHW;
        addArgs();
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
        return "adaptive_maxpool2d_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2,
                "Expected 2 input data types for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
