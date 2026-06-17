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

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeformableConv2DConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Deformable Convolution 2D
 *
 * Implements deformable convolution where learned offsets are added to
 * the regular sampling grid, allowing the convolution to adapt to
 * geometric transformations in the input.
 *
 * Reference: "Deformable Convolutional Networks" (Dai et al., 2017)
 *            "Deformable ConvNets v2" (Zhu et al., 2019)
 *
 * @author Eclipse Deeplearning4j
 */
@NoArgsConstructor
@Getter
public class DeformableConv2D extends DynamicCustomOp {

    protected DeformableConv2DConfig config;

    /**
     * Create a deformable convolution 2D operation.
     *
     * @param input Input tensor [batch, channels, height, width] (NCHW) or NHWC
     * @param weights Convolution weights [out_channels, in_channels/groups, kH, kW]
     * @param offset Learned offsets [batch, 2*kH*kW*offset_groups, out_h, out_w]
     * @param config Configuration for the deformable convolution
     */
    public DeformableConv2D(@NonNull INDArray input, @NonNull INDArray weights,
                           @NonNull INDArray offset, @NonNull DeformableConv2DConfig config) {
        super(new INDArray[]{input, weights, offset}, null);
        this.config = config;
        addArgs();
    }

    /**
     * Create a deformable convolution 2D operation with bias.
     *
     * @param input Input tensor
     * @param weights Convolution weights
     * @param offset Learned offsets
     * @param bias Bias [out_channels]
     * @param config Configuration
     */
    public DeformableConv2D(@NonNull INDArray input, @NonNull INDArray weights,
                           @NonNull INDArray offset, INDArray bias,
                           @NonNull DeformableConv2DConfig config) {
        super(bias != null ?
                new INDArray[]{input, weights, offset, bias} :
                new INDArray[]{input, weights, offset}, null);
        this.config = config;
        addArgs();
    }

    /**
     * Create a deformable convolution 2D operation with bias and mask (v2).
     *
     * @param input Input tensor
     * @param weights Convolution weights
     * @param offset Learned offsets
     * @param bias Bias (optional, can be null)
     * @param mask Modulation mask for v2 (optional, can be null)
     * @param config Configuration
     */
    public DeformableConv2D(@NonNull INDArray input, @NonNull INDArray weights,
                           @NonNull INDArray offset, INDArray bias, INDArray mask,
                           @NonNull DeformableConv2DConfig config) {
        super(buildInputs(input, weights, offset, bias, mask), null);
        this.config = config;
        addArgs();
    }

    /**
     * Create a deformable convolution 2D for SameDiff.
     *
     * @param sd SameDiff instance
     * @param input Input tensor
     * @param weights Convolution weights
     * @param offset Learned offsets
     * @param config Configuration
     */
    @Builder(builderMethodName = "sameDiffBuilder")
    public DeformableConv2D(@NonNull SameDiff sd, @NonNull SDVariable input,
                           @NonNull SDVariable weights, @NonNull SDVariable offset,
                           @NonNull DeformableConv2DConfig config) {
        super(null, sd, new SDVariable[]{input, weights, offset}, false);
        this.config = config;
        addArgs();
    }

    /**
     * Create a deformable convolution 2D for SameDiff with all options.
     *
     * @param sd SameDiff instance
     * @param input Input tensor
     * @param weights Convolution weights
     * @param offset Learned offsets
     * @param bias Bias (optional)
     * @param mask Modulation mask for v2 (optional)
     * @param config Configuration
     */
    public DeformableConv2D(@NonNull SameDiff sd, @NonNull SDVariable input,
                           @NonNull SDVariable weights, @NonNull SDVariable offset,
                           SDVariable bias, SDVariable mask,
                           @NonNull DeformableConv2DConfig config) {
        super(null, sd, buildSdInputs(input, weights, offset, bias, mask), false);
        this.config = config;
        addArgs();
    }

    /**
     * Create a deformable convolution 2D for SameDiff with individual parameters (for codegen).
     */
    public DeformableConv2D(@NonNull SameDiff sd, @NonNull SDVariable input,
                           @NonNull SDVariable weights, @NonNull SDVariable offset,
                           SDVariable bias, SDVariable mask,
                           int kH, int kW, int sH, int sW, int pH, int pW,
                           int dH, int dW, int groups, int deformableGroups, boolean isNCHW) {
        super(null, sd, buildSdInputs(input, weights, offset, bias, mask), false);
        this.config = DeformableConv2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(sH).sW(sW)
                .pH(pH).pW(pW)
                .dH(dH).dW(dW)
                .groups(groups)
                .deformableGroups(deformableGroups)
                .isNCHW(isNCHW)
                .build();
        addArgs();
    }

    /**
     * Create a deformable convolution 2D for INDArray with individual parameters (for codegen).
     */
    public DeformableConv2D(@NonNull INDArray input, @NonNull INDArray weights,
                           @NonNull INDArray offset, INDArray bias, INDArray mask,
                           int kH, int kW, int sH, int sW, int pH, int pW,
                           int dH, int dW, int groups, int deformableGroups, boolean isNCHW) {
        super(buildInputs(input, weights, offset, bias, mask), null);
        this.config = DeformableConv2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(sH).sW(sW)
                .pH(pH).pW(pW)
                .dH(dH).dW(dW)
                .groups(groups)
                .deformableGroups(deformableGroups)
                .isNCHW(isNCHW)
                .build();
        addArgs();
    }

    private static INDArray[] buildInputs(INDArray input, INDArray weights, INDArray offset,
                                          INDArray bias, INDArray mask) {
        List<INDArray> inputs = new ArrayList<>();
        inputs.add(input);
        inputs.add(weights);
        inputs.add(offset);
        if (bias != null) {
            inputs.add(bias);
        }
        if (mask != null) {
            inputs.add(mask);
        }
        return inputs.toArray(new INDArray[0]);
    }

    private static SDVariable[] buildSdInputs(SDVariable input, SDVariable weights, SDVariable offset,
                                              SDVariable bias, SDVariable mask) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(input);
        inputs.add(weights);
        inputs.add(offset);
        if (bias != null) {
            inputs.add(bias);
        }
        if (mask != null) {
            inputs.add(mask);
        }
        return inputs.toArray(new SDVariable[0]);
    }

    private void addArgs() {
        addIArgument(
                config.getKH(),
                config.getKW(),
                config.getSH(),
                config.getSW(),
                config.getPH(),
                config.getPW(),
                config.getDH(),
                config.getDW(),
                config.getGroups(),
                config.getDeformableGroups(),
                config.isNCHW() ? 1 : 0
        );
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() >= 11) {
            this.config = DeformableConv2DConfig.builder()
                    .kH(iArguments.get(0).intValue())
                    .kW(iArguments.get(1).intValue())
                    .sH(iArguments.get(2).intValue())
                    .sW(iArguments.get(3).intValue())
                    .pH(iArguments.get(4).intValue())
                    .pW(iArguments.get(5).intValue())
                    .dH(iArguments.get(6).intValue())
                    .dW(iArguments.get(7).intValue())
                    .groups(iArguments.get(8).intValue())
                    .deformableGroups(iArguments.get(9).intValue())
                    .isNCHW(iArguments.get(10) != 0)
                    .build();
        }
    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "config";
    }

    @Override
    public String opName() {
        return "deformable_conv2d";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 3,
                "Expected at least 3 input data types for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public String onnxName() {
        return "DeformConv";
    }
}
