/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.common.util.ArrayUtil;

import java.lang.reflect.Field;
import java.util.*;


/**
 * Backpropagation for Depthwise Conv2D operation
 */
@Slf4j
@Getter
@NoArgsConstructor
public class DepthwiseConv2DBp extends DynamicCustomOp {

    protected Conv2DConfig config;


    public DepthwiseConv2DBp(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable weights, SDVariable bias, @NonNull SDVariable gradO, @NonNull Conv2DConfig config){
        super(sameDiff, wrapFilterNull(input, weights, bias, gradO));
        this.config = config;
        addArgs();

    }

    public DepthwiseConv2DBp(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable weights, @NonNull SDVariable gradO, @NonNull Conv2DConfig config){
        super(sameDiff, wrapFilterNull(input, weights, gradO));
        this.config = config;
        addArgs();

    }


    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    protected void addArgs() {
        addIArgument(config.getKH(),
                config.getKW(),
                config.getSH(),
                config.getSW(),
                config.getPH(),
                config.getPW(),
                config.getDH(),
                config.getDW(),
                ArrayUtil.fromBoolean(config.isSameMode()),
                config.getDataFormat().equalsIgnoreCase(Conv2DConfig.NCHW) ? 0 : 1);

    }

    @Override
    public Object getValue(Field property) {
        if (config == null) {
            config = Conv2DConfig.builder().build();
        }

        try {
            val t = config.getValue(property);
            return t;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        if (config == null && !iArguments.isEmpty()) {
            config = Conv2DConfig.builder()
                    .kH(iArguments.get(0))
                    .kW(iArguments.get(1))
                    .sH(iArguments.get(2))
                    .sW(iArguments.get(3))
                    .pH(iArguments.get(4))
                    .pW(iArguments.get(5))
                    .dH(iArguments.get(6))
                    .dW(iArguments.get(7))
                    .isSameMode(iArguments.get(8) == 1)
                    .dataFormat(iArguments.get(9) == 1 ? Conv2DConfig.NHWC : Conv2DConfig.NCHW)
                    .build();
        }
        return config.toProperties();
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
        return "depthwise_conv2d_bp";
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        int n = args().length;
        List<DataType> list = new ArrayList<DataType>();
        for(int i=0;i<n-1;i++){list.add(inputDataTypes.get(0));}
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return list;
    }

    @Override
    public int getNumOutputs(){
        return args().length == 4 ? 3 : 2;}
}
