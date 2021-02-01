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
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv3DConfig;
import org.nd4j.common.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.List;
import java.util.Map;


@Slf4j
@Getter
@NoArgsConstructor
public class DeConv3DTF extends DynamicCustomOp {

    protected DeConv3DConfig config;

    public DeConv3DTF(@NonNull SameDiff sameDiff, @NonNull SDVariable shape, @NonNull SDVariable weights, @NonNull SDVariable input, @NonNull DeConv3DConfig config) {
        super(sameDiff, new SDVariable[]{shape, weights, input});

        this.config = config;
        addArgs();
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        if(config == null && !iArguments.isEmpty()){
            config = DeConv3DConfig.builder()
                    .kD(iArguments.get(0))
                    .kH(iArguments.get(1))
                    .kW(iArguments.get(2))
                    .sD(iArguments.get(3))
                    .sH(iArguments.get(4))
                    .sW(iArguments.get(5))
                    .pD(iArguments.get(6))
                    .pH(iArguments.get(7))
                    .pW(iArguments.get(8))
                    .dD(iArguments.get(9))
                    .dH(iArguments.get(10))
                    .dW(iArguments.get(11))
                    .isSameMode(iArguments.get(12) == 1)
                    .dataFormat(iArguments.get(13) == 1 ? DeConv3DConfig.NDHWC : DeConv3DConfig.NCDHW)
                    .build();
        }
        return config.toProperties();
    }

    private void addArgs() {
        addIArgument(config.getKD());
        addIArgument(config.getKH());
        addIArgument(config.getKW());
        addIArgument(config.getSD());
        addIArgument(config.getSH());
        addIArgument(config.getSW());
        addIArgument(config.getPD());
        addIArgument(config.getPH());
        addIArgument(config.getPW());
        addIArgument(config.getDD());
        addIArgument(config.getDH());
        addIArgument(config.getDW());
        addIArgument(ArrayUtil.fromBoolean(config.isSameMode()));
        addIArgument(config.getDataFormat().equalsIgnoreCase(DeConv3DConfig.NCDHW) ? 0 : 1);
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
    public Object getValue(Field property) {
        if (config == null) {
            config = DeConv3DConfig.builder().build();
        }

        return config.getValue(property);
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

        val aStrides = nodeDef.getAttrOrThrow("strides");
        val aDilations = nodeDef.getAttrOrDefault("dilations", null);
        val tfStrides = aStrides.getList().getIList();
        val tfDilation = aDilations == null ? null : aDilations.getList().getIList();
        int sD, sH, sW, dD, dH, dW;

        val aPadding = nodeDef.getAttrOrDefault("padding", null);
        String paddingMode = aPadding.getS().toStringUtf8();

        String dataFormat = DeConv3DConfig.NDHWC;
        if (nodeDef.containsAttr("data_format")) {
            val attr = nodeDef.getAttrOrThrow("data_format");
            dataFormat = attr.getS().toStringUtf8().toLowerCase();
        }

        if (dataFormat.equalsIgnoreCase(DeConv3DConfig.NCDHW)) {
            sD = tfStrides.get(2).intValue();
            sH = tfStrides.get(3).intValue();
            sW = tfStrides.get(4).intValue();


            dD = tfDilation == null ? 1 : tfDilation.get(2).intValue();
            dH = tfDilation == null ? 1 : tfDilation.get(3).intValue();
            dW = tfDilation == null ? 1 : tfDilation.get(4).intValue();
        } else {
            sD = tfStrides.get(1).intValue();
            sH = tfStrides.get(2).intValue();
            sW = tfStrides.get(3).intValue();

            dD = tfDilation == null ? 1 : tfDilation.get(1).intValue();
            dH = tfDilation == null ? 1 : tfDilation.get(2).intValue();
            dW = tfDilation == null ? 1 : tfDilation.get(3).intValue();
        }


        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");
        DeConv3DConfig conv3DConfig = DeConv3DConfig.builder()
                .kD(-1)
                .kH(-1)
                .kW(-1)
                .sD(sD)
                .sH(sW)
                .sW(sH)
                .dD(dD)
                .dH(dH)
                .dW(dW)
                .isSameMode(isSameMode)
                .dataFormat(dataFormat.equalsIgnoreCase(DeConv3DConfig.NCDHW) ? DeConv3DConfig.NCDHW : DeConv3DConfig.NDHWC)
                .build();
        this.config = conv3DConfig;

        addArgs();
    }

    @Override
    public String opName() {
        return "deconv3d_tf";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Conv3DBackpropInput", "Conv3DBackpropInputV2"};
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Backprop not yet implemented for " + getClass());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){ //inShape, weights, input
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(2));
    }
}
