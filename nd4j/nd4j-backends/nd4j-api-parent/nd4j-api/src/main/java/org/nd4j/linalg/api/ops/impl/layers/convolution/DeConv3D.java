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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv3DConfig;
import org.nd4j.common.util.ArrayUtil;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.List;
import java.util.Map;


@Slf4j
@Getter
@NoArgsConstructor
public class DeConv3D extends DynamicCustomOp {

    protected DeConv3DConfig config;

    public DeConv3D(SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable weights, SDVariable bias, @NonNull DeConv3DConfig config) {
        super(sameDiff, toArr(input, weights, bias));
        this.config = config;
        addArgs();
    }

    public DeConv3D(SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable weights, @NonNull DeConv3DConfig config) {
        super(sameDiff, toArr(input, weights, null));
        this.config = config;
        addArgs();
    }

    public DeConv3D(INDArray[] inputs, INDArray[] outputs, DeConv3DConfig config){
        super(inputs, outputs);

        this.config = config;
        addArgs();
    }

    public DeConv3D(@NonNull INDArray input, @NonNull INDArray weights, INDArray bias, INDArray output, @NonNull DeConv3DConfig config){
        this(wrapFilterNull(input, weights, bias), wrapOrNull(output), config);
    }

    public DeConv3D(INDArray input, INDArray weights, INDArray bias, DeConv3DConfig config) {
        this(input, weights, bias, null, config);
    }

    private static SDVariable[] toArr(SDVariable input, SDVariable weights, SDVariable bias){
        if(bias != null){
            return new SDVariable[]{input, weights, bias};
        } else {
            return new SDVariable[]{input, weights};
        }
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        if(config == null && !iArguments.isEmpty()) {
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



    @Override
    public void configureFromArguments() {
        if(config == null  && iArguments.size() >= 14) {
            DeConv3DConfig.DeConv3DConfigBuilder builder = DeConv3DConfig.builder();
            builder.kD(getIArgument(0));
            builder.kH(getIArgument(1));
            builder.kW(getIArgument(2));
            builder.sD(getIArgument(3));
            builder.sH(getIArgument(4));
            builder.sW(getIArgument(5));
            builder.pD(getIArgument(6));
            builder.pH(getIArgument(7));
            builder.pW(getIArgument(8));
            builder.dD(getIArgument(9));
            builder.dH(getIArgument(10));
            builder.dW(getIArgument(11));
            builder.isSameMode(getIArgument(12) > 0);
            builder.dataFormat(getIArgument(13) > 0 ? "NCDHW" : "NCHWDC");
            this.config = builder.build();
        }

    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(config == null) {
            DeConv3DConfig.DeConv3DConfigBuilder builder = DeConv3DConfig.builder();
            Long dD = getLongValueFromProperty("dD",properties);
            if(dD != null)
                builder.dD(dD);
            Long dH = getLongValueFromProperty("dH",properties);
            if(dH != null)
                builder.dH(dH);
            Long sW = getLongValueFromProperty("sW",properties);
            if(sW != null)
                builder.sW(sW);
            Long pW = getLongValueFromProperty("pW",properties);
            if(pW != null)
                builder.pW(pW);

            Long sD = getLongValueFromProperty("sD",properties);
            if(sD != null)
                builder.sD(sD);

            Long dW = getLongValueFromProperty("dW",properties);
            if(dW != null)
                builder.dW(dW);

            Long pD = getLongValueFromProperty("pD",properties);
            if(pD != null)
                builder.pD(pD);

            Long sH = getLongValueFromProperty("sH",properties);
            if(sH != null)
                builder.sH(sH);

            Long pH = getLongValueFromProperty("pH",properties);
            if(pH != null)
                builder.pH(pH);

            Long kD = getLongValueFromProperty("kD",properties);
            if(kD != null)
                builder.kD(kD);

            Long kW = getLongValueFromProperty("kW",properties);
            if(kW != null)
                builder.kW(kW);

            Long kH = getLongValueFromProperty("kH",properties);
            if(kH != null)
                builder.kH(kH);

            Boolean isSameMode = getBooleanFromProperty("isSameMode",properties);
            if(isSameMode != null)
                builder.isSameMode(isSameMode);

            if(properties.containsKey("dataFormat")) {
                builder.dataFormat(properties.get("dataFormat").toString());
            }

            this.config = builder.build();

        }

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
    public String opName() {
        return "deconv3d";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable bias = args().length > 2 ? arg(2) : null;
        return new DeConv3DDerivative(sameDiff, arg(0), arg(1), bias, f1.get(0), config).outputs();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}