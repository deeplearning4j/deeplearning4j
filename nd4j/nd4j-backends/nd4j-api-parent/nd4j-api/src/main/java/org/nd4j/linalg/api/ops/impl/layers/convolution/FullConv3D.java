/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.IntArrayIntIndexAdpater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.FullConv3DConfig;
import org.nd4j.linalg.util.ArrayUtil;

import java.lang.reflect.Field;
import java.util.*;


/**
 * FullConv3D operation
 */
@Slf4j
public class FullConv3D extends DynamicCustomOp {

    protected FullConv3DConfig config;

    @Builder(builderMethodName = "builder")
    public FullConv3D(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputs, INDArray[] outputs, FullConv3DConfig config) {
        super(null,sameDiff, inputFunctions, false);
        this.config = config;
        if(inputs != null) {
            addInputArgument(inputs);
        }

        if(outputs != null) {
            addOutputArgument(outputs);
        }

        addArgs();
    }


    public FullConv3D() {}

    @Override
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
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
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String,Map<String,AttributeAdapter>> ret = new LinkedHashMap<>();
        Map<String,AttributeAdapter> tfAdapters = new LinkedHashMap<>();

        tfAdapters.put("dT", new IntArrayIntIndexAdpater(1));
        tfAdapters.put("dW",  new IntArrayIntIndexAdpater(2));
        tfAdapters.put("dH",new IntArrayIntIndexAdpater(3));


        tfAdapters.put("pT", new IntArrayIntIndexAdpater(1));
        tfAdapters.put("pW",  new IntArrayIntIndexAdpater(2));
        tfAdapters.put("pH",new IntArrayIntIndexAdpater(3));

        ret.put(tensorflowName(),tfAdapters);

        return ret;
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();



        val strideMapping = PropertyMapping.builder()
                .tfAttrName("strides")
                .onnxAttrName("strides")
                .propertyNames(new String[]{"dT","dW","dH"})
                .build();



        val dilationMapping = PropertyMapping.builder()
                .onnxAttrName("dilations")
                .propertyNames(new String[]{"dD","dH","dW"})
                .tfAttrName("rates")
                .build();



        val sameMode = PropertyMapping.builder()
                .onnxAttrName("auto_pad")
                .propertyNames(new String[]{"isSameMode"})
                .tfAttrName("padding")
                .build();

        val paddingWidthHeight = PropertyMapping.builder()
                .onnxAttrName("padding")
                .propertyNames(new String[]{"pT","pW","pH"})
                .build();

        val dataFormat = PropertyMapping.builder()
                .onnxAttrName("data_format")
                .tfAttrName("data_format")
                .propertyNames(new String[]{"dataFormat"})
                .build();


        val outputPadding = PropertyMapping.builder()
                .propertyNames(new String[]{"aT","aH","aW"})
                .build();


        val biasUsed = PropertyMapping.builder()
                .propertyNames(new String[]{"biasUsed"})
                .build();




        for(val propertyMapping : new PropertyMapping[] {
                strideMapping,
                dilationMapping,
                sameMode,
                paddingWidthHeight,
                dataFormat,
                outputPadding,biasUsed}) {
            for(val keys : propertyMapping.getPropertyNames())
                map.put(keys,propertyMapping);

        }


        ret.put(onnxName(),map);
        ret.put(tensorflowName(),map);
        return ret;
    }

    private void addArgs() {
        addIArgument(new long[]{
                config.getDT(),
                config.getDW(),
                config.getDH(),
                config.getPT(),
                config.getPW(),
                config.getPH(),
                config.getDilationT(),
                config.getDilationW(),
                config.getDilationH(),
                config.getAT(),
                config.getAW(),
                config.getAH(),
                ArrayUtil.fromBoolean(config.isBiasUsed())});


    }




    @Override
    public String opName() {
        return "fullconv3d";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.addAll(f1);
        List<SDVariable> ret = new ArrayList<>();
        FullConv3DDerivative fullConv3DDerivative = FullConv3DDerivative.derivativeBuilder()
                .conv3DConfig(config)
                .sameDiff(sameDiff)
                .inputFunctions(inputs.toArray(new SDVariable[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(fullConv3DDerivative.outputVariables()));
        return ret;
    }

}
