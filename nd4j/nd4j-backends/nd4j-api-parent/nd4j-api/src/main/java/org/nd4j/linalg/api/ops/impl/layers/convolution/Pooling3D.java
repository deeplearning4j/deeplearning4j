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
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;


/**
 * Pooling3D operation
 */
@Slf4j
public class Pooling3D extends DynamicCustomOp {
    protected Pooling3DConfig config;

    public enum Pooling3DType {
        MAX, AVG, PNORM,
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    public Pooling3D() {}

    @Builder(builderMethodName = "builder")
    public Pooling3D(SameDiff sameDiff, SDVariable[] inputs,INDArray[] inputArrays, INDArray[] outputs,boolean inPlace,
                     Pooling3DConfig pooling3DConfig, Pooling3DType type) {
        super(null,sameDiff, inputs, inPlace);
        Preconditions.checkState(pooling3DConfig.getDD() > 0 && pooling3DConfig.getDH() > 0 && pooling3DConfig.getDW() > 0,
                "Dilation values must all be > 0: got dD/H/W = %s/%s/%s", pooling3DConfig.getDD(), pooling3DConfig.getDH(), pooling3DConfig.getDW());

        if(type != null) {
            pooling3DConfig.setType(type);
        }

        this.config = pooling3DConfig;
        this.sameDiff = sameDiff;

        if(inputArrays != null) {
            addInputArgument(inputArrays);
        }
        if(outputs != null) {
            addOutputArgument(outputs);
        }
        addArgs();
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
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
    }

    protected void addArgs() {
        if(this.iArguments == null)
            this.iArguments = new ArrayList<>();
        addIArgument(config.getKD());
        addIArgument(config.getKW());
        addIArgument(config.getKH());
        addIArgument(config.getSD());
        addIArgument(config.getSW());
        addIArgument(config.getSH());
        addIArgument(config.getPD());
        addIArgument(config.getPW());
        addIArgument(config.getPH());
        addIArgument(config.getDD());
        addIArgument(config.getDW());
        addIArgument(config.getDH());
        addIArgument(config.isSameMode() ? 1 : 0);       //Ceiling mode == same mode
        addIArgument(0);                                    //0 == "exclude padding from average count"
        addIArgument(config.isNCDHW() ? 0 : 1);

    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool3dnew";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Pooling3DDerivative pooling3DDerivative = Pooling3DDerivative.derivativeBuilder()
                .inPlace(inPlace)
                .sameDiff(sameDiff)
                .inputs(inputs.toArray(new SDVariable[inputs.size()]))
                .pooling3DConfig(config)
                .build();
        ret.addAll(Arrays.asList(pooling3DDerivative.outputVariables()));

        return ret;
    }

    public String getPoolingPrefix() {
        if (config == null)
            return "pooling3d";

        switch(config.getType()) {
            case AVG:return "avg";
            case MAX: return "max";
            default: throw new IllegalStateException("No pooling type found.");
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val aStrides = nodeDef.getAttrOrThrow("strides");
        List<Long> tfStrides = aStrides.getList().getIList();
        val aKernels = nodeDef.getAttrOrThrow("ksize");
        List<Long> tfKernels = aKernels.getList().getIList();
        val aPadding = nodeDef.getAttrOrThrow("padding");
        List<Long> tfPadding = aPadding.getList().getIList();

        String paddingMode = aPadding.getS().toStringUtf8().replaceAll("\"", "");

        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

        String data_format = "ndhwc";
        if (nodeDef.containsAttr("data_format")) {
            val attr = nodeDef.getAttrOrThrow("data_format");

            data_format = attr.getS().toStringUtf8().toLowerCase();
        }

        //Order: depth, height, width
        //TF doesn't have dilation, it seems?
        int[] strides = new int[3];
        int[] padding = new int[3];
        int[] kernel = new int[3];
        for( int i=0; i<3; i++ ) {
            //TF values here have 5 values: minibatch and Channels at positions 0 and 4, which are almost always 1
            strides[i] = tfStrides.get(i+1).intValue();
            if(tfPadding != null && tfPadding.size() > 0) {
                //Empty for SAME mode
                padding[i] = tfPadding.get(i + 1).intValue();
            }
            kernel[i] = tfKernels.get(i+1).intValue();
        }

        Pooling3DType type;
        String name = nodeDef.getOp().toLowerCase();
        if(name.startsWith("max")){
            type = Pooling3DType.MAX;
        } else if(name.startsWith("av")){
            type = Pooling3DType.AVG;
        } else {
            throw new IllegalStateException("Unknown or not supported pooling type: " + name);
        }

        Pooling3DConfig conf = Pooling3DConfig.builder()
                .sD(strides[0]).sH(strides[1]).sW(strides[2])
                .pD(padding[0]).pH(padding[1]).pW(padding[2])
                .kD(kernel[0]).kH(kernel[1]).kW(kernel[2])
                .type(type)
                .isSameMode(isSameMode)
                .isNCDHW(data_format.equalsIgnoreCase("ncdhw"))
                .build();
        this.config = conf;
        addArgs();
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for op " + opName());
    }

    @Override
    public String tensorflowName() {
      throw new NoOpNameFoundException("No op opName found for op " + opName());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected 1 input data type, got %s", inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }

}
