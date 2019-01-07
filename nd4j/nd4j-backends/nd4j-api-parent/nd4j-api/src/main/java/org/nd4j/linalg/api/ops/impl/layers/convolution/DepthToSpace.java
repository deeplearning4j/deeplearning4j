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

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


/**
 * Inverse operation to SpaceToDepth. This operation takes 4D array in, in either NCHW or NHWC format,
 * and moves data from  channels (C) to spatial dimensions (HW) for given blockSize.
 * <p>
 * Example:
 * blockSize = 4
 * dataFormat = "NCHW"
 * input shape =  [128, 4, 4, 48]
 * output shape = [128, 4*4, 4*4, 48/4/4]
 *
 * @author raver119@gmail.com, Max Pumperla
 */
public class DepthToSpace extends DynamicCustomOp {
    private String dataFormat = "NHWC";
    private int blockSize;

    public DepthToSpace() {
    }

    public DepthToSpace(SameDiff sameDiff, SDVariable[] args, int blockSize, String dataFormat) {
        super(null, sameDiff, args, false);
        this.blockSize = blockSize;
        this.dataFormat = dataFormat;
        boolean isNHWC = dataFormat.equals("NHWC");
        addIArgument(blockSize, isNHWC ? 1 : 0);
    }

    public DepthToSpace(INDArray in, INDArray out, int blockSize, String dataFormat) {
        super(null, in, out, null, null);
        this.blockSize = blockSize;
        this.dataFormat = dataFormat;
        boolean isNHWC = dataFormat.equals("NHWC");
        addIArgument(blockSize, isNHWC ? 1 : 0);
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // Gradient to DepthToSpace is just SpaceToDepth of same block size and data format.
        SDVariable gradient = i_v.get(0);
        SDVariable ret = sameDiff.spaceToDepth(gradient, blockSize, dataFormat);
        return Arrays.asList(ret);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        boolean isNHWC = dataFormat.equals("NHWC");
        addIArgument(blockSize, isNHWC ? 1 : 0);
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> attrs = new LinkedHashMap<>();

        val blockSize = PropertyMapping.builder()
                .tfAttrName("block_size")
                .propertyNames(new String[]{"blockSize"})
                .build();
        attrs.put("blockSize", blockSize);

        val dataFormatMapping = PropertyMapping.builder()
                .tfAttrName("data_format")
                .propertyNames(new String[]{"dataFormat"})
                .build();
        attrs.put("dataFormat", dataFormatMapping);

        ret.put(tensorflowName(), attrs);
        return ret;
    }


    @Override
    public String opName() {
        return "depth_to_space";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"DepthToSpace"};
    }

    @Override
    public String tensorflowName() {
        return "DepthToSpace";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        return Collections.singletonList(dataTypes.get(0));
    }
}
