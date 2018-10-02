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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


/**
 * Transforms a given input tensor into numPartitions partitions, as indicated by the indices in "partitions".
 * Output tensor has one more dimension than input tensor, the first dimension indicates the partition.
 *
 * Example:
 *
 * input:           [4, 3, 5, 7, 8, 0]
 * input shape:     [1, 6]
 * partitions:      [1, 0, 1, 0, 0, 1]
 * numPartitions:   2
 * outputs[0]:      [3, 7, 8]
 * outputs[1]:      [4, 5, 0]
 *
 * @author Max Pumperla
 */
public class DynamicPartition extends DynamicCustomOp {

    private int numPartitions;
    private SDVariable partitions;

    public DynamicPartition() {
    }

    public DynamicPartition(SameDiff sameDiff, SDVariable input,  SDVariable partitions, int numPartitions) {
        super(null, sameDiff,  new SDVariable[] {input, partitions}, false);

        this.partitions = partitions;
        this.numPartitions = numPartitions;
        addArgs();
    }


//    @Override
//    public List<SDVariable> doDiff(List<SDVariable> i_v) {
//        // DynamicPartition and DynamicStitch are mutually inverse
//        SDVariable[] gradients = (SDVariable[]) i_v.toArray();
////         TODO: compute indices from partitions
//        SDVariable[] indices = f(partitions, numPartitions);
//        SDVariable ret = sameDiff.dynamicStitch(gradients, indices);
//        return Collections.singletonList(i_v.get(0));
//    }

    protected void addArgs() {
        addIArgument(numPartitions);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
        addArgs();
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> attrs = new LinkedHashMap<>();

        val numPartitions = PropertyMapping.builder()
                .tfAttrName("num_partitions")
                .propertyNames(new String[]{"numPartitions"})
                .build();
        attrs.put("numPartitions", numPartitions);

        ret.put(tensorflowName(),attrs);
        return ret;
    }


    @Override
    public String opName() {
        return "dynamic_partition";
    }


    @Override
    public String tensorflowName() {
        return "DynamicPartition";
    }

    @Override
    public String onnxName() {
        return "Dynamic partitioning currently not supported by ONNX";
    }

    @Override
    public int getNumOutputs(){
        return numPartitions;
    }

}
