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

package org.nd4j.linalg.api.ops.impl.transforms.gradient;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


/**
 * Backprop operation for dynamic partition
 * @author Alex Black
 */
public class DynamicPartitionBp extends DynamicCustomOp {

    private int numPartitions;

    public DynamicPartitionBp() {
    }

    public DynamicPartitionBp(SameDiff sameDiff, SDVariable input, SDVariable partitions, SDVariable[] gradsAtOutput, int numPartitions) {
        super(null, sameDiff,  argsArray(input, partitions, gradsAtOutput), false);
        this.numPartitions = numPartitions;
        addArgs();
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Backprop not supported");
    }

    protected void addArgs() {
        addIArgument(numPartitions);
    }


    @Override
    public String opName() {
        return "dynamic_partition_bp";
    }

    @Override
    public int getNumOutputs(){
        return 2;   //input and partitions
    }

    private static SDVariable[] argsArray(SDVariable input, SDVariable partitions, SDVariable[] grads){
        SDVariable[] out = new SDVariable[grads.length + 2];
        out[0] = input;
        out[1] = partitions;
        System.arraycopy(grads, 0, out, 2, grads.length);
        return out;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //Input gradients and partition 'gradients' - same type as inputs
        return Arrays.asList(dataTypes.get(0), dataTypes.get(1));
    }

}
