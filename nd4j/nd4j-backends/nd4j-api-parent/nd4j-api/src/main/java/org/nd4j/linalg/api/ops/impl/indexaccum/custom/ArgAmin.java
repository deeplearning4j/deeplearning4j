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

package org.nd4j.linalg.api.ops.impl.indexaccum.custom;

import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BaseDynamicCustomIndexReduction;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@Data
public class ArgAmin extends BaseDynamicCustomIndexReduction {
    public ArgAmin() {
    }

    public ArgAmin(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
    }

    public ArgAmin(SameDiff sameDiff, SDVariable[] args, boolean keepDims, int[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
    }

    public ArgAmin(INDArray[] inputs) {
        super(inputs, null);
    }


    public ArgAmin(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public ArgAmin(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
    }

    public ArgAmin(INDArray[] inputs, INDArray[] outputs, boolean keepDims, int... dimensions) {
        super(inputs, outputs, keepDims, dimensions);
    }

    public ArgAmin(INDArray[] inputs, int[] dim) {
        this(inputs,null,false,dim);
    }


    @Override
    public String opName() {
        return "argamin";
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

}
