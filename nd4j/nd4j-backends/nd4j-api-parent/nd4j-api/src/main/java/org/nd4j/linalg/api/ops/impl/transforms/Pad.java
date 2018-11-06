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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 * Pad op
 * @author Alex Black
 */
public class Pad extends DynamicCustomOp {

    public enum Mode {CONSTANT, REFLECT, SYMMETRIC}

    private Mode mode;
    private double constant;

    public Pad(){ }

    public Pad(SameDiff sd, SDVariable in, SDVariable padding, Mode mode){
        super(sd, new SDVariable[]{in, padding}, false);
        this.mode = mode;
        addIArgument(mode.ordinal());
    }

    @Override
    public String opName(){
        return "pad";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"Pad", "PadV2"};
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        //Based on TF codebase: gen_array_ops.mirror_pad is osed for BOTH REFLECT and SYMMETRIC mode. Hence only constant being imported here
        this.mode = Mode.CONSTANT;
        addIArgument(mode.ordinal());
        //Constant value is resolved just before execution
    }

    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        if(args().length == 3){
            INDArray arr = arg(2).getArr();
            this.tArguments.clear();
            this.tArguments.add(arr.getDouble(0));
        }
        super.resolvePropertiesFromSameDiffBeforeExecution();
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
