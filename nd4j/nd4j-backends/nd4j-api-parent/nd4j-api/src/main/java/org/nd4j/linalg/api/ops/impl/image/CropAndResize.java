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

package org.nd4j.linalg.api.ops.impl.image;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * CropAndResize Op
 * @author Alex Black
 */
public class CropAndResize extends DynamicCustomOp {
    public enum Method {BILINEAR, NEAREST};
    protected Method method = Method.BILINEAR;
    protected double extrapolationValue = 0.0;

    @Override
    public String opName() {
        return "crop_and_resize";
    }

    @Override
    public String tensorflowName() {
        return "CropAndResize";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);

        String method = attributesForNode.get("method").getS().toStringUtf8();
        if(method.equalsIgnoreCase("nearest")){
            this.method = Method.NEAREST;
        } else {
            this.method = Method.BILINEAR;
        }

        if(attributesForNode.containsKey("extrapolation_value")){
            extrapolationValue = attributesForNode.get("extrapolation_value").getF();
        }

        addArgs();
    }

    protected void addArgs() {
        iArguments.clear();
        tArguments.clear();
        addIArgument(method == Method.BILINEAR ? 0 : 1);
        addTArgument(extrapolationValue);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //TODO we can probably skip this sometimes...
        List<SDVariable> out = new ArrayList<>();
        for(SDVariable v : args()){
            out.add(sameDiff.zerosLike(v));
        }
        return out;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 4,
                "Expected 4 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(DataType.FLOAT);   //TF import: always returns float32...
    }
}
