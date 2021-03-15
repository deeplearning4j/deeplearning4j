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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.bp.TileBp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Tile extends DynamicCustomOp {

    private int[] jaxis;
    private boolean is_static_reps = false;

    public Tile(SameDiff sameDiff, SDVariable i_v, int[] axis) {
        super(null,sameDiff, new SDVariable[]{i_v}, false);
        this.jaxis = axis;
        addArguments();
    }

    public Tile(SameDiff sameDiff, SDVariable i_v, SDVariable axis) {
        super(null,sameDiff, new SDVariable[]{i_v, axis}, false);
        this.jaxis = null;
    }

    public Tile(INDArray[] inputs, INDArray[] outputs, int[] axis, boolean is_static_reps) {
        super(null, inputs, outputs);
        this.jaxis = axis;
        this.is_static_reps = is_static_reps;
        addArguments();
    }


    public Tile(INDArray[] inputs, INDArray[] outputs, int[] axis) {
        this(inputs,outputs,axis,false);
    }

    public Tile(INDArray x, INDArray repeat){
        super(null, new INDArray[] {x, repeat}, null);
        this.jaxis = null;
    }

    public Tile(INDArray inputs, int... axis){
        super(null, new INDArray[] {inputs}, null);
        this.jaxis = axis;
        this.is_static_reps = true;
        addArguments();
    }

    public Tile() {}

    private void addArguments() {
        this.is_static_reps = true;
        addIArgument(jaxis);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val axisMapping = PropertyMapping.builder()
                .onnxAttrName("axis")
                .tfInputPosition(-1)
                .propertyNames(new String[]{"axis"})
                .build();

        map.put("axis",axisMapping);

        ret.put(tensorflowName(),map);
        ret.put(onnxName(),map);

        return ret;
    }

    @Override
    public String opName() {
        return "tile";
    }

    @Override
    public String onnxName() {
        return "Tile";
    }

    @Override
    public String tensorflowName() {
        return "Tile";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        if(jaxis != null){
            return new TileBp(sameDiff, arg(), i_v.get(0), jaxis).outputs();
        }else{
            return new TileBp(sameDiff, arg(0), arg(1), i_v.get(0)).outputs();
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //2nd isput is dynamic repeat
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || (jaxis == null && dataTypes.size() == 2)),
                "Expected 1 or 2 input datatypes for %s, got %s", getClass(), dataTypes);
        //Output type is same as input type
        return Collections.singletonList(dataTypes.get(0));
    }
}
