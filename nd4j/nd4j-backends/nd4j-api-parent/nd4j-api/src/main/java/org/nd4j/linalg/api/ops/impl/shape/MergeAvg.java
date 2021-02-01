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

import lombok.extern.slf4j.Slf4j;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.bp.MergeAvgBp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

@Slf4j
public class MergeAvg extends DynamicCustomOp {

    public MergeAvg(INDArray... inputs){
        super(inputs, null);
    }

    public MergeAvg(SameDiff sameDiff, SDVariable... inputs) {
        super(null, sameDiff, inputs);
    }

    public MergeAvg(){ }

    @Override
    public String opName() {
        return "mergeavg";
    }


/*
    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        List<LongShapeDescriptor> ret = new ArrayList<>(1);
        ret.add(LongShapeDescriptor.fromShape(arg().getShape(), Nd4j.defaultFloatintPointType()));
        return ret;
    }
*/

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        // no-op
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Arrays.asList(new MergeAvgBp(sameDiff, args(), i_v.get(0)).outputVariables());

    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        DataType first = dataTypes.get(0);
        for( int i=1; i<dataTypes.size(); i++ ){
            DataType dt = dataTypes.get(i);
            Preconditions.checkState(first == dt, "All inputs must have same datatype - got %s and %s for inputs 0 and %s respectively", first, dt, i);
        }
        //Output type is same as input types if FP, or default FP type otherwise
        if(first.isFPType()){
            return Collections.singletonList(first);
        } else {
            return Collections.singletonList(Nd4j.defaultFloatingPointType());
        }
    }

}
