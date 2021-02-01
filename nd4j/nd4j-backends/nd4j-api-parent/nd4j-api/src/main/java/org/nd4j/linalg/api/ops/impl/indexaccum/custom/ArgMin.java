/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * ArgMin function
 *
 * @author Alex Black
 */
@Data
public class ArgMin extends DynamicCustomOp {
    protected boolean keepDims = false;
    private int[] dimensions;

    protected DataType outputType = DataType.INT64;

    public ArgMin(SameDiff sameDiff, SDVariable i_v, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v);

        this.keepDims = keepDims;
        this.dimensions = dimensions;

        if (dimensions != null && dimensions.length > 0)
            addIArgument(dimensions);

        addBArgument(keepDims);

        addDArgument(outputType);
    }

    public ArgMin() {
    }

    public ArgMin(INDArray x, INDArray z, boolean keepDims, int... dimensions) {
        super(new INDArray[]{x}, z != null ? new INDArray[] {z} : new INDArray[0]);

        this.keepDims = keepDims;
        this.dimensions = dimensions;

        if (dimensions != null && dimensions.length > 0)
            addIArgument(dimensions);

        addBArgument(keepDims);

        addDArgument(outputType);
    }

    public ArgMin(INDArray x, INDArray z, int... dimensions) {
        this(x, z, false, dimensions);
    }

    public ArgMin(INDArray x, int... dimensions) {
        this(x, null, dimensions);
    }

    public ArgMin(INDArray x, boolean keepDims, int... dimensions) {
        this(x, null, keepDims, dimensions);
    }

    @Override
    public String opName() {
        return "argmin";
    }

    @Override
    public String tensorflowName() {
        return "ArgMin";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(attributesForNode.containsKey("output_type")) {
            outputType = TFGraphMapper.convertType(attributesForNode.get("output_type").getType());
        } else {
            outputType = DataType.LONG;
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && (inputDataTypes.size() == 1 || inputDataTypes.size() == 2),
                "Expected 1 or 2 input datatype to argmax, got %s", inputDataTypes);    //2nd input: axis
        //TODO make this output datatype configurable! (long/int)
        return Collections.singletonList(outputType == null ? DataType.LONG : outputType);
    }
}
