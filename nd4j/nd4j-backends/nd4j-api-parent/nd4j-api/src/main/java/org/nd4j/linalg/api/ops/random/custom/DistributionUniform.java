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

package org.nd4j.linalg.api.ops.random.custom;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Uniform distribution wrapper
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DistributionUniform extends DynamicCustomOp {
    private double min = 0.0;
    private double max = 1.0;
    private DataType dataType;

    public DistributionUniform() {
        //
    }

    public DistributionUniform(SameDiff sd, SDVariable shape, double min, double max) {
        this(sd, shape, min, max, null);
    }

    public DistributionUniform(SameDiff sd, SDVariable shape, double min, double max, DataType dataType){
        super(null, sd, new SDVariable[]{shape});
        Preconditions.checkState(min <= max, "Minimum (%s) must be <= max (%s)", min, max);
        Preconditions.checkState(dataType == null || dataType.isNumerical(), "Only numerical datatypes can be used with DistributionUniform - rquested output datatype: %s", dataType);
        this.dataType = dataType;
        this.min = min;
        this.max = max;
        addArgs();
    }

    public DistributionUniform(INDArray shape, INDArray out, double min, double max) {
        this(shape, out, min, max, null);
    }

    public DistributionUniform(INDArray shape, INDArray out, double min, double max, DataType dataType){
        super(null, new INDArray[]{shape}, new INDArray[]{out}, Arrays.asList(min, max), (List<Integer>)null);
        this.min = min;
        this.max = max;
        this.dataType = dataType;
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        AttrValue vDtype = attributesForNode.get("dtype");
        AttrValue vTout = attributesForNode.get("Tout");
        if (vDtype == null && vTout == null) {
            throw new ND4JIllegalStateException("Unable to find output data type for node " + nodeDef.getName());
        }
        AttrValue v = vDtype == null ? vTout : vDtype;
        dataType = TFGraphMapper.convertType(v.getType());
        addIArgument(dataType.toInt());
        addTArgument(0.0, 1.0); //TF version is hardcoded 0 to 1
    }

    protected void addArgs() {
        tArguments.clear();
        addTArgument(min, max);
        if(dataType != null){
            iArguments.clear();
            addIArgument(dataType.toInt());
        }
    }

    @Override
    public String opName() {
        return "randomuniform";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"RandomUniform","RandomUniformInt"};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null /*&& inputDataTypes.size() == 1*/, "Expected input datatypes for %s, got %s", getClass(), inputDataTypes);
        //Input data type specifies the shape
        if(dataType != null){
            return Collections.singletonList(dataType);
        }
        return Collections.singletonList(DataType.FLOAT);
    }
}
