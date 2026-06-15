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

package org.nd4j.linalg.api.ops.impl.scalar;

import java.util.List;
import java.util.Map;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUBp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

public class LeakyReLU extends BaseScalarOp {
    public static final double DEFAULT_ALPHA = 0.01;
    private double alpha = DEFAULT_ALPHA;



    public LeakyReLU(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double alpha) {
        super(sameDiff, i_v, alpha, inPlace);
        this.alpha = alpha;
        this.extraArgs = new Object[]{alpha};

    }

    public LeakyReLU(SameDiff sameDiff, SDVariable i_v, double alpha) {
        this(sameDiff, i_v, false, alpha);
    }

    public LeakyReLU(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double alpha) {
        super(sameDiff, i_v, alpha, extraArgs);
        this.alpha = alpha;
        this.extraArgs = new Object[]{alpha};
    }

    public LeakyReLU() {
        super();
    }

    public LeakyReLU(INDArray x, double alpha) {
        super(x, alpha);
        this.alpha = alpha;
        this.extraArgs = new Object[]{alpha};
    }

    public LeakyReLU(INDArray x, INDArray z, double alpha) {
        super(x, null, z, alpha);
        this.alpha = alpha;
        this.extraArgs = new Object[]{alpha};
    }


    public LeakyReLU(INDArray x, INDArray z) {
        this(x, z, 0.01);
    }

    public LeakyReLU(INDArray x) {
        super(x, 0.01);
    }

    @Override
    public int opNum() {
        return 35;
    }

    @Override
    public String opName() {
        return "leakyrelu";
    }

    @Override
    public String onnxName() {
        return "LeakyRelu";
    }

    @Override
    public String tensorflowName() {
        return "LeakyRelu";
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        super.setPropertiesForFunction(properties);
        // After reflection sets the 'alpha' field, sync the scalar value
        // so the C++ kernel receives the correct alpha during execution.
        // Without this, ONNX import leaves scalarValue=null (from no-arg constructor)
        // while the alpha field has the correct value from the attribute mapping.
        this.extraArgs = new Object[]{alpha};
        // Use setScalar(INDArray) to avoid NPE from setScalar(Number) calling x.dataType()
        // when x is not yet set during import-time property initialization.
        // Use the input array's dtype if available, otherwise default to DOUBLE to preserve
        // precision for non-FLOAT inputs (e.g. DOUBLE input must produce DOUBLE output).
        DataType scalarType = (x != null) ? x.dataType() : DataType.DOUBLE;
        this.setScalar(Nd4j.scalar(scalarType, alpha));
    }


    public LeakyReLU(SameDiff sameDiff, SDVariable i_v, Number scalar) {
        super(sameDiff, i_v, scalar);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return new LeakyReLUBp(sameDiff, arg(), i_v.get(0), alpha).outputs();
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode,
            GraphDef graph) {
        alpha = attributesForNode.get("alpha").getF();
        extraArgs = new Object[]{alpha};
        // TensorFlow attribute "alpha" is always a float32 scalar in the protobuf, but the
        // op's output dtype must match the input tensor's dtype.  Use x.dataType() when
        // available; fall back to DOUBLE (not FLOAT) so that DOUBLE inputs are not silently
        // downcast to FLOAT.
        DataType scalarType = (x != null) ? x.dataType() : DataType.DOUBLE;
        this.setScalar(Nd4j.scalar(scalarType, alpha));
    }
}
