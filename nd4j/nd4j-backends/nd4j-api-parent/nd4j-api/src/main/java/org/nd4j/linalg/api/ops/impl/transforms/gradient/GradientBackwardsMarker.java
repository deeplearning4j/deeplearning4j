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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseGradientOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 *
 */
public class GradientBackwardsMarker extends BaseGradientOp  {
    public static final String OP_NAME = "gradientbackwards";

    public GradientBackwardsMarker(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public GradientBackwardsMarker(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public GradientBackwardsMarker(INDArray x, INDArray z) {
        super(x, z);
    }

    public GradientBackwardsMarker() {
    }

    public GradientBackwardsMarker(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public GradientBackwardsMarker(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, z.lengthLong());
    }

    public GradientBackwardsMarker(INDArray x) {
        super(x);
    }

    /**
     * An op number
     *
     * @return
     */
    @Override
    public int opNum() {
        return 0;
    }

    /**
     * The opName of this operation
     *
     * @return the opName of this operation
     */
    @Override
    public String opName() {
        return OP_NAME;
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public void exec() {
       //no-op

    }

    @Override
    public void exec(int... dimensions) {
        super.exec(dimensions);
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return i_v;
    }

    @Override
    public DataType resultType() {
        if (x() != null)
            return x().dataType();

        return Nd4j.defaultFloatingPointType();
    }
}
