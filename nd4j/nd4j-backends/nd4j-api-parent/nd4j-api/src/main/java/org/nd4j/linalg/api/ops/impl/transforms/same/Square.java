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

package org.nd4j.linalg.api.ops.impl.transforms.same;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformSameOp;
import org.nd4j.linalg.api.ops.impl.scalar.PowDerivative;

import java.util.Collections;
import java.util.List;

/**
 * Square function (x ^ 2)
 *
 * @author Adam Gibson
 */
public class Square extends BaseTransformSameOp {
    public Square(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Square(SameDiff sameDiff, SDVariable i_v) {
        this(sameDiff, i_v, false);
    }

    public Square() {
    }

    public Square(INDArray x, INDArray z) {
        super(x, z);
    }

    public Square(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 12;
    }

    @Override
    public String opName() {
        return "square";
    }

    @Override
    public String onnxName() {
        return "Square";
    }

    @Override
    public String tensorflowName() {
        return "Square";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable g = new PowDerivative(sameDiff, arg(), false, 2).outputVariable().mul(i_v.get(0));
        return Collections.singletonList(g);
    }
}
