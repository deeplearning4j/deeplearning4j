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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformSameOp;

import java.util.Arrays;
import java.util.List;

/**
 * Calculate the minimum value between two arrays in an elementwise fashion, broadcasting if required
 *
 * @author raver119@gmail.com
 */
public class Min extends BaseTransformSameOp  {

    public Min(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2) {
        super(sameDiff, i_v, i_v2);
    }

    public Min() {}

    public Min(INDArray x, INDArray y, INDArray z) {
        super(x, y, z);
    }


    @Override
    public int opNum() {
        return 8;
    }

    @Override
    public String opName() {
        return "min_pairwise";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //TODO optimize
        SDVariable gt = arg(0).gt(arg(1)).castTo(arg(0).dataType());
        SDVariable lt = arg(0).lt(arg(1)).castTo(arg(1).dataType());
        return Arrays.asList(lt.mul(f1.get(0)), gt.mul(f1.get(0)));
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }
}
