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

package org.nd4j.linalg.api.ops;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.util.LinAlgExceptions;

import java.util.List;

@Slf4j
public abstract class BaseTransformOp extends BaseOp implements TransformOp {


    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2) {
        this(sameDiff,i_v1,i_v2,false);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2,
                           boolean inPlace) {
        super(sameDiff,inPlace,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v1, this);
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v2, this);
            this.sameDiff = sameDiff;
            this.inPlace = inPlace;
            this.xVertexId = i_v1.name();
            this.yVertexId = i_v2.name();
            sameDiff.addArgsFor(new SDVariable[]{i_v1,i_v2},this);
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }


    }

    public BaseTransformOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2,
                           Object[] extraArgs) {
        super(sameDiff,extraArgs);
        if (i_v1 != null && i_v2 != null) {

            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v1, this);
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v2, this);
            this.sameDiff = sameDiff;
            this.xVertexId = i_v1.name();
            this.yVertexId = i_v2.name();
            sameDiff.addArgsFor(new SDVariable[]{i_v1,i_v2},this);
        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }

    }




    public BaseTransformOp(SameDiff sameDiff,SDVariable i_v,boolean inPlace) {
        this(sameDiff,i_v,i_v.getShape(),inPlace,null);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v,
                           long[] shape,
                           boolean inPlace,
                           Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);

        if (i_v != null) {
            SameDiffUtils.validateDifferentialFunctionSameDiff(sameDiff, i_v, this);
            this.xVertexId = i_v.name();
            sameDiff.addArgsFor(new SDVariable[]{i_v},this);
        } else {
            throw new IllegalArgumentException("Input must not null variable.");
        }

    }


    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v,
                           Object[] extraArgs) {
        this(sameDiff,i_v,i_v.getShape(),false,extraArgs);
    }



    public BaseTransformOp(INDArray x, INDArray z) {
        super(x, z);
        LinAlgExceptions.assertSameShape(x, z);
    }

    public BaseTransformOp() {}

    public BaseTransformOp(INDArray x, INDArray y, INDArray z) {
        super(x, y, z);
        if (y != null)
            LinAlgExceptions.assertSameLength(x, y, z);
        else if (z != null)
            LinAlgExceptions.assertSameLength(x, z);

    }

    public BaseTransformOp(INDArray x) {
        super(x);
    }

    public abstract List<LongShapeDescriptor> calculateOutputShape();


    @Override
    public INDArray z() {
        return z;
    }


}
