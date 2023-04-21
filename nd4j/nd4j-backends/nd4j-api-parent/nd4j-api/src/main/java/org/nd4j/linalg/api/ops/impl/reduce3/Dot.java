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

package org.nd4j.linalg.api.ops.impl.reduce3;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.bp.DotBp;

import java.util.List;

public class Dot extends BaseReduce3Op {

    public Dot(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Dot(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Dot(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Dot() {
    }

    /**
     * Full array dot product reduction, optionally along specified dimensions.<br>
     * See <a href="https://en.wikipedia.org/wiki/Dot_product">wikipedia</a> for details.
     *
     * @param x          input variable.
     * @param y          input variable.
     * @param z          (optional) place holder for the result. Must have the expected shape.
     * @param dimensions (optional) Dimensions to reduce over. If dimensions are not specified, full array reduction is performed.
     * @see org.nd4j.linalg.ops.transforms.Transforms#dot Transforms.dot(...) for a wrapper around the common use case of 2 INDArrays.
     */
    public Dot(INDArray x, INDArray y, INDArray z, long... dimensions) {
        this(x, y, z, true, false, dimensions);
    }


    /**
     * @see #Dot(INDArray x, INDArray y, INDArray z, long...)
     */
    public Dot(INDArray x, INDArray y, long... dimensions) {
        this(x, y, null, dimensions);
    }

    public Dot(INDArray x, INDArray y, boolean allDistances, long... dimensions) {
        super(x, y, allDistances, dimensions);
    }

    /**
     * @see #Dot(INDArray x, INDArray y, INDArray z, long...)
     */
    public Dot(INDArray x, INDArray y, INDArray z) {
        this(x, y, z, null);
    }

    public Dot(INDArray x, INDArray y, INDArray z, boolean keepDims, long... dimensions) {
        super(x, y, z, keepDims, dimensions);
    }

    /**
     * @see #Dot(INDArray x, INDArray y, INDArray z, long...)
     */
    public Dot(INDArray x, INDArray y, INDArray z, boolean newFormat, boolean keepDims, long... dimensions) {
        super(x, y, z, keepDims, false, dimensions);
    }

    public Dot(SameDiff sameDiff, SDVariable i_v, long[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String opName() {
        return "dot";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //TODO KEEP DIMS
        return new DotBp(sameDiff, arg(0), arg(1), f1.get(0), false, dimensions).outputs();
    }
}
