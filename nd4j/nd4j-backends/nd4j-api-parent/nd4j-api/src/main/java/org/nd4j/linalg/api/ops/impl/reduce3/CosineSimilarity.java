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

package org.nd4j.linalg.api.ops.impl.reduce3;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Cosine similarity
 * Note that you need to initialize
 * a scaling constant equal to the norm2 of the
 * vector
 *
 * @author Adam Gibson
 */
public class CosineSimilarity extends BaseReduce3Op {
    public static final String OP_NAME = "cosinesimilarity";

    public CosineSimilarity(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public CosineSimilarity(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public CosineSimilarity() {
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineSimilarity(INDArray x, INDArray y, INDArray z, int... dimensions) {
        super(x, y, z, dimensions);
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineSimilarity(INDArray x, INDArray y, int... dimensions) {
        this(x, y, null, dimensions);
    }

    public CosineSimilarity(INDArray x, INDArray y, INDArray z) {
        this(x, y, z, null);
    }

    public CosineSimilarity(INDArray x, INDArray y, INDArray z, boolean allDistances, int... dimension) {
        this(x, y, z, dimension);
        this.isComplex = allDistances;
    }

    public CosineSimilarity(INDArray x, INDArray y, boolean allDistances, int... dimension) {
        this(x, y, null, allDistances, dimension);
    }

    public CosineSimilarity(INDArray x, INDArray y, INDArray z, boolean keepDims, boolean allDistances, int... dimensions){
        super(x, y, z, keepDims, allDistances, dimensions);
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    @Override
    public int opNum() {
        return 2;
    }

    @Override
    public String opName() {
        return OP_NAME;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //Let cosine(x,y) = a / b
        //a = sum_i (x_i * y_i)
        //b = sqrt(sum_i x_i^2) * sqrt(sum_i y_i^2) = l2(x) * l2(y)
        //Then:
        // dc(x,y)/dx_i = 1/b * (y - x * a / (l2(x))^2)

        return doDiff(sameDiff, larg(), rarg(), i_v1.get(0), keepDims, dimensions);
    }

    public static List<SDVariable> doDiff(SameDiff sameDiff, SDVariable x, SDVariable y,
                                          SDVariable gradOut, boolean keepDims, int... dimensions){
        SDVariable a = sameDiff.sum(x.mul(y),true, dimensions);
        SDVariable l2x = sameDiff.norm2(x, true, dimensions);
        SDVariable l2y = sameDiff.norm2(y, true, dimensions);
        SDVariable b = l2x.mul(l2y);

        SDVariable l2xSq = sameDiff.math().square(l2x);
        SDVariable l2ySq = sameDiff.math().square(l2y);
        SDVariable broadcastableGrad;
        if(keepDims || dimensions == null || dimensions.length == 0 || (dimensions.length == 1 && dimensions[0] == Integer.MAX_VALUE)){
            //keepDims or full array reduction
            broadcastableGrad = gradOut;
        } else {
            broadcastableGrad = SameDiffUtils.reductionBroadcastableWithOrigShape(x, sameDiff.constant(Nd4j.createFromArray(dimensions)), gradOut);
        }

        SDVariable dcdx = y.sub(x.mul(a).div(l2xSq)).div(b);
        SDVariable dcdy = x.sub(y.mul(a).div(l2ySq)).div(b);

        return Arrays.asList(dcdx.mul(broadcastableGrad), dcdy.mul(broadcastableGrad));
    }
}
