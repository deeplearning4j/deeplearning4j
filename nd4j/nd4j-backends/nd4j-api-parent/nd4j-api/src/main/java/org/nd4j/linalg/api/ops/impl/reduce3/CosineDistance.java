/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;

/**
 * Cosine distance
 * Note that you need to initialize
 * a scaling constant equal to the norm2 of the
 * vector
 *
 * @author raver119@gmail.com
 */
public class CosineDistance extends BaseReduce3Op {

    public CosineDistance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public CosineDistance() {
    }

    public CosineDistance(INDArray x, INDArray y, INDArray z) {
        this(x, y, z, null);
    }

    public CosineDistance(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, dimension);
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    public CosineDistance(INDArray x, INDArray y, int... dimension) {
        this(x, y, null, dimension);
    }

    public CosineDistance(INDArray x, INDArray y, INDArray z, boolean allDistances, int... dimension) {
        this(x, y, z, dimension);
        this.isComplex = allDistances;
    }

    public CosineDistance(INDArray x, INDArray y, boolean allDistances, int... dimension) {
        this(x, y, null, allDistances, dimension);
    }

    public CosineDistance(INDArray x, INDArray y, INDArray z, boolean keepDims, boolean allDistances, int... dimensions){
        super(x, y, z, keepDims, allDistances, dimensions);
        extraArgs = new Object[]{0.0f, 0.0f};
    }

    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String opName() {
        return "cosinedistance";
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        //Cosine distance = 1 - cosine similarity
        //Therefore: just need to negate gradients from cosine similarity...

        List<SDVariable> diff = CosineSimilarity.doDiff(sameDiff, larg(), rarg(), i_v1.get(0), keepDims, dimensions);
        return Arrays.asList(sameDiff.math.neg(diff.get(0)), sameDiff.math.neg(diff.get(1)));
    }
}
