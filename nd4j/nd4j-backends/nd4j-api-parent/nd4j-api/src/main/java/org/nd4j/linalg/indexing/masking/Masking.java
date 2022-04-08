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

package org.nd4j.linalg.indexing.masking;

import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.Where;
import org.nd4j.linalg.api.ops.impl.shape.Gather;
import org.nd4j.linalg.api.ops.impl.shape.Squeeze;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.primitives.Longs;

import java.util.ArrayList;
import java.util.List;

public class Masking {


    public static SDVariable applyMask(SameDiff ret, SDVariable input,SDVariable mask,int axis) {
        SDVariable maskShape = mask.shape();
        SDVariable rank = mask.rank();
        SDVariable tensorShape = mask.shape();
        int maskRank = mask.rank().eval().getInt(0);
        SDVariable leadingSize = ret.prod(tensorShape.get(SDIndex.interval(0,mask.rank().eval().getInt(0))),0);
        input = input.reshape(ret.concat(0,tensorShape.get(SDIndex.interval(0,axis)),
                leadingSize,tensorShape.get(SDIndex.interval(axis,maskRank))));
        mask = mask.reshape(-1);
        SDVariable indices = ret.squeeze(ret.where(mask),0);
        SDVariable gathered = ret.gather(input,indices,axis);
        return gathered;
    }

    /**
     * Applies a boolean mask
     * to the given input.
     * This is equivalent to tensorflow's boolean_mask
     * @param input the input to mask
     * @param mask the target mask to apply
     * @param axis the axis to apply along
     * @return
     */
    public static INDArray applyMask(INDArray input,INDArray mask,int axis) {
        long[] maskShape = mask.shape();
        long rank = maskShape.length;
        long[] tensorShape = input.shape();
        Preconditions.checkState(maskShape.length > 0,"Mask shape must not be scalar");
        long leadingSize = 1;
        for(int i = 0; i < axis + rank; i++) {
            leadingSize *= tensorShape[i];
        }

        List<Long> retShape = new ArrayList<>();
        for(int i = 0; i < axis; i++) {
            retShape.add(tensorShape[i]);
        }

        retShape.add(leadingSize);

        for(int i = axis; i < axis + rank; i++) {
            retShape.add(tensorShape[i]);
        }

        INDArray retTensor = input.reshape(Longs.toArray(retShape));
        mask = mask.reshape(-1);
        INDArray whereMask = Nd4j.getExecutioner().exec(new Where(mask))[0];
        INDArray indices = Nd4j.getExecutioner().exec(new Squeeze(whereMask,1))[0];
        INDArray ret = Nd4j.getExecutioner().exec(new Gather(retTensor,indices,axis))[0];
        return ret;
    }


}
