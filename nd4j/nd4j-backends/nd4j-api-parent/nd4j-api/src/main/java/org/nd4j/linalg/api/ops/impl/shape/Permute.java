/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.shape;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;

/**
 * Permute function
 *
 * @author Adam Gibson
 */
public class Permute extends Transpose {

    private int[] reverseDims;

    public Permute(SameDiff sameDiff, SDVariable i_v, int... permuteDims) {
        super(sameDiff, i_v);
        this.permuteDims = permuteDims;
        this.reverseDims = new int[permuteDims.length];
        for (int i = 0; i < reverseDims.length; i++) {
            reverseDims[i] = ArrayUtils.indexOf(permuteDims, i);
        }
        addIArgument(permuteDims);
    }

    public Permute(INDArray input, INDArray result, int... permuteDims){
        super(input, result);
        this.permuteDims = permuteDims;
        this.reverseDims = new int[permuteDims.length];
        for (int i = 0; i < reverseDims.length; i++) {
            reverseDims[i] = ArrayUtils.indexOf(permuteDims, i);
        }
        addIArgument(permuteDims);
    }

    public Permute() {
    }

    @Override
    public String opName() {
        return "permute";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = f().permute(i_v.get(0), reverseDims);
        return Arrays.asList(ret);
    }
}
