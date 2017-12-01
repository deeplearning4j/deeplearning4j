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

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

/**
 * Permute function
 *
 * @author Adam Gibson
 */
public class Permute extends Transpose {

    public Permute(SameDiff sameDiff, DifferentialFunction i_v, int[] permuteDims) {
        super(sameDiff,i_v);
        this.permuteDims = permuteDims;
    }

    public Permute() {}

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "permute";
    }





    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        return Collections.<DifferentialFunction>singletonList(this);
    }

    @Override
    public String onnxName() {
        return "Transpose";
    }

    @Override
    public String tensorflowName() {
        return "Transpose";
    }


}
