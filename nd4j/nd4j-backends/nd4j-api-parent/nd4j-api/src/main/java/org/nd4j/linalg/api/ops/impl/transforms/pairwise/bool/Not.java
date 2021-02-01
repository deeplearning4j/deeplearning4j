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

package org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformBoolOp;

import java.util.Collections;
import java.util.List;

/**
 * Boolean AND pairwise transform
 *
 * @author raver119@gmail.com
 */
public class Not extends BaseTransformBoolOp {

    protected double comparable = 0.0;

    public Not(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v, false);
        this.extraArgs = new Object[] {this.comparable};
    }

    public Not() {
        this.extraArgs = new Object[] {this.comparable};
    }

    public Not(@NonNull INDArray x, INDArray y, INDArray z, Number comparable) {
        super(x, y, z);
        this.comparable = comparable.doubleValue();
        this.extraArgs = new Object[] {this.comparable};
    }

    @Override
    public int opNum() {
        return 10;
    }

    @Override
    public String opName() {
        return "not";
    }

    @Override
    public String onnxName() {
        return "Not";
    }
    
    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }
}
