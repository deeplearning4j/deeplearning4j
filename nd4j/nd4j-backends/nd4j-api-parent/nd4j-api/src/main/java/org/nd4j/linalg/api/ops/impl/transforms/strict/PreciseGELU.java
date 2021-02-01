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

package org.nd4j.linalg.api.ops.impl.transforms.strict;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;

import java.util.Collections;
import java.util.List;

/**
 * GELU activation function - Gaussian Error Linear Units<br>
 * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a>
 * Note: This op implements both the sigmoid and tanh-based approximations; to use the sigmoid approximation (recommended)
 * use precise=false; otherwise, use precise = true for the slower but marginally more accurate tanh version.
 * @author raver119@gmail.com
 */
public class PreciseGELU extends BaseTransformStrictOp {
    public PreciseGELU(SameDiff sameDiff, SDVariable i_v, boolean inPlace, boolean precise) {
        super(sameDiff, i_v, inPlace);
    }

    public PreciseGELU(SameDiff sameDiff, SDVariable i_v) {
        this(sameDiff, i_v, false, true);
    }

    public PreciseGELU() {
    }

    public PreciseGELU(INDArray x, INDArray z) {
        super(x, z);
    }

    public PreciseGELU(INDArray ndArray) {
        super(ndArray);
    }

    @Override
    public int opNum() {
        return 55;
    }

    @Override
    public String opName() {
        return "precise_gelu";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("Tensorflow name not found for " + opName());
        //return "PreciseGELU";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = new PreciseGELUDerivative(sameDiff, arg(), false, true).outputVariable().mul(i_v.get(0));
        return Collections.singletonList(ret);
    }


}
