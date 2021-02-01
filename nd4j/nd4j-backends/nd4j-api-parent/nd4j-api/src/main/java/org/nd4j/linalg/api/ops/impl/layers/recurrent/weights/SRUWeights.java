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

package org.nd4j.linalg.api.ops.impl.layers.recurrent.weights;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell;

/**
 * The weight configuration of a SRU layer.  For {@link SRU} and {@link SRUCell}.
 *
 */
@EqualsAndHashCode(callSuper = true)
@Data
@Builder
public class SRUWeights extends RNNWeights {

    /**
     * Weights, with shape [inSize, 3*inSize].
     */
    private SDVariable weights;

    private INDArray iWeights;

    /**
     * Biases, with shape [2*inSize].
     */
    private SDVariable bias;

    private INDArray iBias;

    @Override
    public SDVariable[] args() {
        return new SDVariable[]{weights, bias};
    }

    @Override
    public INDArray[] arrayArgs() {
        return new INDArray[]{iWeights, iBias};
    }
}
