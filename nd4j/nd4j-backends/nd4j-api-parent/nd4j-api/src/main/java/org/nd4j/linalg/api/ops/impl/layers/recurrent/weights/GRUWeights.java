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
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;

/**
 * The weight configuration of a GRU cell.  For {@link GRUCell}.
 *
 */
@EqualsAndHashCode(callSuper = true)
@Data
@Builder
public class GRUWeights extends RNNWeights {

    /**
     * Reset and Update gate weights, with a shape of [inSize + numUnits, 2*numUnits].
     *
     * The reset weights are the [:, 0:numUnits] subset and the update weights are the [:, numUnits:2*numUnits] subset.
     */
    private SDVariable ruWeight;
    private INDArray iRuWeights;

    /**
     * Cell gate weights, with a shape of [inSize + numUnits, numUnits]
     */
    private SDVariable cWeight;
    private INDArray iCWeight;

    /**
     * Reset and Update gate bias, with a shape of [2*numUnits].  May be null.
     *
     * The reset bias is the [0:numUnits] subset and the update bias is the [numUnits:2*numUnits] subset.
     */
    private SDVariable ruBias;
    private INDArray iRUBias;

    /**
     * Cell gate bias, with a shape of [numUnits].  May be null.
     */
    private SDVariable cBias;
    private INDArray iCBias;

    @Override
    public SDVariable[] args() {
        return filterNonNull(ruWeight, cWeight, ruBias, cBias);
    }

    @Override
    public INDArray[] arrayArgs() {
        return filterNonNull(iRuWeights, iCWeight, iRUBias, iCBias);
    }
}
