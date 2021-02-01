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

package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitXavierUniform implements IWeightInit {

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        //As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
        //Eq 16: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        double s = Math.sqrt(6.0) / Math.sqrt(fanIn + fanOut);
        Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-s, s));
        return paramView.reshape(order, shape);
    }
}
