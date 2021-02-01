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

package org.deeplearning4j.remote.helpers;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.adapters.InferenceAdapter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

@Slf4j
public class HouseToPredictedPriceAdapter implements InferenceAdapter<House, PredictedPrice> {

    @Override
    public MultiDataSet apply(@NonNull House input) {
        // we just create vector array with shape[4] and assign it's value to the district value
        return new MultiDataSet(Nd4j.create(DataType.FLOAT, 1, 4).assign(input.getDistrict()), null);
    }

    @Override
    public PredictedPrice apply(INDArray... nnOutput) {
        return new PredictedPrice(nnOutput[0].getFloat(0));
    }
}
