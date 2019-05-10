/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.space;

import lombok.Value;
import org.json.JSONArray;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/26/16.
 */
@Value
public class HighLowDiscrete extends DiscreteSpace {

    //size of the space also defined as the number of different actions
    INDArray matrix;

    public HighLowDiscrete(INDArray matrix) {
        super(matrix.rows());
        this.matrix = matrix;
    }

    @Override
    public Object encode(Integer a) {
        JSONArray jsonArray = new JSONArray();
        for (int i = 0; i < size; i++) {
            jsonArray.put(matrix.getDouble(i, 0));
        }
        jsonArray.put(a - 1, matrix.getDouble(a - 1, 1));
        return jsonArray;
    }

}
