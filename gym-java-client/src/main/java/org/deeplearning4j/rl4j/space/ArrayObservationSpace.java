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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @param <O> the type of Observation
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *         <p>
 *         An array observation space enables you to create an Observation Space of custom dimension
 */

@Value
public class ArrayObservationSpace<O> implements ObservationSpace<O> {

    String name;
    int[] shape;
    INDArray low;
    INDArray high;

    public ArrayObservationSpace(int[] shape) {
        name = "Custom";
        this.shape = shape;
        low = Nd4j.create(1);
        high = Nd4j.create(1);
    }

}
