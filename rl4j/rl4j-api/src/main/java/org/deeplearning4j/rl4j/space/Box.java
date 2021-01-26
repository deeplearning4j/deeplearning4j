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

package org.deeplearning4j.rl4j.space;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *
 * A Box observation
 *
 * @see <a href="https://gym.openai.com/envs#box2d">https://gym.openai.com/envs#box2d</a>
 */
public class Box implements Encodable {

    private final INDArray data;

    public Box(double... arr) {
        this.data = Nd4j.create(arr);
    }

    public Box(int[] shape, double... arr) {
        this.data = Nd4j.create(arr).reshape(shape);
    }

    private Box(INDArray toDup) {
        data = toDup.dup();
    }

    @Override
    public double[] toArray() {
        return data.data().asDouble();
    }

    @Override
    public boolean isSkipped() {
        return false;
    }

    @Override
    public INDArray getData() {
        return data;
    }

    @Override
    public Encodable dup() {
        return new Box(data);
    }
}
