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

import org.json.JSONArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *
 * A Box observation
 *
 * @see <a href="https://gym.openai.com/envs#box2d">https://gym.openai.com/envs#box2d</a>
 */
public class Box implements Encodable {

    private final double[] array;

    public Box(JSONArray arr) {

        int lg = arr.length();
        this.array = new double[lg];

        for (int i = 0; i < lg; i++) {
            this.array[i] = arr.getDouble(i);
        }
    }

    public double[] toArray() {
        return array;
    }
}
