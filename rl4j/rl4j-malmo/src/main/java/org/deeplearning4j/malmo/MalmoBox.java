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

package org.deeplearning4j.malmo;

import java.util.Arrays;

import org.deeplearning4j.rl4j.space.Encodable;

/**
 * Encodable state as a simple value array similar to Gym Box model, but without a JSON constructor
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class MalmoBox implements Encodable {
    double[] value;

    /**
     * Construct state from an array of doubles
     * @param value state values
     */
    //TODO: If this constructor was added to "Box", we wouldn't need this class at all.
    public MalmoBox(double... value) {
        this.value = value;
    }

    @Override
    public double[] toArray() {
        return value;
    }

    @Override
    public String toString() {
        return Arrays.toString(value);
    }
}
