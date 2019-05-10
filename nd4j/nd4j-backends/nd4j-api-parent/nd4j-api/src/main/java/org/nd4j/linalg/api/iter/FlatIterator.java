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

package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.util.ArrayUtil;

import java.util.Iterator;

/**
 * Created by agibsonccc on 9/15/15.
 */
public class FlatIterator implements Iterator<int[]> {

    private int[] shape;
    private int runningDimension;
    private int[] currentCoord;
    private int length;
    private int current = 0;

    public FlatIterator(int[] shape) {
        this.shape = shape;
        this.currentCoord = new int[shape.length];
        length = ArrayUtil.prod(shape);
    }

    @Override
    public void remove() {

    }

    @Override
    public boolean hasNext() {
        return current < length;
    }

    @Override
    public int[] next() {
        if (currentCoord[runningDimension] == shape[runningDimension]) {
            runningDimension--;
            currentCoord[runningDimension] = 0;
            if (runningDimension < shape.length) {

            }
        } else {
            //bump to the next coordinate
            currentCoord[runningDimension]++;
        }
        current++;
        return currentCoord;
    }
}
