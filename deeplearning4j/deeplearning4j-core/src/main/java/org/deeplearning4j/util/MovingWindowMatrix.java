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

package org.deeplearning4j.util;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * Moving window on a matrix (usually used for images)
 *
 * Given a:          This is a list of flattened arrays:
 * 1 1 1 1          1 1 2 2
 * 2 2 2 2 ---->    1 1 2 2
 * 3 3 3 3          3 3 4 4
 * 4 4 4 4          3 3 4 4
 *
 * @author Adam Gibson
 */
public class MovingWindowMatrix {

    private int windowRowSize = 28;
    private int windowColumnSize = 28;
    private INDArray toSlice;
    private boolean addRotate = false;


    /**
     *
     * @param toSlice matrix to slice
     * @param windowRowSize the number of rows in each window
     * @param windowColumnSize the number of columns in each window
     * @param addRotate whether to add the possible rotations of each moving window
     */
    public MovingWindowMatrix(INDArray toSlice, int windowRowSize, int windowColumnSize, boolean addRotate) {
        this.toSlice = toSlice;
        this.windowRowSize = windowRowSize;
        this.windowColumnSize = windowColumnSize;
        this.addRotate = addRotate;
    }


    /**
     * Same as calling new MovingWindowMatrix(toSlice,windowRowSize,windowColumnSize,false)
     * @param toSlice
     * @param windowRowSize
     * @param windowColumnSize
     */
    public MovingWindowMatrix(INDArray toSlice, int windowRowSize, int windowColumnSize) {
        this(toSlice, windowRowSize, windowColumnSize, false);
    }



    /**
     * Returns a list of non flattened moving window matrices
     * @return the list of matrices
     */
    public List<INDArray> windows() {
        return windows(false);
    }

    /**
     * Moving window, capture a row x column moving window of
     * a given matrix
     * @param flattened whether the arrays should be flattened or not
     * @return the list of moving windows
     */
    public List<INDArray> windows(boolean flattened) {
        List<INDArray> ret = new ArrayList<>();
        int window = 0;

        for (int i = 0; i < toSlice.length(); i++) {
            if (window >= toSlice.length())
                break;
            double[] w = new double[this.windowRowSize * this.windowColumnSize];
            for (int count = 0; count < this.windowRowSize * this.windowColumnSize; count++) {
                w[count] = toSlice.getDouble(count + window);
            }
            INDArray add = Nd4j.create(w);
            if (flattened)
                add = add.ravel();
            else
                add = add.reshape(windowRowSize, windowColumnSize);
            if (addRotate) {
                INDArray currRotation = add.dup();
                //3 different orientations besides the original
                for (int rotation = 0; rotation < 3; rotation++) {
                    Nd4j.rot90(currRotation);
                    ret.add(currRotation.dup());
                }

            }

            window += this.windowRowSize * this.windowColumnSize;
            ret.add(add);
        }


        return ret;
    }
}
