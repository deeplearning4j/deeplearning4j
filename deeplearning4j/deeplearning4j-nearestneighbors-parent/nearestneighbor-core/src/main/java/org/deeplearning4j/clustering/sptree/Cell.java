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

package org.deeplearning4j.clustering.sptree;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public class Cell implements Serializable {
    private int dimension;
    private INDArray corner, width;

    public Cell(int dimension) {
        this.dimension = dimension;
    }

    public double corner(int d) {
        return corner.getDouble(d);
    }

    public double width(int d) {
        return width.getDouble(d);
    }

    public void setCorner(int d, double corner) {
        this.corner.putScalar(d, corner);
    }

    public void setWidth(int d, double width) {
        this.width.putScalar(d, width);
    }

    public void setWidth(INDArray width) {
        this.width = width;
    }

    public void setCorner(INDArray corner) {
        this.corner = corner;
    }


    public boolean contains(INDArray point) {
        INDArray cornerMinusWidth = corner.sub(width);
        INDArray cornerPlusWidth = corner.add(width);
        for (int d = 0; d < dimension; d++) {
            if (cornerMinusWidth.getDouble(d) > point.getDouble(d))
                return false;
            if (cornerPlusWidth.getDouble(d) < point.getDouble(d))
                return false;
        }
        return true;

    }

    public INDArray width() {
        return width;
    }

    public INDArray corner() {
        return corner;
    }



}
