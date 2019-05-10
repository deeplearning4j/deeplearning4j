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

package org.deeplearning4j.clustering.quadtree;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * A cell representing a bounding box forthe quad tree
 * @author Adam Gibson
 */
public class Cell implements Serializable {
    private double x, y, hw, hh;

    public Cell(double x, double y, double hw, double hh) {
        this.x = x;
        this.y = y;
        this.hw = hw;
        this.hh = hh;
    }

    /**
     * Whether the given point is contained
     * within this cell
     * @param point the point to check
     * @return true if the point is contained, false otherwise
     */
    public boolean containsPoint(INDArray point) {
        double first = point.getDouble(0), second = point.getDouble(1);
        return x - hw <= first && x + hw >= first && y - hh <= second && y + hh >= second;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof Cell))
            return false;

        Cell cell = (Cell) o;

        if (Double.compare(cell.hh, hh) != 0)
            return false;
        if (Double.compare(cell.hw, hw) != 0)
            return false;
        if (Double.compare(cell.x, x) != 0)
            return false;
        return Double.compare(cell.y, y) == 0;

    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        temp = Double.doubleToLongBits(x);
        result = (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(y);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(hw);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(hh);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getHw() {
        return hw;
    }

    public void setHw(double hw) {
        this.hw = hw;
    }

    public double getHh() {
        return hh;
    }

    public void setHh(double hh) {
        this.hh = hh;
    }


}
