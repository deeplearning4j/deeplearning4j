/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.clustering.sptree;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 *
 * A vector with an index and function for distance
 * @author Adam Gibson
 */
public class DataPoint implements Serializable {
    private int index;
    private INDArray point;
    private int d;
    private String functionName;
    public DataPoint(int index, INDArray point) {
       this(index,point,"euclidean");
    }
    public DataPoint(int index, INDArray point,String functionName) {
        this.index = index;
        this.point = point;
        this.functionName = functionName;
        this.d = point.length();
    }

    /**
     * Euclidean distance
     * @param point the distance from this point to the given point
     * @return the distance distance between the two points
     */
    public double distance(DataPoint point) {
        switch (functionName) {
            case "euclidean" :     return Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(this.point,point.point)).currentResult().doubleValue();
            case "cosinesimilarity" :     return Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(this.point,point.point)).currentResult().doubleValue();
            case "manhattan" :     return Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(this.point,point.point)).currentResult().doubleValue();
            default: return Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(this.point,point.point)).currentResult().doubleValue();

        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        DataPoint dataPoint = (DataPoint) o;

        if (index != dataPoint.index) return false;
        return !(point != null ? !point.equals(dataPoint.point) : dataPoint.point != null);

    }

    @Override
    public int hashCode() {
        int result = index;
        result = 31 * result + (point != null ? point.hashCode() : 0);
        return result;
    }

    public int getD() {
        return d;
    }

    public void setD(int d) {
        this.d = d;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public INDArray getPoint() {
        return point;
    }

    public void setPoint(INDArray point) {
        this.point = point;
    }
}
