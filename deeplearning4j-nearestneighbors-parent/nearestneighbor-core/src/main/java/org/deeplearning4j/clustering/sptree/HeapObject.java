/*-
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

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
@Data
public class HeapObject implements Serializable, Comparable<HeapObject> {
    private int index;
    private INDArray point;
    private double distance;


    public HeapObject(int index, INDArray point, double distance) {
        this.index = index;
        this.point = point;
        this.distance = distance;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        HeapObject heapObject = (HeapObject) o;

        if (!point.equals(heapObject.point))
            return false;

        return Double.compare(heapObject.distance, distance) == 0;

    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = index;
        temp = Double.doubleToLongBits(distance);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public int compareTo(HeapObject o) {
        return distance < o.distance ? 1 : 0;
    }
}
