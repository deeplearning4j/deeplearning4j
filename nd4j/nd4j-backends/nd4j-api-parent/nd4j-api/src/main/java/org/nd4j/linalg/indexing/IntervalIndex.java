/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.linalg.indexing;

import org.nd4j.shade.guava.primitives.Longs;
import lombok.Getter;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * And indexing representing
 * an interval
 *
 * @author Adam Gibson
 */
public class IntervalIndex implements INDArrayIndex {

    protected long begin, end;
    @Getter
    protected boolean inclusive;
    protected long stride = 1;
    protected long index = 0;
    protected long length = 0;

    /**
     *
     * @param inclusive whether to include the last number
     * @param stride the stride for the interval
     */
    public IntervalIndex(boolean inclusive, long stride) {
        this.inclusive = inclusive;
        this.stride = stride;
    }

    @Override
    public long end() {
        return end;
    }

    @Override
    public long offset() {
        return begin;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public long stride() {
        return stride;
    }

    @Override
    public void reverse() {
        long oldEnd = end;
        long oldBegin = begin;
        this.end = oldBegin;
        this.begin = oldEnd;
    }

    @Override
    public boolean isInterval() {
        return true;
    }

    @Override
    public void init(INDArray arr, long begin, int dimension) {
        if(begin < 0) {
            begin +=  arr.size(dimension);
        }

        this.begin = begin;
        this.index = begin;
        this.end = inclusive ? arr.size(dimension) + 1 : arr.size(dimension);

        //Calculation of length: (endInclusive - begin)/stride + 1
        long endInc = arr.size(dimension) - (inclusive ? 0 : 1);
        this.length = (endInc - begin)/stride + 1;

        Preconditions.checkState(endInc < arr.size(dimension), "Invalid interval: %s on array with shape %ndShape", this, arr);
    }

    @Override
    public void init(INDArray arr, int dimension) {
        init(arr, 0, dimension);
    }


    @Override
    public void init(long begin, long end, long max) {
        if(begin < 0) {
            begin +=  max;
        }

        if(end < 0) {
            end +=  max;
        }
        this.begin = begin;
        this.index = begin;
        this.end = end;

        long endInc = end - (inclusive ? 0 : 1);
        this.length = (endInc - begin)/stride + 1;
    }

    @Override
    public void init(long begin, long end) {
        if(begin < 0 || end < 0)
            throw new IllegalArgumentException("Please pass in an array for negative indices. Unable to determine size for dimension otherwise");
        this.begin = begin;
        this.index = begin;
        this.end = end;

        long endInc = end - (inclusive ? 0 : 1);
        this.length = (endInc - begin)/stride + 1;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof IntervalIndex))
            return false;

        IntervalIndex that = (IntervalIndex) o;

        if (begin != that.begin)
            return false;
        if (end != that.end)
            return false;
        if (inclusive != that.inclusive)
            return false;
        if (stride != that.stride)
            return false;
        return index == that.index;

    }

    @Override
    public int hashCode() {
        int result = Longs.hashCode(begin);
        result = 31 * result + Longs.hashCode(end);
        result = 31 * result + (inclusive ? 1 : 0);
        result = 31 * result + Longs.hashCode(stride);
        result = 31 * result + Longs.hashCode(index);
        return result;
    }

    @Override
    public String toString(){
        return "Interval(b=" + begin + ",e=" + end + ",s=" + stride + (inclusive ? ",inclusive" : "") + ")";
    }
}
