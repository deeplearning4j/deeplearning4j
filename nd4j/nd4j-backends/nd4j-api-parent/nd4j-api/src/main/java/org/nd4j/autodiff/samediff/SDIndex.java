/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff;
import lombok.Getter;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;


/**
 * SDIndex is the {@link SameDiff}
 * equivalent to {@link org.nd4j.linalg.indexing.INDArrayIndex}
 * it uses {@link org.nd4j.linalg.api.ops.impl.shape.StridedSlice} underneath to obtain varying elements.
 * It also supports {@link SDVariable} inputs allowing for graph definitions of
 * indexing operations.
 *
 * @author Alex Black
 * @author Adam Gibson
 */
@Getter
public class SDIndex {

    /**
     * Index types include the following:
     * 1. all: get all elements of this dimension
     * 2. point: get only elements at the particular point in this dimension
     * 3. interval: get only elements from a begin point to an end point in the interval
     * 4. point input: dynamic version of point
     * 5. intervar input: dynamic version of interval
     */
    public enum IndexType {
        ALL,
        POINT,
        INTERVAL,
        //inputs aren't integers/longs but SDVariables
        POINT_INPUT,
        INTERVAL_INPUT
    }

    private IndexType indexType = IndexType.ALL;
    private long pointIndex;

    private SDVariable pointVar;


    private boolean pointKeepDim;
    private Long intervalBegin = null;
    private Long intervalEnd = null;


    private SDVariable intervalInputBegin = null;
    private SDVariable intervalInputEnd = null;
    private SDVariable intervalStrideInput = null;

    private Long intervalStrides = 1l;

    private boolean inclusive = false;

    private SDVariable inclusiveInput = null;


    public SDIndex(){}




    /**
     * Represents all the elements in along this dimension.
     * @return
     */
    public static SDIndex all(){
        return new SDIndex();
    }

    /**
     * Represents all elements at a singular point in this dimension (think row or column)
     * Note this is the SDVariable version. For static please use {@link #point(long)}
     * @param i the input index
     * @return
     */
    public static SDIndex point(SDVariable i) {
        return point(i,false);
    }

    /**
     * Represents all elements at a singular point in this dimension (think row or column)
     * This is a static index
     * @param i the input index
     * @return
     */
    public static SDIndex point(long i) {
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.POINT;
        sdIndex.pointIndex = i;
        sdIndex.pointKeepDim = false;
        return sdIndex;
    }

    /**
     * Represents all elements at a singular point in this dimension (think row or column)
     * This is a dynamic index
     * @param i the input index
     * @return
     */
    public static SDIndex point(SDVariable i, boolean keepDim) {
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.POINT_INPUT;
        sdIndex.pointVar = i;
        sdIndex.pointKeepDim = keepDim;
        return sdIndex;
    }

    /**
     * Represents all elements at a singular point in this dimension (think row or column)
     * This is a static index
     * @param i the input index
     * @return
     */
    public static SDIndex point(long i, boolean keepDim) {
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.POINT;
        sdIndex.pointIndex = i;
        sdIndex.pointKeepDim = keepDim;
        return sdIndex;
    }


    /**
     *  Represents all elements begin to end (think get row from beginning to end)
     *  Note these are dynamic indices.
     * @param begin the begin index
     * @param end the end index
     * @return
     */
    public static SDIndex interval(SDVariable begin, SDVariable end) {
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL_INPUT;
        sdIndex.intervalInputBegin = begin;
        sdIndex.intervalInputEnd = end;
        sdIndex.inclusiveInput = begin.getSameDiff().constant(0);
        return sdIndex;
    }

    /**
     *  Represents all elements begin to end (think get row from beginning to end)
     *  Note these are static indices.
     * @param begin the begin index
     * @param end the end index
     * @return
     */
    public static SDIndex interval(Long begin, Long end) {
        return interval(begin,end,false);
    }

    /**
     *  Represents all elements begin to end (think get row from beginning to end)
     *  Note these are static indices.
     * @param begin the begin index
     * @param end the end index
     * @return
     */
    public static SDIndex interval(Long begin, Long end,Boolean inclusive) {
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        if(begin != null) {
            sdIndex.intervalBegin = begin.longValue();
        }
        if(end != null) {
            sdIndex.intervalEnd = end.longValue();
        }

        if(inclusive != null) {
            sdIndex.inclusive = inclusive;
        } else {
            sdIndex.inclusive = false;
        }

        return sdIndex;
    }


    /**
     *  Represents all elements begin to end (think get row from beginning to end)
     *  Note these are static indices.
     * @param begin the begin index
     * @param end the end index
     * @return
     */
    public static SDIndex interval(Integer begin, Integer end) {
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        if(begin != null) {
            sdIndex.intervalBegin = begin.longValue();
        }
        if(end != null){
            sdIndex.intervalEnd = end.longValue();
        }

        sdIndex.inclusive = false;

        return sdIndex;
    }

    /**
     *  Represents all elements begin to end (think get row from beginning to end)
     *  Note these are static indices.
     * @param begin the begin index
     * @param strides the stride to increment by to end
     * @param end the end index
     * @return
     */
    public static SDIndex interval(Long begin, Long strides, Long end) {
        if(strides == 0){
            throw new ND4JIllegalArgumentException("Invalid index : strides can not be 0.");
        }
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        sdIndex.intervalBegin = begin;
        sdIndex.intervalEnd = end;
        sdIndex.intervalStrides = strides;
        sdIndex.inclusive  = false;
        return sdIndex;
    }

    /**
     *  Represents all elements begin to end (think get row from beginning to end)
     *  Note these are static indices.
     * @param begin the begin index
     * @param strides the stride to increment by to end
     * @param end the end index
     * @param inclusive whether the index is inclusive or not
     * @return
     */
    public static SDIndex interval(Long begin, Long strides, Long end,Boolean inclusive) {
        if(strides == 0) {
            throw new ND4JIllegalArgumentException("Invalid index : strides can not be 0.");
        }

        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        sdIndex.intervalBegin = begin;
        sdIndex.intervalEnd = end;
        sdIndex.intervalStrides = strides;
        if(inclusive != null) {
            sdIndex.inclusive = inclusive;
        } else {
            sdIndex.inclusive = false;
        }
        return sdIndex;
    }


    /**
     *  Represents all elements begin to end (think get row from beginning to end)
     *  Note these are static indices.
     * @param begin the begin index
     * @param strides the stride to increment by to end
     * @param end the end index
     * @return
     */
    public static SDIndex interval(Integer begin, Integer strides, Integer end) {
        return interval(begin.longValue(),strides.longValue(),end.longValue());
    }

    /**
     *  Represents all elements begin to end (think get row from beginning to end)
     *  Note these are static indices.
     * @param begin the begin index
     * @param strides the stride to increment by to end
     * @param end the end index
     * @return
     */
    public static SDIndex interval(SDVariable begin, SDVariable strides, SDVariable end) {
      return interval(begin,strides,end,begin.getSameDiff().constant(false));
    }

    /**
     *  Represents all elements begin to end (think get row from beginning to end)
     *  Note these are static indices.
     * @param begin the begin index
     * @param strides the stride to increment by to end
     * @param end the end index
     * @return
     */
    public static SDIndex interval(SDVariable begin, SDVariable strides, SDVariable end,SDVariable inclusive) {
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL_INPUT;
        if(begin != null) {
            sdIndex.intervalInputBegin = begin;
        }

        if(end != null) {
            sdIndex.intervalInputEnd = end;
        }

        if(strides != null) {
            sdIndex.intervalStrideInput = strides;
        }

        if(inclusive != null) {
            sdIndex.inclusiveInput = inclusive;
        } else {
            sdIndex.inclusiveInput = begin.getSameDiff().constant(false);
        }

        return sdIndex;
    }
}
