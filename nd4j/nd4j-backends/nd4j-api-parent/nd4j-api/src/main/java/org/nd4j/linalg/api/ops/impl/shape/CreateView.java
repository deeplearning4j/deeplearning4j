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
package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class CreateView extends DynamicCustomOp  {

    public static int POINT_TYPE = 0;
    public static int INTERVAL_TYPE = 1;
    public static int ALL_TYPE = 2;
    public static int NEW_AXIS = 3;

    public static int DEFAULT_INCLUSIVE = 1;

    public final static String OP_NAME = "create_view";


    public CreateView() {
    }

    public CreateView(INDArray[] inputs) {
        super(inputs, null);
    }

    public CreateView(SameDiff sameDiff, SDVariable[] args) {
        super(sameDiff, args);
    }

    public CreateView(SameDiff sd, SDVariable input, SDVariable[] indices) {
        this(sd, ArrayUtil.combine(new SDVariable[]{input},indices));
    }

    public CreateView(INDArray input, INDArray[] indices) {
        this(ArrayUtil.combine(new INDArray[]{input},indices));
    }

    public static SDVariable createInterval(SameDiff sameDiff, SDVariable intervalInputBegin, SDVariable intervalInputEnd, SDVariable intervalStrideInput, SDVariable inclusive) {
        return createInterval(sameDiff,null,intervalInputBegin,intervalInputEnd,intervalStrideInput,inclusive);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes) {
        //Output type is same as input type
        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return new CreateView(sameDiff,f1.get(0),Arrays.copyOfRange(args(),1,args().length)).outputs();
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    @Override
    public String opName() {
        return "create_view";
    }

    /**
     *  See
     * {@link #createPoint(SameDiff,String,long)}
     * for mroe information.
     * @param sameDiff
     * @param offset
     * @return
     */
    public static SDVariable createPoint(SameDiff sameDiff,long offset) {
        return createPoint(sameDiff,null,offset);
    }

    /**
     *  See
     * {@link #createPoint(SameDiff,String,long)}
     * for mroe information.
     * @param sameDiff
     * @param offset
     * @return
     */
    public static SDVariable createPoint(SameDiff sameDiff,SDVariable offset) {
        return createPoint(sameDiff,null,offset);
    }


    /**
     * Create a {@link SDVariable}
     * representing a point array with the specified name.
     * This is used for specifying the equivalent of a
     * {@link org.nd4j.autodiff.samediff.SDIndex#point(SDVariable)}
     * @param sameDiff the samediff instance fo use
     * @param name the name of the variable (maybe null)
     * @param offset the offset for the point
     * @return the created variable
     */
    public static SDVariable createPoint(SameDiff sameDiff,String name,long offset) {
        INDArray arr = Nd4j.createFromArray(new long[]{POINT_TYPE,1,1,offset, DEFAULT_INCLUSIVE});
        return sameDiff.var(name,arr);
    }


    /**
     * Create a {@link SDVariable}
     * representing a point array with the specified name.
     * This is used for specifying the equivalent of a
     * {@link org.nd4j.autodiff.samediff.SDIndex#point(SDVariable)}
     * @param sameDiff the samediff instance fo use
     * @param name the name of the variable (maybe null)
     * @param offset the offset for the point
     * @return the created variable
     */
    public static SDVariable createPoint(SameDiff sameDiff,String name,SDVariable offset) {
        return sameDiff.concat(name,0,
                sameDiff.constant(POINT_TYPE).reshape(1).castTo(DataType.INT64),
                sameDiff.constant(1).reshape(1).castTo(DataType.INT64),
                sameDiff.constant(1).reshape(1).castTo(DataType.INT64),
                offset.reshape(1).castTo(DataType.INT64),
                sameDiff.constant(DEFAULT_INCLUSIVE).reshape(1).castTo(DataType.INT64));
    }



    /**
     * See {@link #createAll(SameDiff, String)}
     * for more information
     * @param sameDiff
     * @return
     */
    public static SDVariable createAll(SameDiff sameDiff) {
        return createAll(sameDiff,null);
    }
    /**
     * Create an {@link SDVariable}
     * representing an {@link SDIndex#all()}
     * variable.
     * @param sameDiff the samediff instance to use
     * @param name the name of the variable (maybe null)
     * @return the created variable
     */
    public static SDVariable createAll(SameDiff sameDiff,String name) {
        INDArray arr = Nd4j.createFromArray(new long[]{ALL_TYPE,0,1, DEFAULT_INCLUSIVE});
        return sameDiff.var(name,arr);
    }

    /**
     * Create an {@link SDVariable}
     * representing a new axis which creates a new index
     * of length 1 in the specified input
     * @param sameDiff the samediff instance to use
     * @param name the name of the variable
     * @return the created variable
     */
    public static SDVariable createNewAxis(SameDiff sameDiff,String name) {
        INDArray arr = Nd4j.createFromArray(new long[]{NEW_AXIS,1,10, DEFAULT_INCLUSIVE});
        return sameDiff.var(name,arr);
    }

    /**
     * See {@link #createNewAxis(SameDiff,String)}
     * for more information.
     * @param sameDiff
     * @return
     */
    public static SDVariable createNewAxis(SameDiff sameDiff) {
        return createNewAxis(sameDiff,null);
    }

    /**
     * Create an interval representing {@link SDIndex#interval(Long, Long)}
     *
     * @param sameDiff the samediff instance to use
     * @param name the name of the variable
     * @param start the start of the interval
     * @param end the end of the interval
     * @param stride the stride
     * @param inclusive whether the interval is inclusive or not 0 for false 1 for true
     * @return
     */
    public static SDVariable createInterval(SameDiff sameDiff,String name,long start,long end,long stride,long inclusive) {
        INDArray arr = Nd4j.createFromArray(new long[]{INTERVAL_TYPE,2,1,start,end,stride,inclusive});
        return sameDiff.var(name,arr);
    }


    /**
     * Create an interval representing {@link SDIndex#interval(Long, Long)}
     *
     * @param sameDiff the samediff instance to use
     * @param name the name of the variable
     * @param start the start of the interval
     * @param end the end of the interval
     * @param stride the stride
     * @param inclusive whether the interval is inclusive or not 0 for false 1 for true
     * @return
     */
    public static SDVariable createInterval(SameDiff sameDiff,String name,SDVariable start,SDVariable end,SDVariable stride,SDVariable inclusive) {
       if(stride == null)
           stride = sameDiff.constant(1).castTo(DataType.INT64).reshape(1);
       if(inclusive == null)
           inclusive = sameDiff.constant(0).castTo(DataType.INT64).reshape(1);
        return sameDiff.concat(name,0,
                sameDiff.constant(INTERVAL_TYPE).reshape(1).castTo(DataType.INT64),
                sameDiff.constant(2).reshape(1).castTo(DataType.INT64),
                sameDiff.constant(1).reshape(1).castTo(DataType.INT64),
                start.reshape(1).castTo(DataType.INT64)
                ,end.reshape(1).castTo(DataType.INT64),
                stride.reshape(1).castTo(DataType.INT64),
                inclusive.castTo(DataType.INT64).reshape(1));
    }


    /**
     * See {@link #createInterval(SameDiff, String, long, long, long, long)}
     * for more information.
     * @param sameDiff
     * @param start
     * @param end
     * @param stride
     * @param inclusive
     * @return
     */
    public static SDVariable createInterval(SameDiff sameDiff,long start,long end,long stride,long inclusive) {
        return createInterval(sameDiff,null,start,end,stride,inclusive);
    }


    public static INDArray createFrom(INDArray input,INDArray...indices) {
       return input.get(indices(indices));
    }



   public static INDArrayIndex[] indices(INDArray...indexArrs) {
        return Arrays.stream(indexArrs).map(CreateView::fromIndexArr).collect(Collectors.toList())
                .toArray(new INDArrayIndex[indexArrs.length]);
   }
    public static INDArrayIndex fromIndexArr(INDArray index) {
        int idx = index.getInt(0);
        if(idx == POINT_TYPE) {
            int getPoint = index.getInt(3);
            return NDArrayIndex.point(getPoint);
        } else if(idx == INTERVAL_TYPE) {
            int start = index.getInt(3);
            int end = index.getInt(4);
            int stride = index.getInt(5);
            boolean inclusive =  index.getInt(6) > 0;
            return NDArrayIndex.interval(start,stride,end,inclusive);
        } else if(idx == NEW_AXIS) {
            return NDArrayIndex.newAxis();
        } else if(idx == ALL_TYPE) {
            return NDArrayIndex.all();
        } else {
            throw new IllegalArgumentException("Invalid type. Must be 1 of: " + POINT_TYPE + " (point type) " + INTERVAL_TYPE + " (interval type)" + NEW_AXIS + " (new axis) " + ALL_TYPE + " (all) ");
        }
    }

}
