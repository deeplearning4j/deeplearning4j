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
 *
 */

package org.nd4j.linalg.api.ops;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LinAlgExceptions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * A base op for basic getters and setters
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseTransformOp extends BaseOp implements TransformOp {


    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2) {
        this(sameDiff,i_v1,i_v2,false);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2,
                           boolean inPlace) {
        super(sameDiff,inPlace,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);
            this.sameDiff = sameDiff;
            this.inPlace = inPlace;
            this.xVertexId = i_v1.getVarName();
            this.yVertexId = i_v2.getVarName();
            sameDiff.addArgsFor(new SDVariable[]{i_v1,i_v2},this);
            if(i_v1.getShape() != null)
                this.n = ArrayUtil.prod(i_v1.getShape());


        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }


    }

    public BaseTransformOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v1,
                           SDVariable i_v2,
                           Object[] extraArgs) {
        super(sameDiff,extraArgs);
        if (i_v1 != null && i_v2 != null) {

            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);
            this.sameDiff = sameDiff;
            this.xVertexId = i_v1.getVarName();
            this.yVertexId = i_v2.getVarName();
            sameDiff.addArgsFor(new SDVariable[]{i_v1,i_v2},this);
            if(i_v1.getShape() != null)
                this.n = ArrayUtil.prod(i_v1.getShape());

        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }

    }




    public BaseTransformOp(SameDiff sameDiff,SDVariable i_v,boolean inPlace) {
        this(sameDiff,i_v,i_v.getShape(),inPlace,null);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v,
                           int[] shape,
                           boolean inPlace,
                           Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);

        if (i_v != null) {
            f().validateDifferentialFunctionsameDiff(i_v);
            this.xVertexId = i_v.getVarName();
            sameDiff.addArgsFor(new SDVariable[]{i_v},this);
            if(i_v.getShape() != null)
                this.n = ArrayUtil.prod(i_v.getShape());

        } else {
            throw new IllegalArgumentException("Input must not null variable.");
        }

    }


    public BaseTransformOp(SameDiff sameDiff,
                           SDVariable i_v,
                           Object[] extraArgs) {
        this(sameDiff,i_v,i_v.getShape(),false,extraArgs);

    }



    public BaseTransformOp(INDArray x, INDArray z) {
        super(x, z);
        LinAlgExceptions.assertSameLength(x, z);
        LinAlgExceptions.assertSameShape(x, z);
    }

    public BaseTransformOp() {}

    public BaseTransformOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public BaseTransformOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        if (y != null)
            LinAlgExceptions.assertSameLength(x, y);
        LinAlgExceptions.assertSameLength(x, z);

    }

    public BaseTransformOp(INDArray x) {
        super(x);
    }

    @Override
    public Type opType() {
        if(args() == null || args().length == 1)
            return Type.TRANSFORM;
        else if(args().length == 2)
            return Type.PAIRWISE;

        else throw new ND4JIllegalStateException("Illegal number of args (can only be 1 or 2)");
    }


    @Override
    public void initWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
        if(isArrayInit() || isArrayInitialized()) {
            return;
        }


        super.initWithArrays(arrayMap);



        val args = args();
        for(val arg : args) {
            arg.initWithArrays(arrayMap,extraArgs);
        }
    }


    @Override
    public void initOutputWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
        super.initOutputWithArrays(arrayMap, extraArgs);
        val vertexId = outputVariables()[0].getVarName();
        if(!sameDiff.shapeAlreadyExistsForVarName(vertexId) && sameDiff.getArrForVarName(vertexId) == null) {
            val shape = calculateOutputShape();
            if (shape.isEmpty() || shape.get(0) == null) {
                throw new ND4JIllegalStateException("Shape should not be null or empty");
            }

            sameDiff.putShapeForVarName(vertexId, shape.get(0));

        }



        if(!sameDiff.shapeAlreadyExistsForVarName(vertexId) && sameDiff.getArrForVarName(vertexId) == null) {
            val shape = calculateOutputShape();
            if (shape.isEmpty() || shape.get(0) == null) {
                throw new ND4JIllegalStateException("Shape should not be null or empty");
            }

            sameDiff.putShapeForVarName(vertexId, shape.get(0));

        }

        val args = args();
        val resultVertexId = outputVariables()[0].getVarName();
        if(sameDiff.getArrForVarName(vertexId) == null || x == null) {
            if(sameDiff.getArrForVarName(args[0].getVarName()) != null) {
                this.x = sameDiff.getArrForVarName(args[0].getVarName());
            }
            else
                throw new ND4JIllegalStateException("No input found for vertex id " + resultVertexId + " and op " + opName());
            if(args().length  > 1) {
                if(sameDiff.getArrForVarName(args[1].getVarName()) != null) {
                    this.y = sameDiff.getArrForVarName(args[1].getVarName());
                }

                else
                    throw new ND4JIllegalStateException("No second input found for vertex id " + resultVertexId + " and op " + opName());

            }
        }

        arrayInitialized = true;

        val outputFunctions = outputVariables();
       /*
        for(val arg : outputVariables) {
            arg.initOutputWithArrays(arrayMap,extraArgs);
        }
*/
        if(sameDiff.getArrForVarName(vertexId) == null || z == null) {
            if(sameDiff.getArrForVarName(args[0].getVarName()) != null) {
                this.z = sameDiff.getArrForVarName(outputFunctions[0].getVarName());
            }
            else
                throw new ND4JIllegalStateException("No input found for vertex id " + outputVariables()[0].getVarName() + " and op " + opName());
        }
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        if(arg() == null)
            throw new ND4JIllegalStateException("No arg found for op!");

        val arr = sameDiff.getArrForVarName(arg().getVarName());
        if(arr == null)
            return Collections.emptyList();
        ret.add(arr.shape());
        this.n = arr.length();
        return ret;
    }



}
