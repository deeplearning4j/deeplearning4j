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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.LinAlgExceptions;

import java.util.ArrayList;
import java.util.Arrays;
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
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2) {
        this(sameDiff,i_v1,i_v2,false);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           boolean inPlace) {
        super(sameDiff,inPlace,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);
            f().validateFunctionReference(i_v1);
            f().validateFunctionReference(i_v2);
            this.sameDiff = sameDiff;
            this.inPlace = inPlace;
            addAsNewVertexId();
            sameDiff.putShapeForVertexId(vertexId,i_v1.getResultShape());
            sameDiff.associateFunctionsAsArgs(new DifferentialFunction[] {i_v1,i_v2},this);
            f().addFunctionEdges(this);

        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }


    }

    public BaseTransformOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public BaseTransformOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           Object[] extraArgs) {
        super(sameDiff,extraArgs);
        if (i_v1 != null && i_v2 != null) {

            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);
            this.sameDiff = sameDiff;
            addAsNewVertexId();
            sameDiff.putShapeForVertexId(vertexId,i_v1.getResultShape());
            sameDiff.associateFunctionsAsArgs(new DifferentialFunction[] {i_v1,i_v2},this);

            f().addFunctionEdges(this);

        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }

    }




    public BaseTransformOp(SameDiff sameDiff,DifferentialFunction i_v,boolean inPlace) {
        this(sameDiff,i_v,i_v.getResultShape(),inPlace,null);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           DifferentialFunction i_v,
                           int[] shape,
                           boolean inPlace,
                           Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        ;
        if (i_v != null) {
            f().validateFunctionReference(i_v);
            f().validateDifferentialFunctionsameDiff(i_v);
            addAsNewVertexId();
            sameDiff.putShapeForVertexId(vertexId,shape);
            sameDiff.associateFunctionsAsArgs(new DifferentialFunction[] {sameDiff.setupFunction(i_v)},this);
            f().addFunctionEdges(this);
        } else {
            throw new IllegalArgumentException("Input must not null variable.");
        }

    }


    public BaseTransformOp(SameDiff sameDiff,
                           DifferentialFunction i_v,
                           Object[] extraArgs) {
        this(sameDiff,i_v,i_v.getResultShape(),false,extraArgs);
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

        if(!sameDiff.shapeAlreadyExistsForVertexId(vertexId) && sameDiff.getArrForVertexId(vertexId) == null) {
            val shape = calculateOutputShape();
            if (shape.isEmpty() || shape.get(0) == null) {
                throw new ND4JIllegalStateException("Shape should not be null or empty");
            }

            sameDiff.putShapeForVertexId(vertexId, shape.get(0));

        }



        if(!sameDiff.shapeAlreadyExistsForVertexId(vertexId) && sameDiff.getArrForVertexId(vertexId) == null) {
            val shape = calculateOutputShape();
            if (shape.isEmpty() || shape.get(0) == null) {
                throw new ND4JIllegalStateException("Shape should not be null or empty");
            }

            sameDiff.putShapeForVertexId(vertexId, shape.get(0));

        }

        val args = args();
        if(sameDiff.getArrForVertexId(vertexId) == null || x == null) {
            if(sameDiff.getArrForVertexId(args[0].resultVertexId()) != null) {
                this.x = sameDiff.getArrForVertexId(args[0].resultVertexId());
            }
            else
                throw new ND4JIllegalStateException("No input found for vertex id " + Arrays.toString(resultVertexId()) + " and op " + opName());
            if(args().length  > 1) {
                if(sameDiff.getArrForVertexId(args[1].resultVertexId()) != null) {
                    this.y = sameDiff.getArrForVertexId(args[1].resultVertexId());
                }

                else
                    throw new ND4JIllegalStateException("No second input found for vertex id " + Arrays.toString(resultVertexId()) + " and op " + opName());

            }
        }

        arrayInitialized = true;

        val outputFunctions = outputFunctions();
       /*
        for(val arg : outputFunctions) {
            arg.initOutputWithArrays(arrayMap,extraArgs);
        }
*/
        if(sameDiff.getArrForVertexId(vertexId) == null || z == null) {
            if(sameDiff.getArrForVertexId(args[0].resultVertexId()) != null) {
                this.z = sameDiff.getArrForVertexId(outputFunctions[0].resultVertexId());
            }
            else
                throw new ND4JIllegalStateException("No input found for vertex id " + Arrays.toString(resultVertexId()) + " and op " + opName());
        }
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        val arr = sameDiff.getArrForVertexId(arg().resultVertexId());
        if(arr == null)
            throw new ND4JIllegalStateException("Array must not be null for argument!");
        ret.add(arr.shape());
        return ret;
    }



}
