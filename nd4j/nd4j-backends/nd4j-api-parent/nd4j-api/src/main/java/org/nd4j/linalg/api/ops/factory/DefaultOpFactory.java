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

package org.nd4j.linalg.api.ops.factory;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.accum.StandardDeviation;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.impl.scalar.Pow;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.Step;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftMaxDerivative;

import java.lang.reflect.Constructor;


/**
 * Default operations factory
 *
 * @author Adam Gibson
 */
@Slf4j
public class DefaultOpFactory implements OpFactory {


    public DefaultOpFactory() {
    }


    @Override
    public GradientOp createGradientOp(String name, INDArray x, INDArray y, INDArray z) {
        switch(name) {
            case "softmaxderivative":
                return new SoftMaxDerivative(x,y,z);
            case "sigmoidderivative":
                return new org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative(x,y,z);
            case "tanhderivative":
                return new org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative(x,y,z);
            case "gradientbackwards":
                return new org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker(x,y,z);
            default: throw new IllegalStateException("Illegal opName " + name);
        }
    }

    /**
     *
     * @param name
     * @param x
     * @param z
     * @return
     */
    @Override
    public Op createShape(String name, INDArray x, INDArray z, Object[] extraArgs) {
        throw new IllegalArgumentException("Illegal opName for create shape op" + name);
    }

    @Override
    public LossFunction createLossFunction(String name, INDArray x, INDArray y) {
        try {
            Constructor<DifferentialFunction> constructor =
                    (Constructor<DifferentialFunction>)  DifferentialFunctionClassHolder.getInstance().getInstance(name).getClass().getDeclaredConstructor(INDArray.class, INDArray.class);
            Op create = (Op) constructor.newInstance(x, y);
            return (LossFunction) create;
        } catch (Exception e) {
            throw new IllegalArgumentException("Illegal op " + name);
        }

    }

    @Override
    public ReduceOp createAccum(String name, INDArray x) {
        return createAccum(name,x,null,x,null);
    }

    @Override
    public ReduceOp createAccum(String name, INDArray x, INDArray y, INDArray z) {
        return createAccum(name,x,y,z,null);
    }


    @Override
    public ReduceOp createAccum(String name,
                                INDArray x,
                                INDArray y,
                                INDArray z,
                                Object[] extraArgs) {

        ReduceOp ret = null;

        switch (name) {
            case "mmul":
            case "std":
                ret = new StandardDeviation(x, y,z, x.length(),(boolean) extraArgs[0]);
                break;
            case "var":
                ret = new Variance(x, y, z, x.length(),(boolean) extraArgs[0]);
                break;
            default:
                try {
                    ret = (ReduceOp)  DifferentialFunctionClassHolder.getInstance().getInstance(name).getClass().getConstructor(INDArray.class, INDArray.class, INDArray.class, long.class).newInstance(x, y, z, x.length());
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
        }

        /*
        switch (opName) {

            case "sum":
                ret = new Sum(x, y, z,x.length());
                break;
            case "max":
                ret = new Max(x, y, z,x.length());
                break;
            case "min":
                ret = new Min(x, y, z,x.length());
                break;
            case "norm1":
                ret = new Norm1(x, y,z, x.length());
                break;
            case "norm2":
                ret = new Norm2(x, y,z, x.length());
                break;
            case "prod":
                ret = new Prod(x, y,z, x.length());
                break;
            case "std":
                ret = new StandardDeviation(x, y,z, x.length(),(boolean) extraArgs[0]);
                break;
            case "var":
                ret = new Variance(x, y,z, x.length(),(boolean) extraArgs[0]);
                break;
            case "euclidean":
                ret = new EuclideanDistance(x, y,z, x.length());
                break;
            case "cosine":
            case "cosinesimilarity":
                ret = new CosineSimilarity(x, y,z, x.length());
                break;
            case "cosinedistance":
                ret = new CosineDistance(x,y,z,x.lengthLong());
                break;
            case "manhattan":
                ret = new ManhattanDistance(x, y,z, x.length());
                break;
            case "mmul":
                //of note here is that it's always the last arg

                 * The case to watch out for here is
                 * tensor matrix multiply which has an args format of:
                 * dimensions, mmul transpose

                MMulTranspose mMulTranspose = extraArgs != null  && extraArgs.length >= 1 ? (MMulTranspose) extraArgs[extraArgs.length - 1] : MMulTranspose.allFalse();
                ret = new Mmul(x,y,z,mMulTranspose);
                break;
            case "tensorMmul":
                ret = new TensorMmul(x, y,z,(int[][]) extraArgs[0]);
                break;
        }
        */

        if(ret == null)
            throw new IllegalArgumentException("Illegal operation opName " + name);

        ret.setExtraArgs(extraArgs);
        return ret;
    }


    @Override
    public ReduceOp createAccum(String name, INDArray x, INDArray y) {
        return createAccum(name,x,y,x,null);
    }

    /**
     *
     * @param opName
     * @param x
     * @param y
     *@param z
     * @param extraArgs   @return
     */
    @Override
    public IndexAccumulation createIndexAccum(String opName, INDArray x, INDArray y, INDArray z, Object[] extraArgs) {
        IndexAccumulation ret = null;

        try {
            ret = (IndexAccumulation)  DifferentialFunctionClassHolder.getInstance().getInstance(opName).getClass().getConstructor(INDArray.class, INDArray.class).newInstance(x, y);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        /*
        switch (opName) {
            case "iamax":
                ret = new IAMax(x,y);
                break;
            case "imax":
                ret = new IMax(x,y);
                break;
            case "imin":
                ret = new IMin(x,y);
                break;
        }
        */

        ret.setExtraArgs(extraArgs);
        return ret;
    }

    @Override
    public IndexAccumulation createIndexAccum(String name, INDArray x) {
        return createIndexAccum(name,x,null , x, null);
    }

    @Override
    public IndexAccumulation createIndexAccum(String name, INDArray x, INDArray y) {
        return createIndexAccum(name,x,y,x,null);
    }

    @Override
    public TransformOp createTransform(String name, INDArray x, INDArray y) {
        return createTransform(name,x,y,x,null);

    }

    @Override
    public TransformOp createTransform(String name, INDArray x) {
        return createTransform(name,x,null,x,null);
    }

    @Override
    public TransformOp createTransform(String name, INDArray x, Object[] extraArgs) {
        return createTransform(name,x,null,x,extraArgs);
    }


    @Override
    public TransformOp createTransform(String name, INDArray x, INDArray y, INDArray z) {
        return createTransform(name,x,y,z,null);
    }

    /**
     * @param name
     * @param x
     * @param y
     * @param z
     * @param extraArgs
     * @return
     */
    @Override
    public TransformOp createTransform(String name,
                                       INDArray x,
                                       INDArray y,
                                       INDArray z,
                                       Object[] extraArgs) {
        TransformOp op = null;

        switch (name) {
            case "_softmaxderivative":
                op = new org.nd4j.linalg.api.ops.impl.transforms.strict.SoftMaxDerivative(x, z);
                break;
            case "set":
                op = new org.nd4j.linalg.api.ops.impl.transforms.Set(x,y,z,z.length());
                break;
            case "relu":
                op = new RectifedLinear(x, z, x.length(),extraArgs == null || extraArgs[0] == null ? 0.0 : (double) extraArgs[0]);
                break;
            case "step":
                op = new Step(x,y,z,x.length(),extraArgs == null || extraArgs[0] == null  ? 0.0 : (double) extraArgs[0]);
                break;
            case "pow":
                op = new Pow(x, z, (double) extraArgs[0]);
                break;
            default:
                try {
                    if (y == null)
                        op = (TransformOp)  DifferentialFunctionClassHolder.getInstance().getInstance(name).getClass().getConstructor(INDArray.class, INDArray.class).newInstance(x, z);
                    else
                        op = (TransformOp)  DifferentialFunctionClassHolder.getInstance().getInstance(name).getClass().getConstructor(INDArray.class, INDArray.class, INDArray.class).newInstance(x, y, z);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
        }

        /*

        switch (opName) {
            case "set":
                op = new org.nd4j.linalg.api.ops.impl.transforms.Set(x,y,z,z.length());
                break;
            case "relu":
                op = new RectifedLinear(x, z, x.length(),extraArgs == null || extraArgs[0] == null ? 0.0 : (double) extraArgs[0]);
                break;
            case "step":
                op = new Step(x,y,z,x.length(),extraArgs == null || extraArgs[0] == null  ? 0.0 : (double) extraArgs[0]);
                break;
            case "abs":
                op = new Abs(x, z);
                break;
            case "acos":
                op = new ACos(x, z);
                break;
            case "asin":
                op = new ASin(x, z);
                break;
            case "atan":
                op = new ATan(x, z);
                break;
            case "ceil":
                op = new Ceil(x, z);
                break;
            case "cos":
                op = new Cos(x, z);
                break;
            case "exp":
                op = new Exp(x, z);
                break;
            case "elu":
                op = new ELU(x, z);
                break;
            case "floor":
                op = new Floor(x, z);
                break;
            case "hardtanh":
                op = new HardTanh(x, z);
                break;
            case "hardsigmoid":
                op = new HardSigmoid(x, z);
                break;
            case "identity":
                op = new Identity(x, z);
                break;
            case "leakyrelu":
                op = new LeakyReLU(x, z);
                break;
            case "log":
                op = new Log(x, z);
                break;
            case "logsoftmax":
                op = new LogSoftMax(x, z);
                break;
            case "maxout":
                op = new MaxOut(x, z);
                break;
            case "negative":
                op = new Negative(x, z);
                break;
            case "pow":
                op = new Pow(x, z, (double) extraArgs[0]);
                break;
            case "round":
                op = new Round(x, z);
                break;
            case "sigmoid":
                op = new Sigmoid(x, z);
                break;
            case "sign":
                op = new Sign(x, z);
                break;
            case "sin":
                op = new Sin(x, z);
                break;
            case "softsign":
                op = new SoftSign(x, z);
                break;
            case "sqrt":
                op = new Sqrt(x, z);
                break;
            case "stabilize":
                op = new Stabilize(x, z, 1);
                break;
            case "tanh":
                op = new Tanh(x, z);
                break;
            case "rationaltanh":
                op = new RationalTanh(x, z);
                break;
            case "timesoneminus":
                op = new TimesOneMinus(x, z);
                break;
            case "softmaxderivative":
                op = new org.nd4j.linalg.api.ops.impl.transforms.strict.SoftMaxDerivative(x, z);
                break;
            case "softmax":
                op = new SoftMax(x, z);
                break;
            case "softplus":
                op = new SoftPlus(x, z);
                break;
            case "cube":
                op = new Cube(x, z);
                break;
            case "sigmoidderivative":
                op = new SigmoidDerivative(x,z);
                break;
            case "hard_sigmoidderivative":
                op = new HardSigmoidDerivative(x,z);
                break;
            case "hardtanhderivative":
                op = new HardTanhDerivative(x,z);
                break;
            case "tanhderivative":
                op = new TanhDerivative(x,z);
                break;
            case "leakyreluderivative":
                op = new LeakyReLUDerivative(x,z);
                break;
            case "mul":
                op = new MulOp(x,y,z);
                break;
            case "add":
                op = new AddOp(x,y,z);
                break;
            case "sub":
                op = new SubOp(x,y,z);
                break;
            case "div":
                op = new DivOp(x,y,z);
                break;
            case "rdiv":
                op = new RDivOp(x,y,z);
                break;
            case "rsub":
                op = new RSubOp(x,y,z);
                break;
            case "neg":
                op = new Negative(x,z);
                break;
            default:
                throw new ND4JIllegalStateException("No op found " + opName);
        }
*/


        op.setExtraArgs(extraArgs);
        return op;
    }

    /**
     * @param name
     * @param x
     * @param y
     * @param scalar
     * @return
     */
    @Override
    public ScalarOp createScalarTransform(String name, INDArray x, INDArray y, double scalar) {
        return createScalarTransform(name,x,y,x,null,scalar);
    }

    /**
     * @param name
     * @param x
     * @param scalar
     * @return
     */
    @Override
    public ScalarOp createScalarTransform(String name, INDArray x, double scalar) {
        return createScalarTransform(name,x,null,x,null,scalar);
    }

    /**
     * @param name
     * @param x
     * @param extraArgs
     * @param scalar
     * @return
     */
    @Override
    public ScalarOp createScalarTransform(String name,
                                          INDArray x,
                                          Object[] extraArgs,
                                          double scalar) {
        return createScalarTransform(name,x,null,x,null,scalar);
    }

    /**
     * @param name
     * @param x
     * @param y
     * @param z
     * @param scalar
     * @return
     */
    @Override
    public ScalarOp createScalarTransform(String name,
                                          INDArray x,
                                          INDArray y,
                                          INDArray z,
                                          double scalar) {
        return createScalarTransform(name,x,y,z,null,scalar);
    }

    /**
     * @param name
     * @param x
     * @param y
     * @param z
     * @param extraArgs
     * @param scalar
     * @return
     */
    @Override
    public ScalarOp createScalarTransform(String name,
                                          INDArray x,
                                          INDArray y,
                                          INDArray z,
                                          Object[] extraArgs,
                                          double scalar) {
        ScalarOp ret = null;

        try {
            ret = (ScalarOp)  DifferentialFunctionClassHolder.getInstance().getInstance(name).getClass().getConstructor(INDArray.class, INDArray.class, INDArray.class, long.class, Number.class).newInstance(x, y, z, x.length(), scalar);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        /*
        switch(opName) {
            case "add_scalar":
                ret = new ScalarAdd(x,y,z,x.length(),scalar);
                break;
            case "sub_scalar":
                ret = new ScalarSubtraction(x,y,z,x.length(),scalar);
                break;
            case "mul_scalar":
                ret = new ScalarMultiplication(x,y,z,x.length(),scalar);
                break;
            case "div_scalar":
                ret = new ScalarDivision(x,y,z,x.length(),scalar);
                break;
            case "equals_scalar":
                ret = new ScalarEquals(x,y,z,x.length(),scalar);
                break;
            case "notequals_scalar":
                ret = new ScalarNotEquals(x,y,z,x.length(),scalar);
                break;
            case "fmod_scalar":
                ret = new ScalarFMod(x,y,z,x.length(),scalar);
                break;
            case "max_scalar":
                ret = new ScalarMax(x,y,z,x.length(),scalar);
                break;
            case "min_scalar":
                ret = new ScalarMin(x,y,z,x.length(),scalar);
                break;
            case "greaterthan_scalar":
                ret = new ScalarGreaterThan(x,y,z,x.length(),scalar);
                break;
            case "greaterthanorequal_scalar":
                ret = new ScalarGreaterThanOrEqual(x,y,z,x.length(),scalar);
                break;
            case "lessthan_scalar":
                ret = new ScalarLessThan(x,y,z,x.length(),scalar);
                break;
            case "lessthanorequal_scalar":
                ret = new ScalarLessThanOrEqual(x,y,z,x.length(),scalar);
                break;
            case "remainder_scalar":
                ret = new ScalarRemainder(x,y,z,x.length(),scalar);
                break;
            case   "rdiv_scalar":
                ret = new ScalarReverseDivision(x,y,z,x.length(),scalar);
                break;
            case   "rsub_scalar":
                ret = new ScalarReverseSubtraction(x,y,z,x.length(),scalar);
                break;
        }

        */

        ret.setExtraArgs(extraArgs);
        return ret;
    }

    @Override
    public BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, INDArray z, int... dimension) {
        return createBroadcastOp(name,x,y,z,null,dimension);
    }

    @Override
    public BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, INDArray z, Object[] extraArgs, int... dimension) {
        BroadcastOp broadcastOp = null;

        try {
            broadcastOp = (BroadcastOp) DifferentialFunctionClassHolder.getInstance().getInstance(name).getClass().getConstructor(INDArray.class, INDArray.class, INDArray.class, int[].class).newInstance(x, y, z, dimension);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

 /*
        switch (opName) {
            case "broadcastadd":
                broadcastOp = new BroadcastAddOp(x, y, z, dimension);
                break;
            case "broadcastsub":
                broadcastOp = new BroadcastSubOp(x, y, z, dimension);
                break;
            case "broadcastmul":
                broadcastOp = new BroadcastMulOp(x, y, z, dimension);
                break;
            case "broadcastdiv":
                broadcastOp = new BroadcastDivOp(x, y, z, dimension);
                break;
            case "broadcastrsub":
                broadcastOp = new BroadcastRSubOp(x, y, z, dimension);
                break;
            case "broadcastrdiv":
                broadcastOp = new BroadcastRDivOp(x, y, z, dimension);
                break;
            case "broadcastcopy":
                broadcastOp = new BroadcastCopyOp(x, y, z, dimension);
                break;
        }
*/
        broadcastOp.setExtraArgs(extraArgs);
        return broadcastOp;
    }

    @Override
    public BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, int... dimension) {
        return createBroadcastOp(name,x,y,x,null,dimension);
    }


    /**
     * This method returns op id number for given opName
     *
     * @param opName
     * @return
     */
    @Override
    public int getOpNumByName(String opName) {
        try {
            DifferentialFunction op =  DifferentialFunctionClassHolder.getInstance().getInstance(opName);
            return  op.opNum();
        } catch (Exception e) {
            throw new RuntimeException("OpName failed: [" + opName + "]",e);
        }
    }

    @Override
    public int getOpNumIfExists(String opName) {
        if(DifferentialFunctionClassHolder.getInstance().hasName(opName)) {
            return getOpNumByName(opName);
        } else
            return -1;
    }

    @Override
    public Op getOpByName(String opName) {
        return (Op) DifferentialFunctionClassHolder.getInstance().getInstance(opName);
    }
}
