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

package org.nd4j.linalg.api.ops.factory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.reflections.Reflections;
import org.reflections.scanners.SubTypesScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;

import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;


/**
 * Default operations factory
 *
 * @author Adam Gibson
 */
public class DefaultOpFactory implements OpFactory {
    private Map<String, Class<? extends Op>> opClazzes;


    public DefaultOpFactory() {
        opClazzes = new HashMap<>();

        Reflections f = new Reflections(new ConfigurationBuilder().filterInputsBy(
                new FilterBuilder().include(FilterBuilder.prefix("org.nd4j")).exclude("^(?!.*\\.class$).*$") //Consider only .class files (to avoid debug messages etc. on .dlls, etc
                        .exclude("^(?!org\\.nd4j\\.linalg\\.api\\.ops).*") //Exclude any not in the ops directory
        )

                .setUrls(ClasspathHelper.forPackage("org.nd4j")).setScanners(new SubTypesScanner()));

        Set<Class<? extends Op>> clazzes = f.getSubTypesOf(Op.class);

        for (Class<? extends Op> clazz : clazzes) {
            if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface())
                continue;

            try {
                opClazzes.put(clazz.newInstance().name(), clazz);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public LossFunction createLossFunction(String name, INDArray x, INDArray y) {
        Class<? extends Op> clazz = opClazzes.get(name);
        try {
            Constructor<Op> constructor =
                    (Constructor<Op>) clazz.getDeclaredConstructor(INDArray.class, INDArray.class);
            Op create = constructor.newInstance(x, y);
            return (LossFunction) create;
        } catch (Exception e) {
            throw new IllegalArgumentException("Illegal op " + name);
        }

    }

    @Override
    public Accumulation createAccum(String name, INDArray x) {
        return createAccum(name,x,null,x,null);
    }

    @Override
    public Accumulation createAccum(String name, INDArray x, INDArray y, INDArray z) {
        return createAccum(name,x,y,z,null);
    }


    @Override
    public Accumulation createAccum(String name, INDArray x, INDArray y, INDArray z, Object[] extraArgs) {
        Accumulation ret = null;
        switch (name) {
            case "sum":
                ret = new Sum(x, y, x.length());
                break;
            case "max":
                ret = new Max(x, y, x.length());
                break;
            case "min":
                ret = new Min(x, y, x.length());
                break;
            case "norm1":
                ret = new Norm1(x, y, x.length());
                break;
            case "norm2":
                ret = new Norm2(x, y, x.length());
                break;
            case "prod":
                ret = new Prod(x, y, x.length());
                break;
            case "std":
                ret = new StandardDeviation(x, y, x.length());
                break;
            case "var":
                ret = new Variance(x, y, x.length());
                break;
            case "euclidean":
                ret = new EuclideanDistance(x, y, x.length());
                break;
            case "cosine":
            case "cosinesimilarity":
                ret = new CosineSimilarity(x, y, x.length());
                break;
            case "manhattan":
                ret = new ManhattanDistance(x, y, x.length());
                break;

        }

        ret.setExtraArgs(extraArgs);
        return ret;
    }


    @Override
    public Accumulation createAccum(String name, INDArray x, INDArray y) {
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
    public TransformOp createTransform(String name, INDArray x, INDArray y, INDArray z, Object[] extraArgs) {
        TransformOp op = null;
        switch (name) {
            case "relu":
                op = new RectifedLinear(x, z, 0);
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
                op = new Pow(x, z, 2);
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
            case "softmax":
                op = new SoftMax(x, z);
                break;
            case "softplus":
                op = new SoftPlus(x, z);
                break;
            case "cube":
                op = new Cube(x, z);
                break;

        }


        op.setExtraArgs(extraArgs);
        return op;
    }




    protected Class<? extends Op> lookupFunctionByName(String name) {
        return opClazzes.get(name);

    }

    @Override
    public BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, INDArray z, int... dimension) {
        return createBroadcastOp(name,x,y,z,null,dimension);
    }

    @Override
    public BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, INDArray z, Object[] extraArgs, int... dimension) {
        BroadcastOp broadcastOp = null;
        switch (name) {
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

        broadcastOp.setExtraArgs(extraArgs);
        return broadcastOp;
    }

    @Override
    public BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, int... dimension) {
        return createBroadcastOp(name,x,y,x,null,dimension);
    }
}
