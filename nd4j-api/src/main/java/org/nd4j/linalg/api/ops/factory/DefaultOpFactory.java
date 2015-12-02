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
 *
 */

package org.nd4j.linalg.api.ops.factory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.reflections.Reflections;

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
    private Map<String,Class<? extends Op>> opClazzes;


    public DefaultOpFactory() {
        opClazzes = new HashMap<>();
        Set<Class<? extends Op>> clazzes = new Reflections("org.nd4j").getSubTypesOf(Op.class);
        for(Class<? extends Op> clazz : clazzes) {
            if(Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface())
                continue;

            try {
                opClazzes.put(clazz.newInstance().name(),clazz);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        }
    }

    @Override
    public LossFunction createLossFunction(String name, INDArray x, INDArray y) {
        Class<? extends Op> clazz = opClazzes.get(name);
        try {
            Constructor<Op> constructor = (Constructor<Op>) clazz.getDeclaredConstructor(INDArray.class,INDArray.class);
            Op create = constructor.newInstance(x,y);
            return (LossFunction) create;
        } catch (Exception e) {
            throw new IllegalArgumentException("Illegal op " + name);
        }

    }

    @Override
    public Accumulation createAccum(String name, INDArray x) {
        switch (name) {
            case "sum":
                return new Sum(x);
            case "max":
                return new Max(x);
            case "min":
                return new Min(x);
            case "norm1":
                return new Norm1(x);
            case "norm2":
                return new Norm2(x);
            case "prod":
                return new Prod(x);
            case "std":
                return new StandardDeviation(x);
            case "var":
                return new Variance(x);
            case "euclidean":
                return new EuclideanDistance(x);
            case "cosine":
            case "cosinesimilarity":
                return new CosineSimilarity(x);
            case "manhattan":
                return new ManhattanDistance(x);

            default:
                throw new IllegalArgumentException("Illegal name " + name);
        }
    }

    @Override
    public Accumulation createAccum(String name, INDArray x, INDArray y, INDArray z) {
        switch (name) {
            case "sum":
                return new Sum(x, y, x.length());
            case "max":
                return new Max(x, y, x.length());
            case "min":
                return new Min(x, y, x.length());
            case "norm1":
                return new Norm1(x, y, x.length());
            case "norm2":
                return new Norm2(x, y, x.length());
            case "prod":
                return new Prod(x, y, x.length());
            case "std":
                return new StandardDeviation(x, y, x.length());
            case "var":
                return new Variance(x, y, x.length());
            case "euclidean":
                return new EuclideanDistance(x, y, x.length());
            case "cosine":
            case "cosinesimilarity":
                return new CosineSimilarity(x, y, x.length());
            case "manhattan":
                return new ManhattanDistance(x, y, x.length());

            default:
                throw new IllegalArgumentException("Illegal name " + name);
        }
    }

    @Override
    public Accumulation createAccum(String name, INDArray x, INDArray y) {
        switch (name) {
            case "sum":
                return new Sum(x, y);
            case "max":
                return new Max(x, y);
            case "min":
                return new Min(x, y);
            case "norm1":
                return new Norm1(x, y);
            case "norm2":
                return new Norm2(x, y);
            case "prod":
                return new Prod(x, y);
            case "std":
                return new StandardDeviation(x, y);
            case "var":
                return new Variance(x, y);
            case "euclidean":
                return new EuclideanDistance(x, y, x.length());
            case "cosine":
            case "cosinesimilarity":
                return new CosineSimilarity(x, y, x.length());
            case "manhattan":
                return new ManhattanDistance(x, y, x.length());

            default:
                throw new IllegalArgumentException("Illegal name " + name);
        }
    }

    @Override
    public IndexAccumulation createIndexAccum(String name, INDArray x){
        switch(name){
            case "iamax":
                return new IAMax(x);
            case "imax":
                return new IMax(x);
            case "imin":
                return new IMin(x);
            default:
                throw new IllegalArgumentException("Illegal name: " + name);
        }
    }

    @Override
    public IndexAccumulation createIndexAccum(String name, INDArray x, INDArray y){
        switch(name){
            case "iamax":
                return new IAMax(x,y);
            case "imax":
                return new IMax(x,y);
            case "imin":
                return new IMin(x,y);
            default:
                throw new IllegalArgumentException("Illegal name: " + name);
        }
    }

    @Override
    public TransformOp createTransform(String name, INDArray x, INDArray y) {
        switch (name) {
            case "relu":
                return new RectifedLinear(x,0);
            case "abs":
                return new Abs(x, y);
            case "acos":
                return new ACos(x, y);
            case "asin":
                return new ASin(x, y);
            case "atan":
                return new ATan(x, y);
            case "ceil":
                return new Ceil(x, y);
            case "cos":
                return new Cos(x, y);
            case "exp":
                return new Exp(x, y);
            case "elu":
                return new ELU(x, y);
            case "floor":
                return new Floor(x, y);
            case "hardtanh":
                return new HardTanh(x, y);
            case "identity":
                return new Identity(x, y);
            case "log":
                return new Log(x, y);
            case "logsoftmax":
                return new LogSoftMax(x,y);
            case "leakyrelu":
            	return new LeakyReLU(x,y);
            case "maxout":
                return new MaxOut(x, y);
            case "negative":
                return new Negative(x, y);
            case "pow":
                return new Pow(x, y, 2);
            case "round":
                return new Round(x, y);
            case "sigmoid":
                return new Sigmoid(x, y);
            case "sign":
                return new Sign(x, y);
            case "sin":
                return new Sin(x, y);
            case "softsign":
            	return new SoftSign(x,y);
            case "sqrt":
                return new Sqrt(x, y);
            case "stabilize":
                return new Stabilize(x, y, 1);
            case "tanh":
                return new Tanh(x, y);
            case "timesoneminus":
            	return new TimesOneMinus(x,y);
            case "softmax":
                return new SoftMax(x, y);
            case "softplus":
                return new SoftPlus(x);
            case "step":
            	return new Step(x,y);
            default:
                throw new IllegalArgumentException("Illegal name " + name);
        }

    }

    @Override
    public TransformOp createTransform(String name, INDArray x) {
        switch (name) {
            case "relu":
                return new RectifedLinear(x,0);
            case "abs":
                return new Abs(x);
            case "acos":
                return new ACos(x);
            case "asin":
                return new ASin(x);
            case "atan":
                return new ATan(x);
            case "ceil":
                return new Ceil(x);
            case "cos":
                return new Cos(x);
            case "elu":
                return new ELU(x);
            case "exp":
                return new Exp(x);
            case "floor":
                return new Floor(x);
            case "hardtanh":
                return new HardTanh(x);
            case "identity":
                return new Identity(x);
            case "leakyrelu":
            	return new LeakyReLU(x);
            case "log":
                return new Log(x);
            case "logsoftmax":
                return new LogSoftMax(x);
            case "maxout":
                return new MaxOut(x);
            case "negative":
                return new Negative(x);
            case "pow":
                return new Pow(x, 2);
            case "round":
                return new Round(x);
            case "sigmoid":
                return new Sigmoid(x);
            case "sign":
                return new Sign(x);
            case "sin":
                return new Sin(x);
            case "softsign":
            	return new SoftSign(x);
            case "sqrt":
                return new Sqrt(x);
            case "stabilize":
                return new Stabilize(x, 1);
            case "tanh":
                return new Tanh(x);
            case "timesoneminus":
            	return new TimesOneMinus(x);
            case "softmax":
                return new SoftMax(x);
            case "softplus":
                return new SoftPlus(x);
            case "step":
            	return new Step(x);
            default:
                throw new IllegalArgumentException("Illegal name " + name);
        }

    }

    @Override
    public TransformOp createTransform(String name, INDArray x, INDArray y, INDArray z) {
        switch (name) {
            case "relu":
                return new RectifedLinear(x,z,0);
            case "abs":
                return new Abs(x, z);
            case "acos":
                return new ACos(x, z);
            case "asin":
                return new ASin(x, z);
            case "atan":
                return new ATan(x, z);
            case "ceil":
                return new Ceil(x, z);
            case "cos":
                return new Cos(x, z);
            case "exp":
                return new Exp(x, z);
            case "elu":
                return new ELU(x, z);
            case "floor":
                return new Floor(x, z);
            case "hardtanh":
                return new HardTanh(x, z);
            case "identity":
                return new Identity(x, z);
            case "leakyrelu":
            	return new LeakyReLU(x,z);
            case "log":
                return new Log(x, z);
            case "logsoftmax":
                return new LogSoftMax(x,z);
            case "maxout":
                return new MaxOut(x, z);
            case "negative":
                return new Negative(x, z);
            case "pow":
                return new Pow(x, z, 2);
            case "round":
                return new Round(x, z);
            case "sigmoid":
                return new Sigmoid(x, z);
            case "sign":
                return new Sign(x, z);
            case "sin":
                return new Sin(x, z);
            case "softsign":
            	return new SoftSign(x,z);
            case "sqrt":
                return new Sqrt(x, z);
            case "stabilize":
                return new Stabilize(x, z, 1);
            case "tanh":
                return new Tanh(x, z);
            case "timesoneminus":
            	return new TimesOneMinus(x,z);
            case "softmax":
                return new SoftMax(x, z);
            case "softplus":
                return new SoftPlus(x,z);

            default:
                throw new IllegalArgumentException("Illegal name " + name);
        }
    }

    protected Class<? extends Op> lookupFunctionByName(String name) {
        return opClazzes.get(name);

    }

    @Override
    public BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, INDArray z, int... dimension){
        switch(name){
            case "broadcastadd":
                return new BroadcastAddOp(x,y,z,dimension);
            case "broadcastsub":
                return new BroadcastSubOp(x,y,z,dimension);
            case "broadcastmul":
                return new BroadcastMulOp(x,y,z,dimension);
            case "broadcastdiv":
                return new BroadcastDivOp(x,y,z,dimension);
            case "broadcastrsub":
                return new BroadcastRSubOp(x,y,z,dimension);
            case "broadcastrdiv":
                return new BroadcastRDivOp(x,y,z,dimension);
            case "broadcastcopy":
                return new BroadcastCopyOp(x,y,z,dimension);
            default:
                throw new IllegalArgumentException("Illegal name " + name);
        }
    }

    @Override
    public BroadcastOp createBroadcastOp(String name, INDArray x, INDArray y, int... dimension){
        switch(name){
            case "broadcastadd":
                return new BroadcastAddOp(x,y,x,dimension);
            case "broadcastsub":
                return new BroadcastSubOp(x,y,x,dimension);
            case "broadcastmul":
                return new BroadcastMulOp(x,y,x,dimension);
            case "broadcastdiv":
                return new BroadcastDivOp(x,y,x,dimension);
            case "broadcastrsub":
                return new BroadcastRSubOp(x,y,x,dimension);
            case "broadcastrdiv":
                return new BroadcastRDivOp(x,y,x,dimension);
            case "broadcastcopy":
                return new BroadcastCopyOp(x,y,x,dimension);
            default:
                throw new IllegalArgumentException("Illegal name " + name);
        }
    }
}
