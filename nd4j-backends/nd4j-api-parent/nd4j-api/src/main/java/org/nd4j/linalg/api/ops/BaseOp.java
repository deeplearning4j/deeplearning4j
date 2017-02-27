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

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.Buffer;

/**
 * Base op. An op involves iterating over 2 buffers (x,y)  up to n elements
 * and applying a transform or accumulating a result.
 *
 * @author Adam Gibson
 */
public abstract class BaseOp implements Op {

    protected INDArray x, y, z;
    protected long n;
    protected long numProcessed;
    protected Object[] extraArgs;
    protected boolean passThrough;

    // cached instance, for dataType checks
    protected DataBuffer extraArgz;

    public BaseOp() {}

    @Override
    public boolean isExecSpecial() {
        return false;
    }

    public static Type getOpType(Op op) {
        Type type = null;

        if (op instanceof TransformOp) {
            if (op.y() == null) {
                if (!op.isExecSpecial())
                    type = Op.Type.TRANSFORM;
                else
                    type = Op.Type.SPECIAL;
            } else {
                type = Op.Type.PAIRWISE;
            }
        } else if (op instanceof Accumulation) {
            if (op.y() == null)
                type = Op.Type.REDUCE;
            else
                type = Op.Type.REDUCE3;
        } else if (op instanceof ScalarOp) {
            type = Op.Type.SCALAR;
        } else if (op instanceof BroadcastOp) {
            type = Op.Type.BROADCAST;
        } else if (op instanceof IndexAccumulation) {
            type = Op.Type.INDEXREDUCE;
        } else if (op instanceof MetaOp) {
            type = Type.META;
        } else if (op instanceof GridOp) {
            type = Type.GRID;
        }

        return type;
    }

    /*
    op instanceof Variance
    op instanceof CosineSimilarity
    op instanceof Im2col
    op instanceof Col2Im
    op instanceof Pow
    op instanceof LeakyReLU
    op instanceof RectifedLinear
    op instanceof LeakyReLUDerivative
    op instanceof Stabilize
    op instanceof Step
    op instanceof SetRange
    op instanceof IsMax
    */

    @Override
    public DataBuffer extraArgsDataBuff() {
        if (extraArgz != null)
            return extraArgz;

        if (extraArgs != null) {
            DataBuffer.Type dtype = x != null ? x.data().dataType() : Nd4j.dataType();
            if (dtype == DataBuffer.Type.FLOAT || dtype == DataBuffer.Type.HALF) {
                float extraz[] = new float[extraArgs.length];
                for (int i = 0; i < extraArgs.length; i++) {
                    Number arg = (Number) extraArgs[i];
                    float val = arg.floatValue();
                    extraz[i] = val;
                }
                extraArgz = Nd4j.getConstantHandler().getConstantBuffer(extraz);
                return extraArgz;
            } else if (dtype == DataBuffer.Type.DOUBLE) {
                double extraz[] = new double[extraArgs.length];
                for (int i = 0; i < extraArgs.length; i++) {
                    Number arg = (Number) extraArgs[i];
                    double val = arg.doubleValue();
                    extraz[i] = val;
                }
                extraArgz = Nd4j.getConstantHandler().getConstantBuffer(extraz);;
                return extraArgz;
            }
        }
        return null;
    }

    @Override
    public Buffer extraArgsBuff() {
        if (extraArgs != null) {
            DataBuffer retBuff;
            if (x.data().dataType() == DataBuffer.Type.FLOAT) {
                retBuff = Nd4j.createBuffer(new float[extraArgs.length]);
                for (int i = 0; i < extraArgs.length; i++) {
                    Number val = (Number) extraArgs[i];
                    retBuff.put(i, val.floatValue());
                }
                return retBuff.asNioFloat();
            } else {
                retBuff = Nd4j.createBuffer(new double[extraArgs.length]);
                for (int i = 0; i < extraArgs.length; i++) {
                    Number val = (Number) extraArgs[i];
                    retBuff.put(i, val.doubleValue());
                }
                return retBuff.asNioDouble();
            }


        }
        return null;
    }

    @Override
    public boolean isPassThrough() {
        return passThrough;
    }

    @Override
    public void setX(INDArray x) {
        if (x == null)
            throw new IllegalArgumentException("X must not be null");
        this.x = x;
        numProcessed = 0;
    }

    @Override
    public void setZ(INDArray z) {
        if (z == null)
            throw new IllegalArgumentException("Z must not be null");
        this.z = z;
        numProcessed = 0;
    }

    @Override
    public void setY(INDArray y) {
        if (y == null)
            throw new IllegalArgumentException("Y must not be null");
        this.y = y;
        numProcessed = 0;
    }

    /**
     * Specify an alternative result array
     *
     * @param x the input
     * @param z the output array
     */
    public BaseOp(INDArray x, INDArray z) {
        this(x, z, x.lengthLong());
    }

    /**
     * Specify an alternative output array
     *
     * @param x the input
     * @param z the output
     * @param n the number of elements to iterate on
     */
    public BaseOp(INDArray x, INDArray z, long n) {
        this(x, null, z, n);
    }


    public BaseOp(INDArray x, INDArray y, INDArray z, long n) {
        init(x, y, z, n);
    }



    /**
     * An op for one ndarray
     *
     * @param x the ndarray
     */
    public BaseOp(INDArray x) {
        this(x, null, x, x.lengthLong());
    }

    @Override
    public Object[] extraArgs() {
        return extraArgs;
    }

    @Override
    public INDArray x() {
        return x;
    }

    @Override
    public INDArray y() {
        return y;
    }


    @Override
    public INDArray z() {
        return z;
    }

    @Override
    public long n() {
        return n;
    }


    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.n = n;

    }

    @Override
    public void setN(long n) {
        this.n = n;
    }

    @Override
    public long numProcessed() {
        return numProcessed;
    }

    @Override
    public String toString() {
        return name();
    }

    @Override
    public void exec() {
        //no-op
    }

    @Override
    public void exec(int... dimensions) {
        //no-op
    }
}
