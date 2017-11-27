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

import lombok.Data;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.nio.Buffer;
import java.util.Arrays;
import java.util.Map;

/**
 * Base op. An op involves iterating over 2 buffers (x,y)  up to n elements
 * and applying a transform or accumulating a result.
 *
 * @author Adam Gibson
 */
@Data
public abstract class BaseOp extends DifferentialFunction implements Op {

    protected INDArray x, y, z;
    protected long n;
    protected long numProcessed;
    protected Object[] extraArgs;
    protected boolean passThrough;

    // cached instance, for dataType checks
    protected DataBuffer extraArgz;

    public BaseOp() {
    }


    public BaseOp(SameDiff sameDiff, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, inPlace, extraArgs);
    }

    public BaseOp(SameDiff sameDiff, Object[] extraArgs) {
        super(sameDiff, extraArgs);
    }

    @Override
    public boolean isExecSpecial() {
        return false;
    }

    public static Type getOpType(Op op) {
        Type type = null;

        if (op instanceof CustomOp) {
            return Type.CUSTOM;
        } else if (op instanceof ShapeOp) {
            return Type.SHAPE;
        } else if (op instanceof TransformOp) {
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



    @Override
    protected void addAsNewVertexId() {
        super.addAsNewVertexId();
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
    }

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
                    if (!(extraArgs[i] instanceof Number))
                        continue;
                    Number arg = (Number) extraArgs[i];
                    if (arg == null)
                        arg = 0.0;
                    double val = arg.doubleValue();
                    extraz[i] = val;
                }
                extraArgz = Nd4j.getConstantHandler().getConstantBuffer(extraz);
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
        if (x == null) {
            if (args() != null && args().length >= 1) {
                DifferentialFunction firstArg = args()[0];
                if (firstArg instanceof SDVariable) {
                    SDVariable sdVariable = (SDVariable) firstArg;
                    if (sdVariable.getArr() != null)
                        this.x = sdVariable.getArr();
                }
            } else
                throw new ND4JIllegalStateException("Unable to set null array for x. Also unable to infer from differential function arguments");
        } else
            this.x = x;
        numProcessed = 0;
    }

    @Override
    public void setZ(INDArray z) {
        if (z == null) {
            SDVariable getResult = sameDiff.getVariableForVertexId(vertexId);
            if (getResult != null) {
                if (getResult.getArr() != null)
                    this.z = getResult.getArr();
                else
                    throw new ND4JIllegalStateException("Unable to set null array for z. Also unable to infer from differential function arguments");

            } else
                throw new ND4JIllegalStateException("Unable to set null array for z. Also unable to infer from differential function arguments");
        } else
            this.z = z;
        numProcessed = 0;
    }

    @Override
    public void setY(INDArray y) {
        if (y == null) {
            if (args() != null && args().length > 1) {
                DifferentialFunction firstArg = args()[1];
                if (firstArg instanceof SDVariable) {
                    SDVariable sdVariable = (SDVariable) firstArg;
                    if (sdVariable.getArr() != null)
                        this.y = sdVariable.getArr();
                }
            } else
                throw new ND4JIllegalStateException("Unable to set null array for y. Also unable to infer from differential function arguments");
        } else
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
        this(x, null, x, x == null ? 0 : x.lengthLong());
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
        return opName();
    }


    @Override
    public CustomOp toCustomOp() {
        DynamicCustomOp.DynamicCustomOpsBuilder customOpBuilder = DynamicCustomOp.builder(opName());
        customOpBuilder.callInplace(x() == z());

        if (y() != null)
            customOpBuilder.addInputs(x(), y());
        else
            customOpBuilder.addInputs(x());

        customOpBuilder.addOutputs(z());
        if (extraArgs != null) {
            for (int i = 0; i < extraArgs.length; i++) {
                if (extraArgs[i] instanceof Integer) {
                    customOpBuilder.addIntegerArguments((Integer) extraArgs[i]);
                } else if (extraArgs[i] instanceof Double || extraArgs[i] instanceof Float) {
                    Double num = (Double) extraArgs[i];
                    customOpBuilder.addFloatingPointArguments(num);
                }
            }
        }

        return customOpBuilder.build();

    }

    @Override
    public void exec() {
        //no-op
    }

    @Override
    public void exec(int... dimensions) {
        //no-op
    }



    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        BaseOp baseOp = (BaseOp) o;

        if (n != baseOp.n) return false;
        if (numProcessed != baseOp.numProcessed) return false;
        if (passThrough != baseOp.passThrough) return false;
        if (x != null ? !x.equals(baseOp.x) : baseOp.x != null) return false;
        if (y != null ? !y.equals(baseOp.y) : baseOp.y != null) return false;
        if (z != null ? !z.equals(baseOp.z) : baseOp.z != null) return false;
        // Probably incorrect - comparing Object[] arrays with Arrays.equals
        if (!Arrays.equals(extraArgs, baseOp.extraArgs)) return false;
        return extraArgz != null ? extraArgz.equals(baseOp.extraArgz) : baseOp.extraArgz == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (x != null ? x.hashCode() : 0);
        result = 31 * result + (y != null ? y.hashCode() : 0);
        result = 31 * result + (z != null ? z.hashCode() : 0);
        result = 31 * result + (int) (n ^ (n >>> 32));
        result = 31 * result + (int) (numProcessed ^ (numProcessed >>> 32));
        result = 31 * result + Arrays.hashCode(extraArgs);
        result = 31 * result + (passThrough ? 1 : 0);
        result = 31 * result + (extraArgz != null ? extraArgz.hashCode() : 0);
        return result;
    }
}
