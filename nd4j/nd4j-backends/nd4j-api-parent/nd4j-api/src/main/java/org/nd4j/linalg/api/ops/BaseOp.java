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

package org.nd4j.linalg.api.ops;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
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

    @Getter @Setter
    protected String xVertexId,yVertexId,zVertexId;
    // cached instance, for dataType checks
    protected DataBuffer extraArgz;

    protected INDArray dimensionz;

    public BaseOp() {
    }

    public BaseOp(SameDiff sameDiff, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, inPlace, extraArgs);
    }

    public BaseOp(SameDiff sameDiff, Object[] extraArgs) {
        super(sameDiff, extraArgs);
    }

    /**
     * Specify an alternative result array
     *
     * @param x the input
     * @param z the output array
     */
    public BaseOp(INDArray x, INDArray z) {
        this(x, null, z);
    }


    public BaseOp(INDArray x, INDArray y, INDArray z) {
        super(false);
        this.x = x;
        this.y = y;
        this.z = z;
    }


    /**
     * An op for one ndarray
     *
     * @param x the ndarray
     */
    public BaseOp(INDArray x) {
        this(x, null, x);
    }

    public static Type getOpType(Op op) {
        Type type = null;

        if (op instanceof CustomOp) {
            return Type.CUSTOM;
        } else if (op instanceof TransformOp) {
            if (op.y() == null) {
                type = Type.TRANSFORM_FLOAT;
            } else {
                type = Op.Type.PAIRWISE;
            }
        } else if (op instanceof ReduceOp) {
            if (op.y() == null)
                type = ((ReduceOp) op).getOpType();
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
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
    }

    @Override
    public DataBuffer extraArgsDataBuff(DataType dtype) {
        if (extraArgz != null)
            return extraArgz;

        if (extraArgs != null) {
            if (Shape.isZ(dtype) || Shape.isB(dtype)) {
                long extraz[] = new long[extraArgs.length];
                for (int i = 0; i < extraArgs.length; i++) {
                    if (extraArgs[i] instanceof Number) {
                        Number arg = (Number) extraArgs[i];
                        long val = arg.longValue();
                        extraz[i] = val;
                    }
                }
                extraArgz = Nd4j.getConstantHandler().getConstantBuffer(extraz, dtype);
                return extraArgz;
            } else if (Shape.isR(dtype)) {
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
                extraArgz = Nd4j.getConstantHandler().getConstantBuffer(extraz, dtype);
                return extraArgz;
            }
        }

        return null;
    }

    @Override
    public Buffer extraArgsBuff() {
        if (extraArgs != null) {
            DataBuffer retBuff;
            if (x.data().dataType() == DataType.FLOAT) {
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
    }

    @Override
    public void setZ(INDArray z) {
        if (z == null) {
            SDVariable getResult = sameDiff.getVariable(zVertexId);
            if (getResult != null) {
                if (getResult.getArr() != null)
                    this.z = getResult.getArr();
                else if(sameDiff.getShapeForVarName(getResult.getVarName()) != null) {
                    val shape = sameDiff.getShapeForVarName(getResult.getVarName());
                    sameDiff.setArrayForVariable(getResult.getVarName(),getResult.getWeightInitScheme().create(getResult.dataType(), shape));
                }
                else
                    throw new ND4JIllegalStateException("Unable to set null array for z. Also unable to infer from differential function arguments");

            } else
                throw new ND4JIllegalStateException("Unable to set null array for z. Also unable to infer from differential function arguments");
        } else
            this.z = z;
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
    public SDVariable[] outputVariables(String baseName) {
        if(zVertexId == null)  {
            val outputNames = sameDiff.getOutputsForFunction(this);
            //no need to dynamically create if already exists
            if(outputNames != null) {
                zVertexId = sameDiff.getVariable(outputNames[0]).getVarName();


                return new SDVariable[]{sameDiff.getVariable(outputNames[0])};
            }

            if(isInPlace()) {
                val newVars = sameDiff.generateOutputVariableForOp(this,null,false);
                val inputArr = x();
                //in place op
                if(inputArr == null) {
                    return newVars;
                }

                sameDiff.setArrayForVariable(newVars[0].getVarName(),inputArr);
                z = inputArr;
                if(sameDiff.getOutputsForFunction(this) == null)
                    sameDiff.addOutgoingFor(newVars,this);
                return newVars;
            }

            SDVariable[] newVars = sameDiff.generateOutputVariableForOp(this, baseName, false);
            if (sameDiff.getOutputsForFunction(this) == null)
                sameDiff.addOutgoingFor(newVars, this);
            return newVars;
        }

        return new SDVariable[]{sameDiff.getVariable(zVertexId)};
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
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        BaseOp baseOp = (BaseOp) o;

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
        result = 31 * result + Arrays.hashCode(extraArgs);
        result = 31 * result + (extraArgz != null ? extraArgz.hashCode() : 0);
        return result;
    }

    protected void defineDimensions(int... dimensions){
        if (dimensions != null && dimensions.length > 0) {
            if(x != null) {
                dimensions = Shape.normalizeAxis(x.rank(), dimensions);
            }
        }

        if (dimensions == null || dimensions.length == 0)
            dimensions = new int[]{Integer.MAX_VALUE};

        this.dimensionz = Shape.ndArrayDimFromInt(dimensions);
    }

    public INDArray dimensions() {
        return dimensionz;
    }

    public Number getFinalResult() {
        if (this.z == null)
            throw new ND4JIllegalStateException("Op.Z is null. Op wasn't executed yet?");

        if (z.isEmpty())
            throw new ND4JIllegalStateException("Can't get number from empty array");

        if (!z.isScalar())
            throw new ND4JIllegalStateException("Can't get final result scalar out of N-dim tensor");

        if (z.isR())
            return new Double(z.getDouble(0));
        else if (z.isZ())
            return new Long(z.getInt(0));
        else if (z.isB())
            return new Integer(z.getInt(0));

        throw new ND4JIllegalStateException("???");
    }

    @Override
    public int getNumOutputs(){
        //Always 1 for legacy/base ops
        return 1;
    }
}
