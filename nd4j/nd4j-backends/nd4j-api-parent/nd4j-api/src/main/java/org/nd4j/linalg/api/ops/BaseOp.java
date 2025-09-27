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

package org.nd4j.linalg.api.ops;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.graph.OpType;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
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
                type = Type.PAIRWISE;
            }
        } else if (op instanceof ReduceOp) {
            if (op.y() == null)
                type = ((ReduceOp) op).getOpType();
            else
                type = Type.REDUCE3;
        } else if (op instanceof ScalarOp) {
            type = Type.SCALAR;
        } else if (op instanceof BroadcastOp) {
            type = Type.BROADCAST;
        } else if (op instanceof IndexAccumulation) {
            type = Type.INDEXREDUCE;
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
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
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
        this.x = x;
    }

    @Override
    public void setZ(INDArray z) {
        this.z = z;
    }

    @Override
    public void setY(INDArray y) {
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
    public INDArray getInputArgument(int index){
        Preconditions.checkState(index >= 0 && index < 2, "Input argument index must be 0 or 1, got %s", index);
        return index == 0 ? x : y;
    }

    @Override
    public SDVariable[] outputVariables(String baseName) {
        if(zVertexId == null)  {
            val outputNames = sameDiff.getOutputsForOp(this);
            //no need to dynamically create if already exists
            if(outputNames != null) {
                zVertexId = sameDiff.getVariable(outputNames[0]).name();
                SDVariable[] ret =  new SDVariable[]{sameDiff.getVariable(outputNames[0])};
                return ret;

            }

            if(isInPlace()) {
                val newVars = sameDiff.generateOutputVariableForOp(this,null,false);
                val inputArr = x();
                //in place op
                if(inputArr == null) {
                    computeVariables(newVars);
                    return newVars;
                }

                sameDiff.setArrayForVariable(newVars[0].name(),inputArr);
                z = inputArr;
                if(sameDiff.getOutputsForOp(this) == null)
                    sameDiff.addOutgoingFor(newVars,this);
                computeVariables(newVars);

                return newVars;
            }

            SDVariable[] newVars = sameDiff.generateOutputVariableForOp(this, baseName, false);
            computeVariables(newVars);
            if (sameDiff.getOutputsForOp(this) == null)
                sameDiff.addOutgoingFor(newVars, this);
            return newVars;
        }

        return new SDVariable[]{sameDiff.getVariable(zVertexId)};
    }



    /**
     * Calculate output shape using Op.Type-aware logic
     */
    private long[] calculateOutputShapeByType() {
        Op.Type opType = opType();
        switch (opType) {
            case TRANSFORM_SAME:
            case TRANSFORM_FLOAT:
            case TRANSFORM_STRICT:
                return x.shape();

            case PAIRWISE:
            case PAIRWISE_BOOL:
                if (y != null) {
                    return Shape.broadcastOutputShape(x.shape(), y.shape());
                }
                return x.shape();

            case REDUCE_FLOAT:
            case REDUCE_LONG:
            case REDUCE_BOOL:
            case REDUCE_SAME:
                if (this instanceof ReduceOp) {
                    ReduceOp reduceOp = (ReduceOp) this;
                    return Shape.reductionShape(x, dimensions, true, reduceOp.isKeepDims());
                }
                return x.shape();

            case SCALAR:
                if (this instanceof ScalarOp) {
                    return x.shape(); // Scalar ops typically preserve shape
                }
                return x.shape();

            case INDEXREDUCE:
                if (this instanceof IndexAccumulation) {
                    IndexAccumulation idxOp = (IndexAccumulation) this;
                    return Shape.reductionShape(x, dimensions, true, idxOp.isKeepDims());
                }
                return x.shape();

            default:
                return x.shape();
        }
    }

    /**
     * Compute the output vars using this op
     * and store them in the samediff instance.
     * @param newVars the new variables to compute arrays for
     */
    public void computeVariables(SDVariable[] newVars) {
        if (!sameDiff.isEagerMode()) {
            return;
        }

        // Early check: if any placeholder inputs are missing, skip execution entirely
        if (shouldSkipExecutionForMissingPlaceholders()) {

            return;
        }

        try {
            // Validate and set input arrays
            validateAndSetInputArrays();

            // Calculate output shape using Op.Type-aware logic
            long[] outputShape = calculateOutputShapeByType();
            DataType outputDataType = inferOutputDataType();

            // Allocate output array
            if (z == null) {
                z = Nd4j.create(outputDataType, outputShape);
            }

            // Execute with enhanced validation
            executeWithValidation(newVars);

        } catch (Exception e) {
            String errorContext = buildErrorContext(newVars);
            throw new RuntimeException("Failed to compute variables for op " + getOwnName() + ": " + errorContext, e);
        }
    }

    /**
     * Check if we should skip execution due to missing placeholder arrays
     */
    private boolean shouldSkipExecutionForMissingPlaceholders() {
        SDVariable[] args = args();
        if (args == null) return false;

        for (SDVariable arg : args) {
            // If any input is a placeholder without an array, skip execution during import
            if (arg.isPlaceHolder() && !sameDiff.arrayAlreadyExistsForVarName(arg.name())) {
                return true;
            }

            // Also check if it's an ARRAY type variable that doesn't have an array yet
            if (arg.getVariableType() == VariableType.ARRAY && !sameDiff.arrayAlreadyExistsForVarName(arg.name())) {
                Variable varMeta = sameDiff.getVariables().get(arg.name());
                if (varMeta != null && varMeta.getOutputOfOp() != null) {
                    // This is an operation output that hasn't been computed yet

                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Infer output data type based on operation type
     */
    private DataType inferOutputDataType() {
        Op.Type opType = opType();
        switch (opType) {
            case TRANSFORM_SAME:
            case TRANSFORM_STRICT:
                return x.dataType();

            case TRANSFORM_FLOAT:
            case REDUCE_FLOAT:
                return x.dataType().isFPType() ? x.dataType() : DataType.FLOAT;

            case REDUCE_LONG:
            case INDEXREDUCE:
                return DataType.LONG;

            case REDUCE_BOOL:
            case PAIRWISE_BOOL:
                return DataType.BOOL;

            case PAIRWISE:
                if (y != null) {
                    // Promote to higher precision if different types
                    return DataType.FLOAT; // Safe default
                }
                return x.dataType();

            default:
                return x.dataType();
        }
    }



    /**
     * Validate and set input arrays from SameDiff state
     */
    private void validateAndSetInputArrays() {
        SDVariable[] args = args();
        if (args == null || args.length == 0) {
            throw new IllegalArgumentException("No input variables found for op " + getOwnName());
        }

        for (int i = 0; i < args.length; i++) {
            // Skip validation for placeholders during import - this should have been caught earlier
            if (args[i].isPlaceHolder() && !sameDiff.arrayAlreadyExistsForVarName(args[i].name())) {
                throw new IllegalStateException("Placeholder " + args[i].name() + " should have been handled before validateAndSetInputArrays()");
            }

            Variable argMeta = sameDiff.getVariables().get(args[i].name());
            if (argMeta == null || !argMeta.isArrayReady()) {
                throw new IllegalStateException(String.format(
                        "Input variable '%s' at index %d for op '%s' is not ready",
                        args[i].name(), i, getOwnName()));
            }

            INDArray arr = args[i].getArr(true);
            if (arr == null) {
                throw new IllegalStateException("Input array for variable " + args[i].name() + " is null");
            }

            // Set appropriate input based on operation type and index
            assignInputByTypeAndIndex(arr, i);
        }

        // Harmonize data types if needed
        harmonizeInputDataTypes();
    }



    /**
     * Assign input array based on operation type and argument index
     */
    private void assignInputByTypeAndIndex(INDArray arr, int index) {
        if (index == 0) {
            x = arr;
        } else if (index == 1) {
            Op.Type opType = opType();
            switch (opType) {
                case REDUCE3:
                case PAIRWISE:
                case PAIRWISE_BOOL:
                case TRANSFORM_SAME:
                    y = arr;
                    break;
                case REDUCE_FLOAT:
                case REDUCE_LONG:
                case REDUCE_BOOL:
                case REDUCE_SAME:
                    // For reduce ops, second argument might be dimensions
                    if (!arr.isEmpty()) {
                        this.dimensionz = arr;
                        this.dimensions = arr.toLongVector();
                    } else {
                        this.dimensions = new long[0];
                    }
                    break;
                default:
                    y = arr;
                    break;
            }
        }
    }


    /**
     * Execute with enhanced validation and error handling
     */
    private void executeWithValidation(SDVariable[] outputVars) throws Exception {
        try (OpContext ctx = Nd4j.getExecutioner().buildContext()) {
            // Set context inputs/outputs
            if (y != null) {
                ctx.setInputArrays(x, y);
            } else {
                ctx.setInputArrays(x);
            }
            ctx.setOutputArrays(z);

            // Pre-execution listeners
            SameDiffOp op = sameDiff.getOps().get(getOwnName());
            for (Listener l : sameDiff.getListeners()) {
                l.preOpExecution(sameDiff, At.defaultAt(), op, ctx);
            }

            // Execute
            INDArray result = Nd4j.getExecutioner().exec(this, ctx);

            // Post-execution listeners
            for (Listener l : sameDiff.getListeners()) {
                l.opExecution(sameDiff, At.defaultAt(), null, op, ctx, new INDArray[]{result});
            }

            // Store results with validation
            storeResultWithValidation(outputVars, result);

        } catch (Exception e) {
            throw e;
        }
    }


    /**
     * Store execution result with validation
     */
    private void storeResultWithValidation(SDVariable[] outputVars, INDArray result) {
        if (result != null && outputVars.length > 0) {
            outputVars[0].setShape(result.shape());
            sameDiff.setEagerArrForVarName(outputVars[0].name(), result);

            // Update listeners
            for (Listener l : sameDiff.getListeners()) {
                l.preUpdate(sameDiff, At.defaultAt(), sameDiff.getVariables().get(outputVars[0].name()), result);
            }
        }
    }

    /**
     * Harmonize input data types if they differ
     */
    private void harmonizeInputDataTypes() {
        if (x != null && y != null && x.dataType() != y.dataType()) {
            // Promote to higher precision type if needed
            DataType targetType = DataType.FLOAT; // Default fallback
            if (x.dataType().isFPType() || y.dataType().isFPType()) {
                targetType = x.dataType().isFPType() ? x.dataType() : y.dataType();
            }

            x = x.castTo(targetType);
            y = y.castTo(targetType);
        }
    }

    /**
     * Allocate output arrays with proper metadata
     */
    private void allocateOutputArraysWithMetadata(SDVariable[] outputVars) {
        if (z == null && outputVars.length > 0) {
            DataType outputDataType = outputVars[0].dataType();

            if (this instanceof ReduceOp) {
                ReduceOp reduceOp = (ReduceOp) this;
                long[] outputShape = Shape.reductionShape(x, dimensions, true, reduceOp.isKeepDims());
                z = Nd4j.create(outputShape).castTo(outputDataType);
            } else {
                long[] outputShape = x.shape(); // Default to input shape
                try {
                    if (y != null) {
                        // For binary ops, try to infer broadcast shape
                        outputShape = Shape.broadcastOutputShape(x.shape(), y.shape());
                    }
                } catch (Exception e) {
                    // Fall back to x shape if broadcast fails
                    outputShape = x.shape();
                }
                z = Nd4j.create(outputShape).castTo(outputDataType);
            }

            // Set shape information on the output variable
            outputVars[0].setShape(z.shape());
        }
    }

    /**
     * Build error context string for debugging
     */
    private String buildErrorContext(SDVariable[] outputVars) {
        StringBuilder sb = new StringBuilder();
        sb.append("Op: ").append(getOwnName()).append(", Type: ").append(opType());
        sb.append(", Inputs: ");
        SDVariable[] args = args();
        if (args != null) {
            for (int i = 0; i < args.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(args[i].name()).append("(").append(args[i].dataType()).append(")");
                INDArray arr = args[i].getArr(false);
                if (arr != null) {
                    sb.append("[").append(Arrays.toString(arr.shape())).append("]");
                } else {
                    sb.append("[null]");
                }
            }
        }
        sb.append(", Outputs: ");
        for (int i = 0; i < outputVars.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(outputVars[i].name()).append("(").append(outputVars[i].dataType()).append(")");
        }
        return sb.toString();
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

    protected void defineDimensions(long... dimensions) {
        if (dimensions != null && dimensions.length > 0) {
            if(x != null) {
                dimensions = Shape.normalizeAxis(x.rank(), dimensions);
            }
        }

        if (dimensions == null || dimensions.length == 0)
            dimensions = new long[]{Integer.MAX_VALUE};

        this.dimensionz = Shape.ndArrayDimFromLong(dimensions).detach();

    }

    public long[] dimensionsArr() {
        return dimensions;
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
            return Double.valueOf(z.getDouble(0));
        else if (z.isZ())
            return Long.valueOf(z.getInt(0));
        else if (z.isB())
            return  Integer.valueOf(z.getInt(0));

        throw new ND4JIllegalStateException("???");
    }

    @Override
    public int getNumOutputs(){
        //Always 1 for legacy/base ops
        return 1;
    }

    @Override
    public void clearArrays(){
        x = null;
        y = null;
        z = null;
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " + opName());
    }

}
