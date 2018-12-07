package org.nd4j.autodiff.samediff.internal;

import com.google.common.collect.Iterables;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.controlflow.If;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
public class InferenceSession extends AbstractSession<INDArray,DifferentialFunction> {

    public InferenceSession(@NonNull SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public INDArray[] getOutputs(DifferentialFunction op) {

        if(op instanceof Identity) {
            Identity i = (Identity) op;
            String[] argNames = i.argNames();
            Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in identity op, got %s", argNames);
            return new INDArray[]{nodeOutputs.get(argNames[0])};

        } else if(op instanceof Switch){
            Switch s = (Switch)op;
            String[] argNames = s.argNames();       //Order: input, boolean array
            INDArray predicate = this.nodeOutputs.get(argNames[1]);
            Preconditions.checkState(predicate.isScalar() && predicate.dataType() == DataType.BOOL, "Expected boolean predicate: got %ndSInfo", predicate);
            if(predicate.getDouble(0) == 0.0){
                return new INDArray[]{this.nodeOutputs.get(argNames[0]), null};
            } else {
                return new INDArray[]{null, this.nodeOutputs.get(argNames[0])};
            }
        } else if(op instanceof If) {
            If i = (If) op;
            String[] argNames = i.argNames();       //Order should be: [boolean], true, false


            throw new UnsupportedOperationException("Execution not yet implemented for: " + op.getClass().getName());
        } else if(op instanceof Merge){
            //Merge avairable for forward pass when any of its inputs are available. When multiple are available, behaviour
            // is undefined
            Merge m = (Merge)op;
            String[] in = sameDiff.getInputsForFunction(op);
            for(String s : in){
                if(nodeOutputs.containsKey(s)){
                    log.info("Returning input \"{}\" for merge node \"{}\"", m.getOwnName(), s);
                    return new INDArray[]{nodeOutputs.get(s)};
                }
            }
            throw new IllegalStateException("Merge node " + m.getOwnName() + " has no available inputs (all inputs: " + Arrays.toString(in) +
                    ") - should not be executed at this point");
        } else if(op instanceof CustomOp){
            CustomOp c = (CustomOp)op;
            Nd4j.getExecutioner().exec(c);
            return c.outputArguments();
        } else if(op instanceof Op) {
            Op o = (Op) op;
            Nd4j.getExecutioner().exec(o);
            return new INDArray[]{o.z()};
        } else {
            throw new UnsupportedOperationException("Execution not yet implemented for: " + op.getClass().getName());
        }
    }

    @Override
    public DifferentialFunction getAndParameterizeOp(String opName) {

        DifferentialFunction df = sameDiff.getFunctionById(opName);

        //TODO We should clone these - probably - as we don't want them shared between threads/sessions!
        //But let's only clone them *once* and cache in inference session - not on every exec

        Preconditions.checkNotNull(df, "No differential function fond with name %s", opName);

        if(df instanceof LoopCond || df instanceof Enter || df instanceof Exit || df instanceof NextIteration ||
                df instanceof Merge || df instanceof Switch || df instanceof If || df instanceof While){
            return df;
        }

        if(df instanceof CustomOp){
            DynamicCustomOp customOp = (DynamicCustomOp) df;
            try {
                customOp.populateInputsAndOutputsFromSameDiff();
            } catch (Throwable t) {
                throw new RuntimeException("Error populating inputs and outputs for function \"" + df.getOwnName()
                        + "\" of type " + df.getClass().getName(), t);
            }
        } else if(df instanceof Op){
            Op op = (Op) df;
            String outVarName = ((BaseOp) op).outputVariable().getVarName();

            SDVariable[] inputs = sameDiff.getInputVariablesForFunction(df);

            // ops in differential function might have stale NDArrays used. we should renew them
            if(inputs != null && inputs.length > 0) {
                op.setX(inputs[0].getArr());
                if (inputs.length == 2)
                    op.setY(inputs[1].getArr());
            }

            //Check output shape; allocate a new Z if required
            //For example, if minibatch size has changed since last op execution
            List<LongShapeDescriptor> outputShape = ((BaseOp)op).calculateOutputShape();
            Preconditions.checkState(outputShape != null && outputShape.size() == 1, "Could not calculate output shape for op: %s", op.getClass());
            INDArray z = op.z();
            Preconditions.checkNotNull(z, "Could not get output array for op: %s", op.getClass());
            if(!outputShape.get(0).equals(z.shapeDescriptor())){
                if(log.isTraceEnabled()){
                    log.trace("Existing op result (z) array shape for op {} was {}, allocating new array of shape {}",
                            op.getClass().getSimpleName(), Arrays.toString(z.shape()), outputShape.get(0).toString());
                }
                //Get output variable:
                String outputName = sameDiff.getOutgoingArgsReverse().get(opName)[0];
                SDVariable outputVar = sameDiff.getVariable(outputName);

                z = outputVar.storeAndAllocateNewArray();       //TODO this shouldn't be done - or stored - in the SameDiff instance
                op.setZ(z);
            }
        }

        //TODO actually set inputs etc. This is just placeholder for testing order etc
        return sameDiff.getFunctionById(opName);
    }

    @Override
    public void preprocessPlaceholderValues(Map<String, INDArray> placeholderValues) {
        //TODO eventually placeholders will NOT be stored in SameDiff itself. But we'll set them for now until that is changed

        for(Map.Entry<String,INDArray> placeholder : placeholderValues.entrySet() ){
            //TODO let's add a "getPlaceholder(String)" method...
            sameDiff.getVariable(placeholder.getKey()).setArray(placeholder.getValue());
        }
    }


}
