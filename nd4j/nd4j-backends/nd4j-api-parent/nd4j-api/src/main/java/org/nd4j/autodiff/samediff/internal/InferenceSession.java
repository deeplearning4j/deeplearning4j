package org.nd4j.autodiff.samediff.internal;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
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

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
public class InferenceSession extends AbstractSession<INDArray,DifferentialFunction> {

    public InferenceSession(@NonNull SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public INDArray[] getOutputs(DifferentialFunction op, VarId anOutput, Set<VarId> opInputs, Set<String> constAndPhInputs) {

        int totalInputs = (opInputs == null ? 0 : opInputs.size()) + (constAndPhInputs == null ? 0 : constAndPhInputs.size());

        if(op instanceof Identity ) {
            Identity i = (Identity) op;
            String[] argNames = i.argNames();
            Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in identity op, got %s", argNames);
            VarId vid = newVarId(argNames[0], anOutput);
            return new INDArray[]{nodeOutputs.get(vid)};

        } else if(op instanceof Switch) {
            Switch s = (Switch) op;
            String[] argNames = s.argNames();       //Order: input, boolean array
            VarId vidPredicate = newVarId(argNames[1], anOutput);
            INDArray predicate = this.nodeOutputs.get(vidPredicate);
            Preconditions.checkState(predicate.isScalar() && predicate.dataType() == DataType.BOOL, "Expected boolean predicate: got %ndSInfo", predicate);
            VarId vid = newVarId(argNames[0], anOutput);
            if (predicate.getDouble(0) == 0.0) {
                return new INDArray[]{this.nodeOutputs.get(vid), null};
            } else {
                return new INDArray[]{null, this.nodeOutputs.get(vid)};
            }
        } else if(op instanceof Enter) {
            //Enter op: forwards input to specified execution frame
            Enter e = (Enter)op;
            String frame = e.getFrameName();
            String[] input = e.argNames();
            Preconditions.checkState(input.length == 1, "Expected only 1 arg name for enter op: got %s", input);
            Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input, got %s+%s", opInputs, constAndPhInputs);

            VarId inputVarId;
            if(opInputs == null || opInputs.size() == 0){
                //Constant or placeholder
                inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0);
            } else {
                inputVarId = opInputs.iterator().next();
            }
            INDArray enterInput = this.nodeOutputs.get(inputVarId);

            Preconditions.checkNotNull(enterInput, "Could not get enter op input: output variable %s", anOutput);
            return new INDArray[]{enterInput};
        } else if(op instanceof Exit) {
            //Exit node forwards input to parent frame

            VarId inputVarId;
            if(opInputs == null || opInputs.size() == 0){
                //Constant or placeholder
                inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0);
            } else {
                inputVarId = opInputs.iterator().next();
            }
            INDArray exitInput = this.nodeOutputs.get(inputVarId);
            return new INDArray[]{exitInput};
        } else if(op instanceof NextIteration){
            //NextIteration op: forwards its single input to the output of the current frame, but increments the iteration number
            Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input for NextIteration: got %s+%s", opInputs, constAndPhInputs);
            VarId in = opInputs.iterator().next();
            Preconditions.checkState(anOutput.getFrame().equals(in.getFrame()), "Expected same frame for NextIteration input vs. output:" +
                    " got input %s, output %s", in, anOutput);
            Preconditions.checkState(anOutput.getIteration() == in.getIteration()+1, "Expected output iteration for NextIteration output to" +
                    " be 1 larger than the input iteration. Input: %s, output %s", in, anOutput);

            INDArray inArr = this.nodeOutputs.get(in);
            return new INDArray[]{inArr};
        } else if(op instanceof If) {
            If i = (If) op;
            String[] argNames = i.argNames();       //Order should be: [boolean], true, false


            throw new UnsupportedOperationException("Execution not yet implemented for: " + op.getClass().getName());
        } else if(op instanceof Merge) {
            //Merge avairable for forward pass when any of its inputs are available. When multiple are available, behaviour
            // is undefined
            Merge m = (Merge) op;
            String[] in = sameDiff.getInputsForFunction(op);
            for (String s : in) {
                VarId vid = newVarId(s, anOutput);
                if (nodeOutputs.containsKey(vid)) {
                    log.info("Returning input \"{}\" for merge node \"{}\"", m.getOwnName(), s);
                    return new INDArray[]{nodeOutputs.get(vid)};
                }
            }
            throw new IllegalStateException("Merge node " + m.getOwnName() + " has no available inputs (all inputs: " + Arrays.toString(in) +
                    ") - should not be executed at this point");
        } else if(op instanceof LoopCond) {
            //LoopCond just forwards scalar boolean to output
            LoopCond lc = (LoopCond) op;
            String[] argNames = lc.argNames();
            Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in LoopCond op, got %s", argNames);
            VarId vid = newVarId(argNames[0], anOutput);
            INDArray arr = nodeOutputs.get(vid);
            Preconditions.checkNotNull(arr, "Input to LoopCond op must not be null");
            Preconditions.checkState(arr.isScalar() && arr.dataType() == DataType.BOOL, "LoopCond input must be a scalar boolean, got %ndShape");
            return new INDArray[]{arr};
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
    public INDArray getConstant(VarId varId) {
        return sameDiff.getArrForVarName(varId.getVariable());
    }

    @Override
    public DifferentialFunction getAndParameterizeOp(String opName, VarId anOutput, Set<VarId> opInputs, Set<String> constAndPhInputs) {

        DifferentialFunction df = sameDiff.getFunctionById(opName);

        //TODO We should clone these - probably - as we don't want them shared between threads/sessions!
        //But let's only clone them *once* and cache in inference session - not on every exec

        Preconditions.checkNotNull(df, "No differential function fond with name %s", opName);

        if(df instanceof LoopCond || df instanceof Enter || df instanceof Exit || df instanceof NextIteration ||
                df instanceof Merge || df instanceof Switch || df instanceof If || df instanceof While){
            return df;
        }

        String[] argNames = df.argNames();
        int numArgs = (argNames == null ? 0 : argNames.length);
        int numNonConstIns = (opInputs == null ? 0 : opInputs.size());
        int numConstPhIns = (constAndPhInputs == null ? 0 : constAndPhInputs.size());
        Preconditions.checkState(numArgs == (numNonConstIns + numConstPhIns),
                "Different number of arg names as op inputs for op %s (%s): arg names %s vs. op inputs %s+%s", df.getClass().getSimpleName(),
                    opName, argNames, opInputs, constAndPhInputs);
        INDArray[] args = null;
        if(argNames != null && argNames.length > 0) {
            args = new INDArray[argNames.length];
            if(opInputs != null) {
                for (VarId vid : opInputs) {
                    int idx = ArrayUtils.indexOf(argNames, vid.getVariable());
                    Preconditions.checkState(idx >= 0, "Variable %s not found in arg names: %s", vid.getVariable(), argNames);
                    args[idx] = this.nodeOutputs.get(vid);
                }
            }
            if(constAndPhInputs != null) {
                for (String s : constAndPhInputs) {
                    int idx = ArrayUtils.indexOf(argNames, s);
                    Preconditions.checkState(idx >= 0, "Variable %s not found in arg names: %s", s, argNames);
                    VarId constPhVarId = newVarId(s, OUTER_FRAME, 0);
                    args[idx] = this.nodeOutputs.get(constPhVarId);
                }
            }
        }

        if(df instanceof CustomOp){
            DynamicCustomOp customOp = (DynamicCustomOp) df;
            try {
                customOp.populateInputsAndOutputsFromSameDiff();
            } catch (Throwable t) {
                throw new RuntimeException("Error populating inputs and outputs for function \"" + df.getOwnName()
                        + "\" of type " + df.getClass().getName(), t);
            }

            //TODO we'll remove populateInputsAndOutputsFromSameDiff call soon, and populate directly here
//            String[] argNames = customOp.argNames();
//            Preconditions.checkState((argNames == null && opInputs.size() == 0) || (argNames != null && argNames.length == opInputs.size()),
//                    "Different number of arg names as op inputs: %s vs. %s", argNames, opInputs);
//            if(argNames != null && argNames.length > 0) {
//                //INDArray[] newInputs = new INDArray[argNames.length];
//                for (VarId vid : opInputs) {
//                    int idx = ArrayUtils.indexOf(argNames, vid.getVariable());
//                    Preconditions.checkState(idx >= 0, "Variable %s not found in arg names: %s", vid.getVariable(), argNames);
//                    //newInputs[idx] = this.nodeOutputs.get(vid);
//
//                    customOp.setInputArgument(idx, this.nodeOutputs.get(vid));
//                }
//            }

            //TODO why doesn't CustomOp have a setInputs(INDArray[])?
            if(args != null) {
                for (int i = 0; i < args.length; i++) {
                    customOp.setInputArgument(i, args[i]);
                }
            }
        } else if(df instanceof Op){
            Op op = (Op) df;
            String outVarName = ((BaseOp) op).outputVariable().getVarName();

//            SDVariable[] inputs = sameDiff.getInputVariablesForFunction(df);
//
//            // ops in differential function might have stale NDArrays used. we should renew them
//            //TODO let's remove this getArr usage here, and populate directly
//            if(inputs != null && inputs.length > 0) {
//                op.setX(inputs[0].getArr());
//                if (inputs.length == 2)
//                    op.setY(inputs[1].getArr());
//            }

            if(args != null && args.length > 0){
                op.setX(args[0]);
                if (args.length == 2)
                    op.setY(args[1]);
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
