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
package org.nd4j.autodiff.samediff;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.custom.Invoke;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Enter;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Exit;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.NextIteration;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.collect.Sets;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Top level class for looping constructs in samediff.
 * This includes the ability to create for and while loops as well as
 * encapsulate the usage of invoke as a function body. This spec can be read here:
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop
 *
 * The core components of the looping function are as follows:
 * 1. Loop variables:
 *     a. current iteration (gets updated during loop body) (defaults to 0)
 *     b. max number of iterations (defaults to {@link Long#MAX_VALUE}
 *     c. a current condition a user passes in and is updated during lambda invocation
 *  Any variables beyond the first 3 are extra variables by the user
 *
 *
 */
public class ControlFlow {




    /**
     * Initializes the loop variables with default parameters. The variables are as follows:
     * current iteration
     * max number of iterations
     * extra condition to use
     *
     *
     *
     * The passed in variable names will be assumed to be names for each of these variables
     * mentioned above respectively. Please ensure that these are the intended names
     * of the variables.
     * @param namesToUse the names of the variables to use. Must be length 2.
     * @param loopBody the loop body to initialize
     * @param maxIterations the max iterations to iterate over
     */
    public static SDVariable[] initializeLoopBody(String[] namesToUse,SameDiff loopBody,int maxIterations) {
        Preconditions.checkState( namesToUse != null && namesToUse.length == 2,"Number of input names must be 2.");
        SDVariable[] ret = new SDVariable[] {
                loopBody.constant(namesToUse[1], maxIterations),
                loopBody.var(namesToUse[0], Nd4j.zeros(1)),
        };
        return ret;
    }

    /**
     * Initializes the loop variables with default parameters. The variables are as follows:
     * current iteration
     * max number of iterations
     * extra condition to use
     *
     * The passed in variable names will be assumed to be names for each of these variables
     * mentioned above respectively. Please ensure that these are the intended names
     * of the variables.
     * @param namesToUse the names of the variables to use. Must be length 3.
     * @param loopBody the loop body to initialize
     * @param maxIterations the max iterations to iterate over
     * @param extraCond the extra condition to use
     */
    public static SDVariable[] initializeLoopBody(String[] namesToUse,SameDiff loopBody,int maxIterations,boolean extraCond) {
        Preconditions.checkState( namesToUse != null && namesToUse.length == 3,"Number of input names must be 3.");
        SDVariable[] ret = new SDVariable[] {
                loopBody.var(namesToUse[0], Nd4j.zeros(1)),
                loopBody.constant(namesToUse[1], maxIterations),
                loopBody.constant(namesToUse[2], extraCond)
        };
        return ret;
    }

    /**
     * Create the arguments used in {@link #condBody()}
     * and {@link #loopWithConditions(String[], String, SameDiff, SameDiff, String, SDVariable[], String[], String[])}
     * @param maxIterations the max number of iterations
     * @param condIn the input conditions
     * @param startIterations the start iterations
     * @param extraArgs the extra arguments for the user
     * @return the ordered arguments
     */
    public static SDVariable[] args(SDVariable maxIterations,SDVariable condIn,SDVariable startIterations,SDVariable[] extraArgs) {
        return LoopArgs.builder().extraArgs(extraArgs)
                .condIn(condIn)
                .maxIters(maxIterations)
                .startIter(startIterations).build().toArgs();
    }

    /**
     * Constructs a If statement using the tensorflow style control flow operations (Switch and Merge)
     *
     * If the result of cond is true, returns the result of trueBody, otherwise returns the result of falseBody
     *
     * Note that cond and body lambdas are only called once to construct the graph.  The constructed graph is used to evaluate.
     *
     * See <a href="http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf">Tensorflow Control Flow Implementation</a>
     *
     * @param outputName Name to give the output variable.  If null, doesn't rename
     * @param ifName  The name of the if block.  If null, uses "if"
     * @param cond  A lambda evaluating to the if condition
     * @param trueBody  A lambda to be executed if cond is true (the if block)
     * @param falseBody  A lambda to be executed if cond is false (the else block)
     * @return The value of trueBody if cond is true, or falseBody if it isn't
     */
    public static SDVariable ifCond(SameDiff sameDiff,String outputName, String ifName, @NonNull SameDiffNoArgSingleLambda cond,
                                    @NonNull SameDiffNoArgSingleLambda trueBody, @NonNull SameDiffNoArgSingleLambda falseBody){

        ifName = sameDiff.newBlockName(ifName == null ? "if" : ifName);

        NameScope ifScope = sameDiff.withNameScope(ifName);

        NameScope condScope = sameDiff.withNameScope("cond");
        final SDVariable pred = cond.define(sameDiff);
        condScope.close();

        if (pred.dataType() != DataType.BOOL) {
            //cleanup partially added block

            for(SDVariable v : sameDiff.getVariablesInScope(ifScope))
                sameDiff.getVariables().remove(v.name());

            for(SameDiffOp op : sameDiff.getOpsInScope(ifScope)) {
                for(String in : op.getInputsToOp()){
                    sameDiff.removeArgFromOp(in, op.getOp());
                }
                sameDiff.getOps().remove(op.getName());
            }


            throw new IllegalStateException("Can not use " + pred.name()
                    + " as the condition of an If statement, the condition must be a boolean.");
        }

        final Map<String, SDVariable[]> switches = new HashMap<>();

        final Set<String> declared = Sets.newHashSet(sameDiff.variableMap().keySet());

        sameDiff.addArgumentInterceptor(argument -> {

            if(argument == null)
                return null;
            // if its declared in the if, we don't care about it
            if(declared == null || !declared.contains(argument.name()))
                return argument;

            // if we've already added a switch, move on
            if(switches.containsKey(argument.name()))
                return switches.get(argument.name())[1];

            SDVariable[] s = sameDiff.switchOp(argument, pred);
            switches.put(argument.name(), s);
            return s[1];
        });
        NameScope trueScope = sameDiff.withNameScope("trueBody");
        SDVariable trueOut = trueBody.define(sameDiff);
        sameDiff.removeArgumentInterceptor();

        if(declared.contains(trueOut.name())) {
            SDVariable[] s = sameDiff.switchOp(trueOut, pred);
            switches.put(trueOut.name(), s);
            trueOut = s[1];
        }

        trueScope.close();

        final Set<String> declared2 = Sets.newHashSet(sameDiff.variableMap().keySet());
        sameDiff.addArgumentInterceptor(argument -> {

            // if its declared in the if, we don't care about it
            if(!declared2.contains(argument.name()))
                return argument;

            // if we've already added a switch, move on
            if(switches.containsKey(argument.name()))
                return switches.get(argument.name())[0];

            SDVariable[] s = sameDiff.switchOp(argument, pred);
            switches.put(argument.name(), s);
            return s[0];
        });
        NameScope falseScope = sameDiff.withNameScope("falseBody");
        SDVariable falseOut = falseBody.define(sameDiff);
        sameDiff.removeArgumentInterceptor();

        if(declared2.contains(falseOut.name())) {
            SDVariable[] s = sameDiff.switchOp(falseOut, pred);
            switches.put(falseOut.name(), s);
            falseOut = s[0];
        }
        falseScope.close();

        SDVariable output = sameDiff.merge(trueOut, falseOut);

        ifScope.close();

        return sameDiff.updateVariableNameAndReference(output, outputName);
    }

    @Builder
    @Data
    public static class LoopArgs {
        private SDVariable condIn,maxIters,startIter;
        private SDVariable[] extraArgs;

        public SDVariable[] toArgs() {
            SDVariable[] ret = new SDVariable[3 + extraArgs.length];
            ret[0] = startIter;
            ret[1] = maxIters;
            ret[2] = condIn;
            for(int i = 0; i < extraArgs.length; i++) {
                ret[i + 3] = extraArgs[i];
            }
            return ret;
        }

    }

    @Builder
    @Data
    public static class LoopParams {
        private String[] outputVarNames;
        private String loopName;
        private SameDiff parent;
        private SameDiff functionBody;
        private String functionName;
        private SDVariable[] loopVars;
        private String[] functionBodyInputs;
        private String[] functionBodyOutputs;
    }


    /**
     * A simplified function using {@link LoopParams}
     * invoking the same function as {@link #loopWithConditions(String[], String, SameDiff, SameDiff, String, SDVariable[], String[], String[])}
     * @param loopParams the loop parameters to use
     * @return
     */
    public static SDVariable[] loopWithConditions(LoopParams loopParams) {
        return loopWithConditions(loopParams.outputVarNames,
                loopParams.loopName,loopParams.parent,
                loopParams.functionBody,
                loopParams.functionName,
                loopParams.loopVars,
                loopParams.functionBodyInputs,
                loopParams.functionBodyOutputs);
    }

    /**
     * Loop with conditions allows a user to provide a lambda to invoke
     * any number of times.
     * @param outputVarNames the output variable names to use
     * @param loopName the name of the loop to use when creating the variables/ops
     * @param parent the parent samediff instance to put the loop
     * @param functionBody the function body to use
     * @param functionName the name of the function to use within the samediff instance
     * @param loopVars the loop variables to use during execution
     * @param functionBodyInputs the inputs to invoke the function with
     * @param functionBodyOutputs the outputs to be retrieved from the function itself
     * @return the output exit variables at the end of the loop
     */
    public static SDVariable[] loopWithConditions(
            String[] outputVarNames,
            String loopName,
            SameDiff parent,
            SameDiff functionBody,
            String functionName,
            SDVariable[] loopVars,
            String[] functionBodyInputs,
            String[] functionBodyOutputs) {
        Preconditions.checkState(functionBodyInputs != null && functionBodyOutputs != null && functionBodyInputs.length == functionBodyOutputs.length,"Sub graph input and output names must  be defined and equal in length.");
        Preconditions.checkState(loopVars.length == functionBodyInputs.length,"Loop variables and function body inputs must be equal in length.");
        SameDiffSingleLambda cond = condBody();
        SameDiffLambda loopBody = loopBody(parent,functionBody,functionName,functionBodyInputs,functionBodyOutputs);
        return parent.whileLoop(outputVarNames,loopName,loopVars,cond,loopBody);

    }

    /**
     * Create {@link LoopLambdaArgs} from the given arguments.
     * This is used to properly order arguments for use with {@link #loopBody(SameDiff, SameDiff, String, String[], String[])}
     * and {@link #condBody()}
     * @param inputs the inputs to order, these generally should be from within a lambda. The first 3 arguments are:
     *               current iter count, maximum number of iterations, extra arguments if any
     * @return
     */
    public static LoopLambdaArgs argsFromInputs(SDVariable[] inputs) {

        SDVariable[] extraArgs = inputs.length > 3 ? new SDVariable[inputs.length - 3] : new SDVariable[0];
        //add extra arguments offset by 3 representing custom inputs
        if(extraArgs.length > 0) {
            for(int i = 0; i < extraArgs.length; i++) {
                extraArgs[i] = inputs[i + 3];
            }
        }
        return LoopLambdaArgs.builder()
                .iterCount(inputs[1])
                .iterStart(inputs[0])
                .condIn(inputs[2])
                .extraArgs(extraArgs)
                .build();
    }

    @Data
    public static class LoopLambdaArgs {

        private SDVariable iterStart;
        private SDVariable iterCount;
        private SDVariable condIn;
        private SDVariable[] extraArgs;

        @Builder
        public LoopLambdaArgs(SDVariable iterStart,SDVariable iterCount,SDVariable[] extraArgs,SDVariable condIn) {
            if(condIn.dataType() != DataType.BOOL) {
                throw new IllegalArgumentException("Data type for condition must be boolean!");
            }

            if(!iterCount.dataType().isNumerical()) {
                throw new IllegalArgumentException("Data type for condition must be numerical!");
            }

            this.iterCount = iterCount;
            this.extraArgs = extraArgs;
            this.condIn = condIn;
            this.iterStart = iterStart;
        }

        /**
         * Construct {@link org.nd4j.linalg.api.ops.custom.Invoke.InvokeParams}
         * for usage with {@link SameDiff#invoke(Invoke.InvokeParams)}
         * the variables here reflect what is used in the loop.
         * A user can use {@link LoopLambdaArgs} to create an appropriately configured
         * {@link org.nd4j.linalg.api.ops.custom.Invoke.InvokeParams} to be used
         * with the body.
         *
         *
         *
         * @param functionName the name of the function to invoke
         * @param subGraphInputNames the subgraph input names to invoke the function with
         * @param subGraphOutputNames the subgraph output names to expect returned from the function
         * @return the appropriate invoke parameters for use with {@link #condBody()} and {@link #loopBody(SameDiff, SameDiff, String, String[], String[])}
         */
        public Invoke.InvokeParams invokeParams(String functionName,String[] subGraphInputNames,String[] subGraphOutputNames) {
            List<SDVariable> inputs = new ArrayList<>();
            //starting iteration
            inputs.add(iterStart);
            //ending iteration
            inputs.add(iterCount);
            //user custom condition
            inputs.add(condIn);
            inputs.addAll(Arrays.asList(extraArgs));
            return Invoke.InvokeParams.builder()
                    .functionName(functionName)
                    .inputs(inputs.toArray(new SDVariable[inputs.size()]))
                    .subGraphInputVarNames(subGraphInputNames)
                    .subGraphOutputVarNames(subGraphOutputNames)
                    .inputVarNames(inputs.stream().map(input ->
                                    input.name()).collect(Collectors.toList())
                            .toArray(new String[inputs.size()]))
                    .build();
        }

    }


    /**
     * Create a {@link SameDiffLambda} to be used in combination with
     * {@link #condBody()} and {@link SameDiff#invoke(Invoke.InvokeParams)}
     * this lambda will use samediff invoke as the function bdoy
     * and setup the appropriate parameters to create a looping construct
     * as described in {@link #loopBody(SameDiff, SameDiff, String, String[], String[])}
     * @param parent
     * @param functionBody
     * @param functionName
     * @param subGraphInputNames  the subgraph input names for use to invoke the graph with
     * @param subGraphOutputNames the subgraph output names to expect to be returned from the subgraph invoke
     * @return
     */
    public static SameDiffLambda loopBody(SameDiff parent,
                                          SameDiff functionBody,
                                          String functionName,
                                          String[] subGraphInputNames,
                                          String[] subGraphOutputNames) {
        Preconditions.checkState(subGraphInputNames != null && subGraphOutputNames != null && subGraphInputNames.length == subGraphOutputNames.length,"Sub graph input and output names must  be defined and equal in length.");
        parent.putSubFunction(functionName,functionBody);
        return (sameDiff, inputs) -> {
            LoopLambdaArgs loopLambdaArgs = ControlFlow.argsFromInputs(inputs);
            Invoke.InvokeParams invokeParams = loopLambdaArgs.invokeParams(functionName, subGraphInputNames, subGraphOutputNames);
            SDVariable[] invoke = sameDiff.invoke(invokeParams);
            List<SDVariable> retList = new ArrayList<>();
            //current iterations + 1 (each time the body is invoked update the current iteration)
            retList.add(inputs[0].add(1.0));
            retList.add(inputs[1]);
            retList.add(invoke[2]);

            //assign extra parameters to the invoke output
            //loop over non condition out variables starting from the end
            for(int i =  3; i <  invoke.length; i++) {
                retList.add(invoke[i]);
            }

            return retList.toArray(new SDVariable[retList.size()]);
        };
    }


    /**
     * Constructs a While loop using the tensorflow style control flow operations (Switch, Merge, Enter, Exit, and NextIteration)
     * <p>
     * Repeatedly executes body on the loop variables and updates them with the results, until cond evaluates to false
     * <p>
     * Note that cond and body lambdas are only called once to construct the graph.  The constructed graph is used for further iterations.
     * <p>
     * See <a href="http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf">Tensorflow Control Flow Implementation</a>
     *
     * @param outputNames Names to give the output variables.  If null, doesn't rename
     * @param loopName    The name of the loop block and frame (must be unique).  If null, uses "if"
     * @param loopVars    Loop variables' inputs
     * @param cond        A lambda evaluating to the loop condition
     * @param body        A lambda doing the loop operation and returning the new loop variable values
     * @return The values of the loop variables once condition is false
     */
    public static SDVariable[] whileLoop(SameDiff sameDiff, String[] outputNames, final String loopName, @NonNull SDVariable[] loopVars,
                                         @NonNull SameDiffSingleLambda cond, @NonNull SameDiffLambda body) {

        final String frameName = sameDiff.newBlockName(loopName == null ? "while" : loopName);

        NameScope loopScope = sameDiff.withNameScope(frameName);

        SDVariable counter = sameDiff.scalar(sameDiff.generateNewVarName("counter", 0), 0);

        SDVariable[] entered = new SDVariable[loopVars.length];
        for (int i = 0; i < loopVars.length; i++) {
            entered[i] = new Enter(sameDiff, frameName, loopVars[i]).outputVariable();
        }

        SDVariable[] merged = new SDVariable[loopVars.length];
        Merge[] mergeOps = new Merge[loopVars.length];
        for (int i = 0; i < loopVars.length; i++) {
            // the second arg will later be replaced with the output of NextIteration
            // but that isn't available yet (and can't be, as it depends on this)
            mergeOps[i] = new Merge(sameDiff, entered[i], entered[i]);
            merged[i] = mergeOps[i].outputVariable();
        }

        Merge counterMerge = new Merge(sameDiff, counter, counter);
        counter = counterMerge.outputVariable();

        NameScope condScope = sameDiff.withNameScope("cond");
        SDVariable condResult = cond.define(sameDiff, merged);
        condScope.close();


        if (condResult.dataType() != DataType.BOOL)
            throw new IllegalStateException("Can not use " + condResult.name() + " as the condition of an While loop, the condition must be a boolean.");


        final Set<String> alreadyEntered = Sets.newHashSet();
        SDVariable[] trueSwitches = new SDVariable[loopVars.length];
        SDVariable[] exits = new SDVariable[loopVars.length];
        for (int i = 0; i < loopVars.length; i++) {
            SDVariable[] s = sameDiff.switchOp(merged[i], condResult);
            trueSwitches[i] = s[1];
            alreadyEntered.add(s[1].name());
            exits[i] = new Exit(sameDiff, s[0]).outputVariable();
        }

        final Set<String> declared = Sets.newHashSet(sameDiff.variableMap().keySet());
        final Map<String, SDVariable> done = new HashMap<>();

        final SameDiff sd = sameDiff;
        sameDiff.addArgumentInterceptor(argument -> {
            if (argument == null)
                return null;

            if (!declared.contains(argument.name()))
                return argument;

            if (alreadyEntered.contains(argument.name()))
                return argument;

            if (done.containsKey(argument.name()))
                return done.get(argument.name());

            SDVariable e = new Enter(sd, frameName, argument, true).outputVariable();
            done.put(argument.name(), e);
            return e;
        });

        NameScope bodyScope = sameDiff.withNameScope("body");
        SDVariable[] outs = body.define(sameDiff, trueSwitches);
        if (outs.length != mergeOps.length)
            throw new IllegalArgumentException("Number of loop variables must be equal to number of outputs.");
        bodyScope.close();
        sameDiff.removeArgumentInterceptor();

        counter.add(1);

        for (int i = 0; i < outs.length; i++) {
            SDVariable n = new NextIteration(sameDiff, outs[i]).outputVariable();
            mergeOps[i].replaceArg(1, n);
        }

        counterMerge.replaceArg(1, counter);

        loopScope.close();
        return sameDiff.updateVariableNamesAndReferences(exits, outputNames);
    }


    /**
     * Returns a lambda that takes in a custom condition and a built-in for
     * loop counter concept in the following manner:
     * int currIteration = 0;
     * boolean cond = ...;
     * int maxIterations = ...;
     * for(int i = currIteration; i < maxIterations && cond; i++) {
     *     //body....
     * }
     *
     * The inputs to the lambda are the following order:
     * currIteration (the starting iteration)
     * maxIterations (the number of times to loop)
     * cond: the custom condition the user passes in
     *
     *
     * @return the lambda described above for usage in the {@link #whileLoop(SameDiff, String[], String, SDVariable[], SameDiffSingleLambda, SameDiffLambda)}
     * routine
     */
    public static SameDiffSingleLambda condBody() {
        //  combine for loop and while loop together
        return (sameDiff, inputs) -> {
            SDVariable currIteration = inputs[0];
            SDVariable maxIterations = inputs[1];
            SDVariable extraCond = inputs[2];
            SDVariable and = sameDiff.bitwise().and(
                    currIteration.lt(maxIterations.castTo(currIteration.dataType()))
                            .castTo(DataType.INT64),
                    extraCond.castTo(DataType.INT64));


            SDVariable ret = and.castTo( DataType.BOOL);
            return ret;
        };


    }
}