# Control flow

## Status
**Discussion**

Proposed by: Adam Gibson (10th April 2022)


## Context

Samediff supports control flow such as if statements and while loops. However this is not enough and common looping structures
are still hard to use. Onnx has introduced a Loop operation. Loop requires a graph with pre configured graph.
The graph takes in and outputs:
1. current iteration
2. max number of iterations
3. extra condition to use

It approximates a for loop with the following code:
```java
boolean cond = ...;
int maxIterations = ...;
for(int i = 0; i < maxIterations && cond; i++) {
        loop body...
}


```

The loop body is represented as a sub graph attribute on the operation.


## Proposal

Similar to onnx's loop operation coupled with [Invoke](./0019%20-%20Invoke.md)
we provide a new loop that leverages invoke and some fixed conventions of the graph to use a loop body:
```java
  /**
     * Loop with conditions.
     * For more information see the underlying class
     * {@link ControlFlow#loopWithConditions(String[], String, SameDiff, SameDiff, String, SDVariable[], String[], String[])}
     * @param loopParams the loop parameters to loop with
     * @return
     */
    public SDVariable[]  loopWithConditions(ControlFlow.LoopParams loopParams) {
```


### LoopParams
LoopParams looks like the following:
```java
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

```


LoopParams has the following fields:
1. outputVarNames: the  output variable names for the loop
2. The name of the loop controls the name in the control flow with scopeName/loopName/variableName as the convention
scopeName is the current frame (such as within the loop body), the loop name is the loop name showing which loop it is
and the variableName represents the variable within the loop and frame.
3. Parent: the invoking samediff function
4. functionName: The name of the function to be looked up from the parent and invoked using invoke
5. loopVars: the input and outputs of the loops
6. functionBodyInputs:  the function input names to use with Invoke
7. functionBodyOutputs: the function output names to be returned from Invoke 



### Example Usage

```java
        //setup the parent graph to pass inputs to the lambda
        SameDiff parent = SameDiff.create();
        SDVariable input = parent.placeHolder("input",DataType.FLOAT);
        //setup the loop body
        SameDiff loopBody = SameDiff.create();
        SDVariable loopInput = loopBody.placeHolder("input", DataType.FLOAT);
        SDVariable output = loopBody.math().add("output",loopInput,1.0);
        //initialize the control flow with the default parameters such as the current iteration, the max number of iterations and the conditional output from the graph
        SDVariable[] args = ControlFlow.initializeLoopBody(new String[]{"curr_iteration", "max_iterations", "cond_in"}, parent, 5, true);
        SDVariable[] childArgs = ControlFlow.initializeLoopBody(new String[]{"curr_iteration", "max_iterations", "cond_in"}, loopBody, 5, true);

        //input names for the input graph with the 4th input being the input from the parent
        String[] inputNames = {
                "curr_iteration",
                "max_iterations",
                "cond_in",
                "input"
        };
        
        //output names from the output of the lmabda with the 4th being the result of the lamda's application of the input
        String[] outputNames = {
                "curr_iteration",
                "max_iterations",
                "cond_in",
                "output"
        };


        //setup the loop variables for input
        SDVariable[] finalArgs = new SDVariable[args.length + 1];
        for(int i = 0; i < args.length; i++) {
            finalArgs[i] = args[i];
        }
        finalArgs[3] = input;

        
        //put it all together in the loop parameters
        ControlFlow.LoopParams loopParams = ControlFlow.LoopParams.builder()
                .parent(parent)
                .functionBody(loopBody)
                .functionBodyInputs(inputNames)
                .functionBodyOutputs(outputNames)
                .loopVars(finalArgs)
                .loopName("loop")
                .functionName("func")
                .build();

         //control the output parameter names
        String[] finalOutputNames = new String[outputNames.length];
        for(int i = 0; i < finalOutputNames.length; i++) {
            finalOutputNames[i] = outputNames[i] + "_final";
        }
         
        //test the output variables, the names will match the specified output names
        SDVariable[] loopWithConditions = parent.loopWithConditions(finalOutputNames,loopParams);

        INDArray assertion = Nd4j.ones(5).addi(5);
        Map<String, INDArray> output2 = parent.output(Collections.singletonMap("input", Nd4j.ones(5)), "output_final");


```






## Consequences

### Advantages

* More explicit wrapper for loop
* More conventions for looping
* Easier to use


### Disadvantages
* More parameters than while loop
* Inherits the complexity of invoke with the number of parameters needed
* Number of inputs and outputs must match. Might not be intuitive for the user.