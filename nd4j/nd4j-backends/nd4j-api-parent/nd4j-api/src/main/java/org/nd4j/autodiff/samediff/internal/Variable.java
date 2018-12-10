package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;

@AllArgsConstructor
@NoArgsConstructor
@Data   //TODO immutable?
@Builder
public class Variable {
    protected String name;
    protected SDVariable variable;
    protected Object shapeInfo;         //TODO decide type, or if even to include (Variable class should ideally be immutable)
    protected String[] inputsForOp;
    protected String outputOfOp;        //Null for placeholders/constants. For array type SDVariables, the name of the op it's an output of
    protected String[] controlDeps;     //Control dependencies: name of variables that must be available before this variable is considered available for execution
    protected int outputOfOpIdx;        //Index of the output for the op (say, variable is output number 2 of op "outputOfOp")
    protected SDVariable gradient;      //Variable corresponding to the gradient of this variable
}
