package org.nd4j.autodiff.samediff.internal;

import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;

@Data   //TODO immutable?
public class Variable {
    protected String name;
    protected SDVariable variable;
    protected Object shapeInfo;         //TODO decide type, or if even to include (Variable class should ideally be immutable)
    protected String[] inputsForOp;
    protected String outputOfOp;        //Null for placeholders/constants. For array type SDVariables, the name of the op it's an output of
    protected String[] controlDeps;     //Control dependencies: name of variables that must be available before this variable is considered available for execution
    protected int outputOfOpIdx;        //Index of the output for the op (say, variable is output number 2 of op "outputOfOp")
    protected String gradVariableName;  //Name of the variable corresponding to the gradient of this variable
}
