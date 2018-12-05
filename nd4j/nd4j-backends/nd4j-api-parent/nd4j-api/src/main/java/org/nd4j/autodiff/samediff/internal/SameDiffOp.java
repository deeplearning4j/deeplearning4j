package org.nd4j.autodiff.samediff.internal;

import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;

@Data
public class SameDiffOp {
    protected String name;
    protected DifferentialFunction op;	//Actual op (note: should be mutable: i.e., cloneable, no arrays set)
    protected String[] inputsToOp;		//Name of SDVariables as input
    protected String[] outputsOfOp;	    //Name of SDVariables as output
    protected String[] controlDeps;	    //Name of SDVariables as control dependencies (not data inputs, but need to be available before exec)
}
