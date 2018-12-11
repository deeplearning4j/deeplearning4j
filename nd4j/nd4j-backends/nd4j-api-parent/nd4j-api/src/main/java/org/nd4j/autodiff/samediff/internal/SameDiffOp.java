package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class SameDiffOp {
    protected String name;
    protected DifferentialFunction op;	//Actual op (note: should be mutable: i.e., cloneable, no arrays set)
    protected List<String> inputsToOp;		//Name of SDVariables as input
    protected List<String> outputsOfOp;	    //Name of SDVariables as output
    protected List<String> controlDeps;	    //Name of SDVariables as control dependencies (not data inputs, but need to be available before exec)
}
