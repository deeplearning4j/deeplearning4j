package org.nd4j.imports.TFGraphs.listener;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

public class OpExecOrderListener extends BaseListener {

    @Getter @Setter
    protected List<String> opNamesList;
    protected Set<String> opSet;

    public OpExecOrderListener(){
        this.opNamesList = new ArrayList<>();
        this.opSet = new HashSet<>();
    }

    @Override
    public void opExecution(SameDiff sd, At at, boolean training, SameDiffOp op, INDArray[] outputs) {
        String opName = op.getName();
        if(!opSet.contains(opName)){
            opNamesList.add(opName);
            opSet.add(opName);
        }
    }

}
