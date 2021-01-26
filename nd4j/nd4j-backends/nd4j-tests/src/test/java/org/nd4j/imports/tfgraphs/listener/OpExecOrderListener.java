package org.nd4j.imports.tfgraphs.listener;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;

public class OpExecOrderListener extends BaseListener {

    @Getter @Setter
    protected List<String> opNamesList;
    protected Set<String> opSet;

    public OpExecOrderListener(){
        this.opNamesList = new ArrayList<>();
        this.opSet = new HashSet<>();
    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        String opName = op.getName();
        if(!opSet.contains(opName)){
            opNamesList.add(opName);
            opSet.add(opName);
        }
    }

    @Override
    public boolean isActive(Operation operation) {
        return true;
    }
}
