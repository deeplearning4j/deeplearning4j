package org.nd4j.imports.listeners;

import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * A very quick and dirty debugging listener
 * This listener just prints the outputs of any ops during execution
 * @author Alex Black
 */
public class ExecPrintListener extends BaseListener {
    @Override
    public boolean isActive(Operation operation) {
        return true;
    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        System.out.println("------ Op: " + op.getName() + " - opName = " + op.getOp().opName() + ", class = " + op.getOp().getClass().getName() + " ------");
        for(INDArray arr : outputs){
            System.out.println(arr);
        }
    }
}
