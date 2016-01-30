package org.nd4j.linalg.cpu.ops;


import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.LinearViewNDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.cpu.javacpp.Loop;
import org.nd4j.linalg.cpu.util.ArgsConverter;


/**
 *
 * Native operation executioner in c++
 *
 * @author Adam Gibson
 */

public class NativeOpExecutioner extends DefaultOpExecutioner {
    private NativeOps loop = new NativeOps();

    @Override
    public Op exec(Op op) {
        if(op.isPassThrough()  || executionMode() == ExecutionMode.JAVA)
            return super.exec(op);

        if(op instanceof ScalarOp) {
            ScalarOp s = (ScalarOp) op;
            exec(s);
        }
        else if(op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            exec(t);
        }
        else if(op instanceof Accumulation) {
            Accumulation ac = (Accumulation) op;
            exec(ac);
        }
        else if(op instanceof IndexAccumulation){
            IndexAccumulation iac = (IndexAccumulation)op;
            exec(iac);  //Currently using DefaultOpExecutioner
        }

        return op;
    }

    private void exec(ScalarOp op) {
        if(op.x() instanceof IComplexNDArray || op.x() instanceof LinearViewNDArray || executionMode() == ExecutionMode.JAVA) {
            super.exec(op);
        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            }
            else {


            }
        }
    }

    private void exec(TransformOp op) {
        if(op.x() instanceof IComplexNDArray || op.x() instanceof LinearViewNDArray ||   executionMode() == ExecutionMode.JAVA) {
            super.exec(op);
        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.y() != null) {

                }
                else {

                }
            }
            else {
                if(op.y() != null) {


                }
                else {

                }
            }
        }
    }


    private void exec(Accumulation op) {
        if(op.x() instanceof IComplexNDArray || op.x() instanceof LinearViewNDArray || executionMode() == ExecutionMode.JAVA) {
            super.exec(op);

        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.y() != null) {

                }
                else {

                }
            }
            else {
                if(op.y() != null) {

                }
                else {

                }
            }
        }
    }
}
