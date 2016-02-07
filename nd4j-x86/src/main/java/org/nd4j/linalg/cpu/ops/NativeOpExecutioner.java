package org.nd4j.linalg.cpu.ops;


import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
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
    private Loop loop = new Loop();

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
                loop.execScalarDouble(
                        op.x().data().asDouble()
                        ,op.z().data().asDouble()
                        ,op.n()
                        ,op.x().offset(),
                        op.z().offset()
                        ,BlasBufferUtil.getBlasStride(op.x())
                        ,BlasBufferUtil.getBlasStride(op.z())
                        ,op.name()
                        ,new double[]{op.scalar().doubleValue()});
            }
            else {
                loop.execScalarFloat(
                        op.x().data().asFloat()
                        , op.z().data().asFloat()
                        , op.n()
                        , op.x().offset(),
                        op.z().offset()
                        , BlasBufferUtil.getBlasStride(op.x())
                        , BlasBufferUtil.getBlasStride(op.z())
                        , op.name()
                        , new float[]{op.scalar().floatValue()});

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
                    loop.execDoubleTransform(
                            op.x().data().asDouble()
                            ,op.y().data().asDouble()
                            ,op.n()
                            ,op.x().offset()
                            ,op.y().offset()
                            ,op.z().offset(),
                            BlasBufferUtil.getBlasStride(op.x())
                            ,BlasBufferUtil.getBlasStride(op.y())
                            ,BlasBufferUtil.getBlasStride(op.z())
                            ,op.name()
                            ,ArgsConverter.convertExtraArgsDouble(op)
                            ,op.z().data().asDouble());
                }
                else {
                      loop.execDoubleTransform(
                            op.x().data().asDouble()
                            , op.n()
                            , op.x().offset()
                            , op.z().offset(),
                            BlasBufferUtil.getBlasStride(op.x())
                            , BlasBufferUtil.getBlasStride(op.z())
                            , op.name()
                            , ArgsConverter.convertExtraArgsDouble(op)
                            , op.z().data().asDouble());
                }
            }
            else {
                if(op.y() != null) {
                    loop.execFloatTransform(
                            op.x().data().asFloat()
                            , op.y().data().asFloat()
                            , op.n()
                            , op.x().offset()
                            , op.y().offset(),
                            op.z().offset()
                            , BlasBufferUtil.getBlasStride(op.x())
                            , BlasBufferUtil.getBlasStride(op.y())
                            , BlasBufferUtil.getBlasStride(op.z())
                            , op.name()
                            , ArgsConverter.convertExtraArgsFloat(op)
                            , op.z().data().asFloat());

                }
                else {
                    loop.execFloatTransform(
                            op.x().data().asFloat()
                            , op.n()
                            , op.x().offset(),
                            op.z().offset()
                            , BlasBufferUtil.getBlasStride(op.x())
                            , BlasBufferUtil.getBlasStride(op.z())
                            , op.name()
                            , ArgsConverter.convertExtraArgsFloat(op)
                            , op.z().data().asFloat());
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
                    op.setFinalResult(loop.reduce3(
                            op.x().data().asDouble()
                            ,op.y().data().asDouble()
                            ,op.n()
                            ,op.x().offset()
                            ,op.y().offset()
                            ,BlasBufferUtil.getBlasStride(op.x())
                            ,BlasBufferUtil.getBlasStride(op.y())
                            ,op.name()
                            , ArgsConverter.convertExtraArgsDouble(op)));
                }
                else {
                    op.setFinalResult(loop.reduce(
                            op.x().data().asDouble()
                            ,op.n()
                            ,op.x().offset()
                            ,BlasBufferUtil.getBlasStride(op.x())
                            ,op.name()
                            ,ArgsConverter.convertExtraArgsDouble(op)));
                }
            }
            else {
                if(op.y() != null) {
                    op.setFinalResult(loop.reduce3Float(
                            op.x().data().asFloat()
                            , op.y().data().asFloat()
                            , op.n()
                            , op.x().offset()
                            , op.y().offset()
                            , BlasBufferUtil.getBlasStride(op.x())
                            , BlasBufferUtil.getBlasStride(op.y())
                            , op.name()
                            , ArgsConverter.convertExtraArgsFloat(op)));
                }
                else {
                    op.setFinalResult(loop.reduceFloat(
                            op.x().data().asFloat()
                            , op.n()
                            , op.x().offset()
                            , BlasBufferUtil.getBlasStride(op.x())
                            , op.name()
                            , ArgsConverter.convertExtraArgsFloat(op)));
                }
            }
        }
    }
}
