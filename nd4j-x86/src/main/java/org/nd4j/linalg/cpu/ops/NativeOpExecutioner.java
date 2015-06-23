package org.nd4j.linalg.cpu.ops;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.LinearViewNDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.cpu.javacpp.Loop;
import org.nd4j.linalg.cpu.util.ArgsConverter;


/**
 * @author Adam Gibson
 */

public class NativeOpExecutioner extends DefaultOpExecutioner {
    private Loop loop = new Loop();

    @Override
    public Op exec(Op op) {
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

        return op;
    }

    private void exec(ScalarOp op) {
        if(op.x() instanceof IComplexNDArray || op.x() instanceof LinearViewNDArray) {
            super.exec(op);
            ;        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                loop.execScalarDouble(op.x().data().asDouble(),op.z().data().asDouble(),op.n(),op.x().offset(),op.x().majorStride(),op.z().majorStride(),op.name(),new double[]{op.scalar().doubleValue()});
            }
            else {
                loop.execScalarFloat(op.x().data().asFloat(), op.z().data().asFloat(), op.n(), op.x().offset(), op.x().majorStride(), op.z().majorStride(), op.name(), new float[]{op.scalar().floatValue()});

            }
        }
    }

    private void exec(TransformOp op) {
        if(op.x() instanceof IComplexNDArray || op.x() instanceof LinearViewNDArray) {
            super.exec(op);

        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.y() != null) {
                    loop.execDoubleTransform(op.x().data().asDouble(),op.y().data().asDouble(),op.n(),op.x().offset(),op.y().offset(),op.x().majorStride(),op.y().elementStride(),op.z().elementStride(),op.name(),new double[1],op.z().data().asDouble());
                }
                else
                    loop.execDoubleTransform(op.x().data().asDouble(),op.n(),op.x().offset(),op.x().majorStride(),op.z().majorStride(),op.name(),ArgsConverter.convertExtraArgsDouble(op),op.z().data().asDouble());
            }
            else {
                if(op.y() != null) {
                    loop.execFloatTransform(op.x().data().asFloat(), op.y().data().asFloat(), op.n(), op.x().offset(), op.y().offset(), op.x().majorStride(), op.y().elementStride(), op.z().elementStride(), op.name(), new float[1], op.z().data().asFloat());

                }
                else
                    loop.execFloatTransform(op.x().data().asFloat(),op.n(),op.x().offset(),op.x().majorStride(),op.z().majorStride(),op.name(),ArgsConverter.convertExtraArgsFloat(op),op.z().data().asFloat());
            }
        }
    }


    private void exec(Accumulation op) {
        if(op.x() instanceof IComplexNDArray || op.x() instanceof LinearViewNDArray) {
            super.exec(op);

        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.y() != null) {
                    op.setCurrentResult(loop.reduce3(op.x().data().asDouble(),op.y().data().asDouble(),op.n(),op.x().offset(),op.y().offset(),op.x().majorStride(),op.y().majorStride(),op.name()
                            , ArgsConverter.convertExtraArgsDouble(op)));
                }
                else {
                    op.setCurrentResult(loop.reduce(op.x().data().asDouble(),op.n(),op.x().offset(),op.x().majorStride(),op.name(),ArgsConverter.convertExtraArgsDouble(op)));
                }
            }
            else {
                if(op.y() != null) {
                    op.setCurrentResult(loop.reduce3Float(op.x().data().asFloat(), op.y().data().asFloat(), op.n(), op.x().offset(), op.y().offset(), op.x().majorStride(), op.y().majorStride(), op.name()
                            , ArgsConverter.convertExtraArgsFloat(op)));
                }
                else {
                    op.setCurrentResult(loop.reduceFloat(op.x().data().asFloat(), op.n(), op.x().offset(), op.x().majorStride(), op.name(), ArgsConverter.convertExtraArgsFloat(op)));
                }
            }
        }
    }
}
