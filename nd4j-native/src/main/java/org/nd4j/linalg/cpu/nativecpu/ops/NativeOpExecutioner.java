package org.nd4j.linalg.cpu.nativecpu.ops;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.Floor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;


/**
 *
 * Native operation
 * executioner in c++
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
            IndexAccumulation iac = (IndexAccumulation) op;
            exec(iac);  //Currently using DefaultOpExecutioner
        }

        return op;
    }


    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        java.nio.IntBuffer dimensionBuffer = Shape.toBuffer(dimension);
        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            loop.execIndexReduce(op.opNum(),
                    op.x().data().asNioDouble(),
                    op.x().shapeInfo(), (DoubleBuffer) op.extraArgsBuff(),
                    op.z().data().asNioDouble(),
                    op.z().shapeInfo(),
                    dimensionBuffer, dimension.length);

        }
        else {
            loop.execIndexReduce(op.opNum(),
                    op.x().data().asNioFloat(),
                    op.x().shapeInfo(), (FloatBuffer) op.extraArgsBuff(),
                    op.z().data().asNioFloat(),
                    op.z().shapeInfo(),
                    dimensionBuffer, dimension.length);

        }
        return op.z();
    }

    @Override
    protected void doAccumulationOp(Accumulation op) {
        exec(op);
    }

    @Override
    protected void doBroadcastOp(BroadcastOp op) {
        exec(op);
    }

    @Override
    protected void doIndexAccumulationOp(IndexAccumulation op) {
        exec(op);
    }

    @Override
    protected void doScalarOp(ScalarOp op) {
        exec(op);
    }

    @Override
    protected void doTransformOp(TransformOp op) {
        exec(op);
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1,1} : ArrayUtil.removeIndex(op.x().shape(), dimension);
        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new int[]{1, retShape[0]};
            else
                retShape = new int[]{retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new int[]{1, 1};
        }

        INDArray ret = Nd4j.valueArrayOf(retShape,op.zeroDouble());
        op.setZ(ret);
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};

        java.nio.IntBuffer dimensionBuffer = Shape.toBuffer(dimension);
        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            if(op instanceof Variance) {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execSummaryStatsScalar(
                            op.opNum()
                            ,op.x().data().asNioDouble(),
                            op.x().shapeInfo(),
                            (DoubleBuffer) op.extraArgsBuff()));
                }
                else {
                    loop.execSummaryStats(op.opNum(),
                            op.x().data().asNioDouble(),
                            op.x().shapeInfo(), (DoubleBuffer) op.extraArgsBuff(),
                            op.z().data().asNioDouble(),
                            op.z().shapeInfo(),
                            dimensionBuffer, dimension.length);
                }

            }

            else if(op.y() != null) {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduce3Scalar(op.opNum(),
                            op.x().data().asNioDouble(),
                            op.x().shapeInfo(),
                            (DoubleBuffer)op.extraArgsBuff(),
                            op.y().data().asNioDouble(),
                            op.y().shapeInfo()));
                }
                else {
                    loop.execReduce3(op.opNum(),
                            op.x().data().asNioDouble(),
                            op.x().shapeInfo(), (DoubleBuffer) op.extraArgsBuff(),
                            op.y().data().asNioDouble(),
                            op.y().shapeInfo(),
                            op.z().data().asNioDouble(),
                            op.z().shapeInfo(),
                            dimensionBuffer,dimension.length);
                }

            }
            else {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduceScalar(
                            op.opNum(),
                            op.x().data().asNioDouble(),
                            op.x().shapeInfo(),
                            (DoubleBuffer) op.extraArgsBuff()));
                }
                else {
                    loop.execReduce(op.opNum(),
                            op.x().data().asNioDouble(),
                            op.x().shapeInfo(), (DoubleBuffer) op.extraArgsBuff(),
                            op.z().data().asNioDouble(),
                            op.z().shapeInfo(),
                            dimensionBuffer, dimension.length);
                }

            }
        }
        else {
            if(op instanceof Variance) {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execSummaryStatsScalar(
                            op.opNum()
                            ,op.x().data().asNioFloat(),
                            op.x().shapeInfo(),
                            (FloatBuffer) op.extraArgsBuff()));
                }
                else {
                    loop.execSummaryStats(op.opNum(),
                            op.x().data().asNioFloat(),
                            op.x().shapeInfo(), (FloatBuffer) op.extraArgsBuff(),
                            op.z().data().asNioFloat(),
                            op.z().shapeInfo(),
                            dimensionBuffer, dimension.length);
                }

            }

            else if(op.y() != null) {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduce3Scalar(op.opNum(),
                            op.x().data().asNioFloat(),
                            op.x().shapeInfo(),
                            (FloatBuffer)op.extraArgsBuff(),
                            op.y().data().asNioFloat(),
                            op.y().shapeInfo()));
                }
                else {
                    loop.execReduce3(op.opNum(),
                            op.x().data().asNioFloat(),
                            op.x().shapeInfo(), (FloatBuffer) op.extraArgsBuff(),
                            op.y().data().asNioFloat(),
                            op.y().shapeInfo(),
                            op.z().data().asNioFloat(),
                            op.z().shapeInfo(),
                            dimensionBuffer,dimension.length);
                }

            }
            else {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduceScalar(
                            op.opNum(),
                            op.x().data().asNioFloat(),
                            op.x().shapeInfo(),
                            (FloatBuffer) op.extraArgsBuff()));
                }
                else {
                    loop.execReduce(op.opNum(),
                            op.x().data().asNioFloat(),
                            op.x().shapeInfo(), (FloatBuffer) op.extraArgsBuff(),
                            op.z().data().asNioFloat(),
                            op.z().shapeInfo(),
                            dimensionBuffer, dimension.length);
                }

            }
        }

        return ret;
    }

    private void exec(ScalarOp op) {
        if(op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA) {
            super.exec(op);
        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                loop.execScalar(
                        op.opNum()
                        ,op.x().data().asNioDouble(),
                        op.x().shapeInfo(),
                        op.z().data().asNioDouble(),
                        op.z().shapeInfo(),
                        op.scalar().doubleValue(),
                        (DoubleBuffer) op.extraArgsBuff(),
                        op.n());
            }
            else {
                loop.execScalar(
                        op.opNum()
                        ,op.x().data().asNioFloat(),
                        op.x().shapeInfo(),
                        op.z().data().asNioFloat(),
                        op.z().shapeInfo(),
                        op.scalar().floatValue(),
                        (FloatBuffer) op.extraArgsBuff(),
                        op.n());

            }
        }
    }

    private void exec(TransformOp op) {
        if(op.x() instanceof IComplexNDArray ||  executionMode() == ExecutionMode.JAVA) {
            super.exec(op);
        }
        else {

            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.y() != null) {
                    if(op.x().elementWiseStride() >=1 && op.y().elementWiseStride() >= 1) {
                        loop.execPairwiseTransform
                                (op.opNum(),
                                        op.x().data().asNioDouble(),
                                        op.x().elementWiseStride(),
                                        op.y().data().asNioDouble(),
                                        op.y().elementWiseStride(),
                                        op.z().data().asNioDouble(),
                                        op.z().elementWiseStride(),
                                        (DoubleBuffer) op.extraArgsBuff(),
                                        op.n());

                    }
                    else {
                        loop.execPairwiseTransform
                                (op.opNum(),
                                        op.x().data().asNioDouble(),
                                        op.x().elementWiseStride(),
                                        op.y().data().asNioDouble(),
                                        op.y().elementWiseStride(),
                                        op.z().data().asNioDouble(),
                                        op.z().elementWiseStride(),
                                        (DoubleBuffer) op.extraArgsBuff(),
                                        op.n());
                    }

                }
                else {
                    if(op.x().elementWiseStride() >= 1) {
                        loop.execTransform(op.opNum(),
                                op.x().data().asNioDouble(),
                                op.x().elementWiseStride(),
                                op.z().data().asNioDouble(),
                                op.z().elementWiseStride(),
                                (DoubleBuffer) op.extraArgsBuff(), op.n());
                    }
                    else {
                        loop.execTransform(op.opNum(),
                                op.x().data().asNioDouble(),
                                op.x().shapeInfo(),
                                op.z().data().asNioDouble(),
                                op.z().shapeInfo(),
                                (DoubleBuffer) op.extraArgsBuff(), op.n());
                    }

                }
            }
            else {
                if(op.y() != null) {
                    if(op.x().elementWiseStride() >=1 && op.y().elementWiseStride() >= 1) {
                        loop.execPairwiseTransform
                                (op.opNum(),
                                        op.x().data().asNioFloat(),
                                        op.x().elementWiseStride(),
                                        op.y().data().asNioFloat(),
                                        op.y().elementWiseStride(),
                                        op.z().data().asNioFloat(),
                                        op.z().elementWiseStride(),
                                        (FloatBuffer) op.extraArgsBuff(),
                                        op.n());

                    }
                    else {
                        loop.execPairwiseTransform
                                (op.opNum(),
                                        op.x().data().asNioFloat(),
                                        op.x().elementWiseStride(),
                                        op.y().data().asNioFloat(),
                                        op.y().elementWiseStride(),
                                        op.z().data().asNioFloat(),
                                        op.z().elementWiseStride(),
                                        (FloatBuffer) op.extraArgsBuff(),
                                        op.n());
                    }

                }
                else {
                    if(op.x().elementWiseStride() >= 1) {
                        loop.execTransform(op.opNum(),
                                op.x().data().asNioFloat(),
                                op.x().elementWiseStride(),
                                op.z().data().asNioFloat(),
                                op.z().elementWiseStride(),
                                (FloatBuffer) op.extraArgsBuff(), op.n());
                    }
                    else {
                        loop.execTransform(op.opNum(),
                                op.x().data().asNioFloat(),
                                op.x().shapeInfo(),
                                op.z().data().asNioFloat(),
                                op.z().shapeInfo(),
                                (FloatBuffer) op.extraArgsBuff(), op.n());
                    }

                }
            }
        }
    }

    @Override
    public INDArray exec(BroadcastOp op,int...dimension) {
        java.nio.IntBuffer dimensionBuffer = Shape.toBuffer(dimension);
        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            loop.execBroadcast(op.opNum(),
                    op.x().data().asNioDouble()
                    ,op.x().shapeInfo(),
                    op.y().data().asNioDouble(),op.y().shapeInfo()
                    ,op.z().data().asNioDouble(),op.z().shapeInfo(),
                    dimensionBuffer,dimension.length);
        }
        else {
            loop.execBroadcast(op.opNum(),
                    op.x().data().asNioFloat()
                    ,op.x().shapeInfo(),
                    op.y().data().asNioFloat(),op.y().shapeInfo()
                    ,op.z().data().asNioFloat(),op.z().shapeInfo(),
                    dimensionBuffer,dimension.length);
        }

        return op.z();
    }

    private void exec(IndexAccumulation op) {
        if(op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA) {
            super.exec(op);

        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                op.setFinalResult((int) loop.execIndexReduceScalar(
                        op.opNum(),
                        op.x().data().asNioDouble()
                        , op.x().shapeInfo(), (DoubleBuffer) op.extraArgsBuff()));

            }
            else {
                op.setFinalResult((int) loop.execIndexReduceScalar(
                        op.opNum(),
                        op.x().data().asNioFloat()
                        ,op.x().shapeInfo(), (FloatBuffer) op.extraArgsBuff()));
            }

        }
    }

    private void exec(Accumulation op) {
        if(op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA) {
            super.exec(op);

        }
        else {
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op instanceof Variance) {
                    op.setFinalResult(loop.execSummaryStatsScalar(
                            op.opNum(),
                            op.x().data().asNioDouble()
                            , op.x().shapeInfo(), (DoubleBuffer) op.extraArgsBuff()));
                }
                else if(op.y() != null) {
                    op.setFinalResult(loop.execReduce3Scalar(
                            op.opNum(),
                            op.x().data().asNioDouble()
                            ,op.x().shapeInfo(), (DoubleBuffer) op.extraArgsBuff(),
                            op.y().data().asNioDouble(),op.y().shapeInfo()));
                }
                else {
                    op.setFinalResult(loop.execReduceScalar(
                            op.opNum(),
                            op.x().data().asNioDouble()
                            , op.x().shapeInfo(), (DoubleBuffer) op.extraArgsBuff()));
                }
            }
            else {
                if(op instanceof Variance) {
                    op.setFinalResult(loop.execSummaryStatsScalar(
                            op.opNum(),
                            op.x().data().asNioFloat()
                            , op.x().shapeInfo(), null));
                }
                else if(op.y() != null) {
                    op.setFinalResult(loop.execReduce3Scalar(
                            op.opNum(),
                            op.x().data().asNioFloat()
                            ,op.x().shapeInfo(),null,
                            op.y().data().asNioFloat(),op.y().shapeInfo()));
                }
                else {
                    op.setFinalResult(loop.execReduceScalar(
                            op.opNum(),
                            op.x().data().asNioFloat()
                            ,op.x().shapeInfo(),null));
                }
            }
        }
    }
}
