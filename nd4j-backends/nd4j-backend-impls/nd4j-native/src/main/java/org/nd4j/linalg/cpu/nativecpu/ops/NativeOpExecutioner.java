package org.nd4j.linalg.cpu.nativecpu.ops;


import org.apache.commons.math3.util.Pair;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.cpu.nativecpu.CpuTADManager;
import org.nd4j.linalg.cpu.nativecpu.cache.ConstantBuffersCache;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOps;

import java.util.Arrays;


/**
 *
 * Native operation
 * executioner in c++
 *
 * @author Adam Gibson
 */

public class NativeOpExecutioner extends DefaultOpExecutioner {
    private NativeOps loop = new NativeOps();
    private ConstantHandler constantHandler = new ConstantBuffersCache();
    private CpuTADManager tadManager = new CpuTADManager();

    public NativeOpExecutioner() {
        tadManager.init(loop, constantHandler);
    }

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
        else if(op instanceof IndexAccumulation) {
            IndexAccumulation iac = (IndexAccumulation) op;
            exec(iac);  //Currently using DefaultOpExecutioner
        }
        else if(op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;
            exec(broadcastOp,broadcastOp.getDimension());
        }

        return op;
    }


    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        Arrays.sort(dimension);
        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};



        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1,1} : ArrayUtil.removeIndex(op.x().shape(), dimension);
        if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return op.x();


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


        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension).addressPointer();

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer hostTadOffsets = offsets == null ? null : offsets.addressPointer();

        PointerPointer dummy = new PointerPointer(
                hostTadShapeInfo,
                hostTadOffsets
        );

        Pointer x = op.x().data().addressPointer();
        Pointer z = op.z().data().addressPointer();

        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            loop.execIndexReduceDouble(
                    dummy,
                    op.opNum(),
                    x,
                    op.x().shapeInfoDataBuffer().addressPointer(),
                    getPointerForExtraArgs(op),
                    z,
                    op.z().shapeInfoDataBuffer().addressPointer(),
                    dimensionAddress, dimension.length);

        }
        else {
            loop.execIndexReduceFloat(
                    dummy,
                    op.opNum(),
                    op.x().data().addressPointer(),
                    op.x().shapeInfoDataBuffer().addressPointer(),
                    getPointerForExtraArgs(op),
                    op.z().data().addressPointer(),
                    op.z().shapeInfoDataBuffer().addressPointer(),
                    dimensionAddress, dimension.length);

        }
        return op.z();
    }



    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        Arrays.sort(dimension);

        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};


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

        if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return op.noOp();

        INDArray ret = Nd4j.valueArrayOf(retShape,op.zeroDouble());
        op.setZ(ret);


        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer hostTadOffsets = offsets == null ? null : offsets.addressPointer();

        PointerPointer dummy = new PointerPointer(
                hostTadShapeInfo,
                hostTadOffsets
        );

        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension).addressPointer();

        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            if(op instanceof Variance) {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execSummaryStatsScalarDouble(
                            dummy,
                            op.opNum()
                            , op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op), true));
                }
                else {
                    Variance var = (Variance) op;
                    loop.execSummaryStatsDouble(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.z().data().addressPointer(),
                            op.z().shapeInfoDataBuffer().addressPointer(),dimensionAddress,dimension.length,
                            var.isBiasCorrected());
                }

            }

            else if(op.y() != null) {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduce3ScalarDouble(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.y().data().addressPointer(),
                            op.y().shapeInfoDataBuffer().addressPointer()));
                }
                else {
                    loop.execReduce3Double(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.y().data().addressPointer(),
                            op.y().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            op.z().shapeInfoDataBuffer().addressPointer(),
                            dimensionAddress, dimension.length);
                }

            }
            else {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduceScalarDouble(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op)));
                }
                else {
                    loop.execReduceDouble(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(), getPointerForExtraArgs(op),
                            op.z().data().addressPointer(),
                            op.z().shapeInfoDataBuffer().addressPointer(),
                            dimensionAddress, dimension.length);
                }

            }
        }
        else {
            if(op instanceof Variance) {
                Variance variance = (Variance) op;
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execSummaryStatsScalarFloat(
                            dummy,
                            op.opNum()
                            , op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),variance.isBiasCorrected()));
                }
                else {
                    loop.execSummaryStatsFloat(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),getPointerForExtraArgs(op),
                            op.z().data().addressPointer(),
                            op.z().shapeInfoDataBuffer().addressPointer(),
                            dimensionAddress, dimension.length,variance.isBiasCorrected());
                }

            }

            else if(op.y() != null) {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduce3ScalarFloat(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.y().data().addressPointer(),
                            op.y().shapeInfoDataBuffer().addressPointer()));
                }
                else {
                    loop.execReduce3Float(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),getPointerForExtraArgs(op),
                            op.y().data().addressPointer(),
                            op.y().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            op.z().shapeInfoDataBuffer().addressPointer(),
                            dimensionAddress, dimension.length);
                }

            }
            else {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduceScalarFloat(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op)));
                }
                else {
                    loop.execReduceFloat(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.z().data().addressPointer(),
                            op.z().shapeInfoDataBuffer().addressPointer(),
                            dimensionAddress, dimension.length);
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
            PointerPointer dummy = new PointerPointer(new Pointer[] {null});
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.x(). elementWiseStride() >= 1 && !op.isExecSpecial() && op.z(). elementWiseStride() >= 1 && !op.isExecSpecial()) {
                    loop.execScalarDouble(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().elementWiseStride(),
                            op.z().data().addressPointer(),
                            op.z().elementWiseStride(),
                            op.scalar().doubleValue(),
                            getPointerForExtraArgs(op),
                            op.n());
                }
                else
                    loop.execScalarDouble(
                            dummy,
                            op.opNum()
                            , op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            op.z().shapeInfoDataBuffer().addressPointer(),
                            op.scalar().doubleValue(),
                            getPointerForExtraArgs(op));
            }
            else {
                if(op.x(). elementWiseStride() >= 1 && !op.isExecSpecial() && op.z(). elementWiseStride() >= 1 && !op.isExecSpecial()) {
                    loop.execScalarFloat(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer(),
                            op.x().elementWiseStride(),
                            op.z().data().addressPointer(),
                            op.z().elementWiseStride(),
                            op.scalar().floatValue(),
                            getPointerForExtraArgs(op),
                            op.n());
                }
                else
                    loop.execScalarFloat(
                            dummy,
                            op.opNum()
                            , op.x().data().addressPointer(),
                            op.x().shapeInfoDataBuffer().addressPointer(),
                            op.z().data().addressPointer(),
                            op.z().shapeInfoDataBuffer().addressPointer(),
                            op.scalar().floatValue(),
                            getPointerForExtraArgs(op));

            }
        }
    }

    private Pointer getPointerForExtraArgs(Op op) {
        if(op.extraArgs() != null)
            return op.extraArgsDataBuff().addressPointer();
        return null;
    }

    private void exec(TransformOp op) {
            PointerPointer dummy = new PointerPointer(new Pointer[] {null});
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.y() != null) {
                    if(op.x().elementWiseStride() >=1 && op.y(). elementWiseStride() >= 1 && op.x().elementWiseStride() == op.y(). elementWiseStride()  && !op.isExecSpecial() && op.x().ordering() == op.y().ordering() && op.x().ordering() == op.z().ordering()) {
                        loop.execPairwiseTransformDouble(
                                dummy,
                                op.opNum(),
                                op.x().data().addressPointer(),
                                op.x().elementWiseStride(),
                                op.y().data().addressPointer(),
                                op.y().elementWiseStride(),
                                op.z().data().addressPointer(),
                                op.z().elementWiseStride(),
                                getPointerForExtraArgs(op),
                                op.n());

                    }
                    else {
                        loop.execPairwiseTransformDouble(
                                dummy,
                                op.opNum(),
                                op.x().data().addressPointer(),
                                op.x().shapeInfoDataBuffer().addressPointer(),
                                op.y().data().addressPointer(),
                                op.y().shapeInfoDataBuffer().addressPointer(),
                                op.z().data().addressPointer(),
                                op.z().shapeInfoDataBuffer().addressPointer(),
                                getPointerForExtraArgs(op));
                    }

                }
                else {
                    if(op.x(). elementWiseStride() >= 1 && !op.isExecSpecial() && !op.isExecSpecial() && op.x().ordering() == op.z().ordering()) {
                        loop.execTransformDouble(
                                dummy,
                                op.opNum(),
                                op.x().data().addressPointer(),
                                op.x().elementWiseStride(),
                                op.z().data().addressPointer(),
                                op.z().elementWiseStride(),
                                getPointerForExtraArgs(op), op.n());
                    }
                    else {
                        loop.execTransformDouble(
                                dummy,
                                op.opNum(),
                                op.x().data().addressPointer(),
                                op.x().shapeInfoDataBuffer().addressPointer(),
                                op.z().data().addressPointer(),
                                op.z().shapeInfoDataBuffer().addressPointer(),
                                getPointerForExtraArgs(op));
                    }

                }
            }
            else {
                if(op.y() != null) {
                    if(op.x().elementWiseStride() >=1 && op.y(). elementWiseStride() >= 1 && op.x().elementWiseStride() == op.y(). elementWiseStride() && !op.isExecSpecial() && op.x().ordering() == op.y().ordering()) {
                        loop.execPairwiseTransformFloat
                                (dummy,op.opNum(),
                                        op.x().data().addressPointer(),
                                        op.x().elementWiseStride(),
                                        op.y().data().addressPointer(),
                                        op.y().elementWiseStride(),
                                        op.z().data().addressPointer(),
                                        op.z().elementWiseStride(),
                                        getPointerForExtraArgs(op),
                                        op.n());

                    }
                    else {
                        loop.execPairwiseTransformFloat(
                                dummy,
                                op.opNum(),
                                op.x().data().addressPointer(),
                                op.x().shapeInfoDataBuffer().addressPointer(),
                                op.y().data().addressPointer(),
                                op.y().shapeInfoDataBuffer().addressPointer(),
                                op.z().data().addressPointer(),
                                op.z().shapeInfoDataBuffer().addressPointer(),
                                getPointerForExtraArgs(op));
                    }

                }
                else {
                    if(op.x(). elementWiseStride() >= 1 && !op.isExecSpecial() && op.x().ordering() == op.z().ordering()) {
                        loop.execTransformFloat(dummy,op.opNum(),
                                op.x().data().addressPointer(),
                                op.x().elementWiseStride(),
                                op.z().data().addressPointer(),
                                op.z().elementWiseStride(),
                                getPointerForExtraArgs(op), op.n());
                    }
                    else {
                        loop.execTransformFloat(
                                dummy,
                                op.opNum(),
                                op.x().data().addressPointer(),
                                op.x().shapeInfoDataBuffer().addressPointer(),
                                op.z().data().addressPointer(),
                                op.z().shapeInfoDataBuffer().addressPointer(),
                                getPointerForExtraArgs(op));
                    }

                }
            }

    }

    @Override
    public INDArray exec(BroadcastOp op,int...dimension) {
        Arrays.sort(dimension);

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();
        Pointer hostTadOffsets = tadBuffers.getSecond().addressPointer();

        PointerPointer dummy = new PointerPointer(
                hostTadShapeInfo,
                hostTadOffsets
        );

        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension).addressPointer();

        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            loop.execBroadcastDouble(dummy,op.opNum(),
                    op.x().data().addressPointer()
                    ,op.x().shapeInfoDataBuffer().addressPointer(),
                    op.y().data().addressPointer(), op.y().shapeInfoDataBuffer().addressPointer()
                    , op.z().data().addressPointer(), op.z().shapeInfoDataBuffer().addressPointer(),
                    dimensionAddress, dimension.length);
        }
        else {
            loop.execBroadcastFloat(dummy,op.opNum(),
                    op.x().data().addressPointer()
                    ,op.x().shapeInfoDataBuffer().addressPointer(),
                    op.y().data().addressPointer(),
                    op.y().shapeInfoDataBuffer().addressPointer()
                    , op.z().data().addressPointer(),
                    op.z().shapeInfoDataBuffer().addressPointer(),
                    dimensionAddress, dimension.length);
        }

        return op.z();
    }

    private void exec(IndexAccumulation op) {
        if(op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA) {
            super.exec(op);

        }
        else {
            PointerPointer dummy = new PointerPointer(new Pointer[] {null});
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                op.setFinalResult((int) loop.execIndexReduceScalarDouble(
                        dummy,
                        op.opNum(),
                        op.x().data().addressPointer()
                        ,op.x().shapeInfoDataBuffer().addressPointer(), getPointerForExtraArgs(op)));

            }
            else {
                op.setFinalResult((int) loop.execIndexReduceScalarFloat(
                        dummy,
                        op.opNum(),
                        op.x().data().addressPointer()
                        ,op.x().shapeInfoDataBuffer().addressPointer(),
                        getPointerForExtraArgs(op)));
            }

        }
    }

    private void exec(Accumulation op) {
        if(op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA) {
            super.exec(op);

        }
        else {
            PointerPointer dummy = new PointerPointer(new Pointer[] {null});
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op instanceof Variance) {
                    op.setFinalResult(loop.execSummaryStatsScalarDouble(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer()
                            ,op.x().shapeInfoDataBuffer().addressPointer(), getPointerForExtraArgs(op), true));
                }
                else if(op.y() != null) {
                    op.setFinalResult(loop.execReduce3ScalarDouble(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer()
                            ,op.x().shapeInfoDataBuffer().addressPointer(),getPointerForExtraArgs(op),
                            op.y().data().addressPointer(), op.y().shapeInfoDataBuffer().addressPointer()));
                }
                else {
                    op.setFinalResult(loop.execReduceScalarDouble(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer()
                            ,op.x().shapeInfoDataBuffer().addressPointer(), getPointerForExtraArgs(op)));
                }
            }
            else {
                if(op instanceof Variance) {
                    Variance variance = (Variance) op;
                    op.setFinalResult(loop.execSummaryStatsScalarFloat(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer()
                            ,op.x().shapeInfoDataBuffer().addressPointer(),  getPointerForExtraArgs(op),variance.isBiasCorrected()));
                }
                else if(op.y() != null) {
                    op.setFinalResult(loop.execReduce3ScalarFloat(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer()
                            ,op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op),
                            op.y().data().addressPointer(),
                            op.y().shapeInfoDataBuffer().addressPointer()));
                }
                else {
                    op.setFinalResult(loop.execReduceScalarFloat(
                            dummy,
                            op.opNum(),
                            op.x().data().addressPointer()
                            ,op.x().shapeInfoDataBuffer().addressPointer(),
                            getPointerForExtraArgs(op)));
                }
            }
        }
    }
}
