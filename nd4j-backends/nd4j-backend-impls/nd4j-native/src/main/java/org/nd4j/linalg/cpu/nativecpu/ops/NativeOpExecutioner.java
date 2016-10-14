package org.nd4j.linalg.cpu.nativecpu.ops;


import lombok.Getter;
import org.apache.commons.math3.util.Pair;
import org.bytedeco.javacpp.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.cpu.nativecpu.CpuTADManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;


/**
 *
 * Native operation
 * executioner in c++
 *
 * @author Adam Gibson
 */

public class NativeOpExecutioner extends DefaultOpExecutioner {
    private NativeOps loop = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private ConstantHandler constantHandler = Nd4j.getConstantHandler();
    @Getter private CpuTADManager tadManager = new CpuTADManager();

    public NativeOpExecutioner() {
        tadManager.init(loop, constantHandler);
    }

    @Override
    public Op exec(Op op) {
        checkForCompression(op);

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
        if (dimension == null || dimension.length == 0)
            dimension = new int[]{Integer.MAX_VALUE};

        checkForCompression(op);

        Arrays.sort(dimension);
        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};



        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1,1} : ArrayUtil.removeIndex(op.x().shape(), dimension);

        // This is obviously wrong for IndexReduce, op.x has no real value as return
        // if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
        //     return op.x();


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
            if (op.z().isScalar()) {
                int res = (int) loop.execIndexReduceScalarDouble(
                        dummy,
                        op.opNum(),
                        (DoublePointer)op.x().data().addressPointer(),
                        (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(), (DoublePointer)getPointerForExtraArgs(op));


                op.setFinalResult(res);
                op.z().putScalar(0, (float) res);
            } else {
                loop.execIndexReduceDouble(
                        dummy,
                        op.opNum(),
                        (DoublePointer)x,
                        (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                        (DoublePointer)getPointerForExtraArgs(op),
                        (DoublePointer)z,
                        (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                        (IntPointer)dimensionAddress, dimension.length);
            }

        }
        else {
            if (op.z().isScalar()) {
                int res = (int) loop.execIndexReduceScalarFloat(
                        dummy,
                        op.opNum(),
                        (FloatPointer)op.x().data().addressPointer(),
                        (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(), (FloatPointer)getPointerForExtraArgs(op));

                op.setFinalResult(res);
                op.z().putScalar(0, (float) res);
            } else {
                loop.execIndexReduceFloat(
                        dummy,
                        op.opNum(),
                        (FloatPointer)op.x().data().addressPointer(),
                        (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                        (FloatPointer)getPointerForExtraArgs(op),
                        (FloatPointer)op.z().data().addressPointer(),
                        (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                        (IntPointer)dimensionAddress, dimension.length);
            }

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
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer)getPointerForExtraArgs(op), true));
                }
                else {
                    Variance var = (Variance) op;
                    loop.execSummaryStatsDouble(
                            dummy,
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer)getPointerForExtraArgs(op),
                            (DoublePointer)op.z().data().addressPointer(),
                            (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),(IntPointer)dimensionAddress,dimension.length,
                            var.isBiasCorrected());
                }

            }

            else if(op.y() != null) {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduce3ScalarDouble(
                            dummy,
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer)getPointerForExtraArgs(op),
                            (DoublePointer)op.y().data().addressPointer(),
                            (IntPointer)op.y().shapeInfoDataBuffer().addressPointer()));
                }
                else {
                    loop.execReduce3Double(
                            dummy,
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer)getPointerForExtraArgs(op),
                            (DoublePointer)op.y().data().addressPointer(),
                            (IntPointer)op.y().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer)op.z().data().addressPointer(),
                            (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                            (IntPointer)dimensionAddress, dimension.length);
                }

            }
            else {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduceScalarDouble(
                            dummy,
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer)getPointerForExtraArgs(op)));
                }
                else {
                    loop.execReduceDouble(
                            dummy,
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(), (DoublePointer)getPointerForExtraArgs(op),
                            (DoublePointer)op.z().data().addressPointer(),
                            (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                            (IntPointer)dimensionAddress, dimension.length);
                }

            }
        }
        else {
            if(op instanceof Variance) {
                Variance variance = (Variance) op;
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execSummaryStatsScalarFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (FloatPointer)getPointerForExtraArgs(op),variance.isBiasCorrected()));
                }
                else {
                    loop.execSummaryStatsFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),(FloatPointer)getPointerForExtraArgs(op),
                            (FloatPointer)op.z().data().addressPointer(),
                            (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                            (IntPointer)dimensionAddress, dimension.length,variance.isBiasCorrected());
                }

            }

            else if(op.y() != null) {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduce3ScalarFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (FloatPointer)getPointerForExtraArgs(op),
                            (FloatPointer)op.y().data().addressPointer(),
                            (IntPointer)op.y().shapeInfoDataBuffer().addressPointer()));
                }
                else {
                    loop.execReduce3Float(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),(FloatPointer)getPointerForExtraArgs(op),
                            (FloatPointer)op.y().data().addressPointer(),
                            (IntPointer)op.y().shapeInfoDataBuffer().addressPointer(),
                            (FloatPointer)op.z().data().addressPointer(),
                            (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                            (IntPointer)dimensionAddress, dimension.length);
                }

            }
            else {
                if(ret.isScalar()) {
                    ret.putScalar(0,loop.execReduceScalarFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (FloatPointer)getPointerForExtraArgs(op)));
                }
                else {
                    loop.execReduceFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (FloatPointer)getPointerForExtraArgs(op),
                            (FloatPointer)op.z().data().addressPointer(),
                            (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                            (IntPointer)dimensionAddress, dimension.length);
                }

            }
        }

        return ret;
    }

    /**
     * ScalarOp along dimension
     * @param op
     * @param dimension
     */
    private void invoke(ScalarOp op, int[] dimension) {
        Arrays.sort(dimension);
        // do tad magic

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();
        Pointer hostTadOffsets = tadBuffers.getSecond().addressPointer();

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = tadBuffersZ.getFirst().addressPointer();
        devTadOffsetsZ = tadBuffersZ.getSecond().addressPointer();

        PointerPointer dummy = new PointerPointer(
                hostTadShapeInfo,
                hostTadOffsets,
                devTadShapeInfoZ,
                devTadOffsetsZ
        );


        if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
            loop.execScalarFloat(dummy,
                    op.opNum(),
                    (FloatPointer) op.x().data().addressPointer(),
                    (IntPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                    (FloatPointer) op.z().data().addressPointer(),
                    (IntPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                    (FloatPointer) op.y().data().addressPointer(),
                    (FloatPointer) getPointerForExtraArgs(op),
                    new IntPointer(dimension),
                    dimension.length
                    );
        } else if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            loop.execScalarDouble(dummy,
                    op.opNum(),
                    (DoublePointer) op.x().data().addressPointer(),
                    (IntPointer) op.x().shapeInfoDataBuffer().addressPointer(),
                    (DoublePointer) op.z().data().addressPointer(),
                    (IntPointer) op.z().shapeInfoDataBuffer().addressPointer(),
                    (DoublePointer) op.y().data().addressPointer(),
                    (DoublePointer) getPointerForExtraArgs(op),
                    new IntPointer(dimension),
                    dimension.length
            );
        }
    }

    private void exec(ScalarOp op) {
        if(op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA) {
            super.exec(op);
        }
        else {
            if (op.getDimension() != null) {
                invoke(op, op.getDimension());
                return;
            }
            PointerPointer dummy = new PointerPointer(new Pointer[] {null});
            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.x(). elementWiseStride() >= 1 && !op.isExecSpecial() && op.z(). elementWiseStride() >= 1 && !op.isExecSpecial()) {
                    loop.execScalarDouble(
                            dummy,
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            op.x().elementWiseStride(),
                            (DoublePointer)op.z().data().addressPointer(),
                            op.z().elementWiseStride(),
                            op.scalar().doubleValue(),
                            (DoublePointer)getPointerForExtraArgs(op),
                            op.n());
                }
                else
                    loop.execScalarDouble(
                            dummy,
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (DoublePointer)op.z().data().addressPointer(),
                            (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                            op.scalar().doubleValue(),
                            (DoublePointer)getPointerForExtraArgs(op));
            }
            else {
                if(op.x(). elementWiseStride() >= 1 && !op.isExecSpecial() && op.z(). elementWiseStride() >= 1 && !op.isExecSpecial()) {
                    loop.execScalarFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            op.x().elementWiseStride(),
                            (FloatPointer)op.z().data().addressPointer(),
                            op.z().elementWiseStride(),
                            op.scalar().floatValue(),
                            (FloatPointer)getPointerForExtraArgs(op),
                            op.n());
                }
                else
                    loop.execScalarFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (FloatPointer)op.z().data().addressPointer(),
                            (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                            op.scalar().floatValue(),
                            (FloatPointer)getPointerForExtraArgs(op));

            }
        }
    }

    private Pointer getPointerForExtraArgs(Op op) {
        if(op.extraArgs() != null)
            return op.extraArgsDataBuff().addressPointer();
        return null;
    }

    private void exec(TransformOp op) {
            PointerPointer dummy = new PointerPointer(4);

        if(op.opNum() == 41 && op.extraArgs() != null) {
            int[] dimension = new int[] {(int) op.extraArgs()[1] };

            Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.z(), dimension);


            Pointer tad = tadBuffers.getFirst().addressPointer();

            DataBuffer offsets = tadBuffers.getSecond();
            Pointer off = offsets == null ? null : offsets.addressPointer();
            dummy.put(0, tad);
            dummy.put(1, off);
        }

            if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                if(op.y() != null) {
                    if(op.x().elementWiseStride() >=1 && op.y(). elementWiseStride() >= 1 && op.x().elementWiseStride() == op.y(). elementWiseStride()  && !op.isExecSpecial() && op.x().ordering() == op.y().ordering() && op.x().ordering() == op.z().ordering()) {
                        loop.execPairwiseTransformDouble(
                                dummy,
                                op.opNum(),
                                (DoublePointer)op.x().data().addressPointer(),
                                op.x().elementWiseStride(),
                                (DoublePointer)op.y().data().addressPointer(),
                                op.y().elementWiseStride(),
                                (DoublePointer)op.z().data().addressPointer(),
                                op.z().elementWiseStride(),
                                (DoublePointer)getPointerForExtraArgs(op),
                                op.n());

                    }
                    else {
                        loop.execPairwiseTransformDouble(
                                dummy,
                                op.opNum(),
                                (DoublePointer)op.x().data().addressPointer(),
                                (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                                (DoublePointer)op.y().data().addressPointer(),
                                (IntPointer)op.y().shapeInfoDataBuffer().addressPointer(),
                                (DoublePointer)op.z().data().addressPointer(),
                                (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                                (DoublePointer)getPointerForExtraArgs(op));
                    }

                }
                else {
                    if(op.x(). elementWiseStride() >= 1 && !op.isExecSpecial() && !op.isExecSpecial() && op.x().ordering() == op.z().ordering()) {
                        loop.execTransformDouble(
                                dummy,
                                op.opNum(),
                                (DoublePointer)op.x().data().addressPointer(),
                                op.x().elementWiseStride(),
                                (DoublePointer)op.z().data().addressPointer(),
                                op.z().elementWiseStride(),
                                (DoublePointer)getPointerForExtraArgs(op), op.n());
                    }
                    else {
                        loop.execTransformDouble(
                                dummy,
                                op.opNum(),
                                (DoublePointer)op.x().data().addressPointer(),
                                (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                                (DoublePointer)op.z().data().addressPointer(),
                                (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                                (DoublePointer)getPointerForExtraArgs(op));
                    }

                }
            }
            else {
                if(op.y() != null) {
                    if(op.x().elementWiseStride() >=1 && op.y(). elementWiseStride() >= 1 && op.x().elementWiseStride() == op.y(). elementWiseStride() && !op.isExecSpecial() && op.x().ordering() == op.y().ordering()) {
                        loop.execPairwiseTransformFloat
                                (dummy,op.opNum(),
                                        (FloatPointer)op.x().data().addressPointer(),
                                        op.x().elementWiseStride(),
                                        (FloatPointer)op.y().data().addressPointer(),
                                        op.y().elementWiseStride(),
                                        (FloatPointer)op.z().data().addressPointer(),
                                        op.z().elementWiseStride(),
                                        (FloatPointer)getPointerForExtraArgs(op),
                                        op.n());

                    }
                    else {
                        loop.execPairwiseTransformFloat(
                                dummy,
                                op.opNum(),
                                (FloatPointer)op.x().data().addressPointer(),
                                (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                                (FloatPointer)op.y().data().addressPointer(),
                                (IntPointer)op.y().shapeInfoDataBuffer().addressPointer(),
                                (FloatPointer)op.z().data().addressPointer(),
                                (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                                (FloatPointer)getPointerForExtraArgs(op));
                    }

                }
                else {
                    if(op.x(). elementWiseStride() >= 1 && !op.isExecSpecial() && op.x().ordering() == op.z().ordering()) {
                        loop.execTransformFloat(dummy,op.opNum(),
                                (FloatPointer)op.x().data().addressPointer(),
                                op.x().elementWiseStride(),
                                (FloatPointer)op.z().data().addressPointer(),
                                op.z().elementWiseStride(),
                                (FloatPointer)getPointerForExtraArgs(op), op.n());
                    }
                    else {
                        loop.execTransformFloat(
                                dummy,
                                op.opNum(),
                                (FloatPointer)op.x().data().addressPointer(),
                                (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                                (FloatPointer)op.z().data().addressPointer(),
                                (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                                (FloatPointer)getPointerForExtraArgs(op));
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

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

//        if (!Arrays.equals(op.x().shape(),op.z().shape()) || !Arrays.equals(op.x().stride(),op.z().stride()) || op.x().ordering() != op.z().ordering()) {
        // that's the place where we're going to have second TAD in place
        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = tadBuffersZ.getFirst().addressPointer();
        devTadOffsetsZ = tadBuffersZ.getSecond().addressPointer();

        PointerPointer dummy = new PointerPointer(
                hostTadShapeInfo,
                hostTadOffsets,
                devTadShapeInfoZ,
                devTadOffsetsZ
        );

        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension).addressPointer();

        if(op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
            loop.execBroadcastDouble(dummy,op.opNum(),
                    (DoublePointer)op.x().data().addressPointer(),
                    (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                    (DoublePointer)op.y().data().addressPointer(), (IntPointer)op.y().shapeInfoDataBuffer().addressPointer(),
                    (DoublePointer)op.z().data().addressPointer(), (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                    (IntPointer)dimensionAddress, dimension.length);
        }
        else {
            loop.execBroadcastFloat(dummy,op.opNum(),
                    (FloatPointer)op.x().data().addressPointer(),
                    (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                    (FloatPointer)op.y().data().addressPointer(),
                    (IntPointer)op.y().shapeInfoDataBuffer().addressPointer(),
                    (FloatPointer)op.z().data().addressPointer(),
                    (IntPointer)op.z().shapeInfoDataBuffer().addressPointer(),
                    (IntPointer)dimensionAddress, dimension.length);
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
                        (DoublePointer)op.x().data().addressPointer(),
                        (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(), (DoublePointer)getPointerForExtraArgs(op)));

            }
            else {
                op.setFinalResult((int) loop.execIndexReduceScalarFloat(
                        dummy,
                        op.opNum(),
                        (FloatPointer)op.x().data().addressPointer(),
                        (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                        (FloatPointer)getPointerForExtraArgs(op)));
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
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(), (DoublePointer)getPointerForExtraArgs(op), true));
                }
                else if(op.y() != null) {
                    op.setFinalResult(loop.execReduce3ScalarDouble(
                            dummy,
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(), (DoublePointer)getPointerForExtraArgs(op),
                            (DoublePointer)op.y().data().addressPointer(), (IntPointer)op.y().shapeInfoDataBuffer().addressPointer()));
                }
                else {
                    op.setFinalResult(loop.execReduceScalarDouble(
                            dummy,
                            op.opNum(),
                            (DoublePointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(), (DoublePointer)getPointerForExtraArgs(op)));
                }
            }
            else {
                if(op instanceof Variance) {
                    Variance variance = (Variance) op;
                    op.setFinalResult(loop.execSummaryStatsScalarFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),  (FloatPointer)getPointerForExtraArgs(op),variance.isBiasCorrected()));
                }
                else if(op.y() != null) {
                    op.setFinalResult(loop.execReduce3ScalarFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (FloatPointer)getPointerForExtraArgs(op),
                            (FloatPointer)op.y().data().addressPointer(),
                            (IntPointer)op.y().shapeInfoDataBuffer().addressPointer()));
                }
                else {
                    op.setFinalResult(loop.execReduceScalarFloat(
                            dummy,
                            op.opNum(),
                            (FloatPointer)op.x().data().addressPointer(),
                            (IntPointer)op.x().shapeInfoDataBuffer().addressPointer(),
                            (FloatPointer)getPointerForExtraArgs(op)));
                }
            }
        }
    }

    @Override
    public void exec(Batch batch) {
        if (batch.size() == 0)
            return;

        int[] ops = new int[batch.size()];
        int[] numShapes = new int[batch.size()];
        int[] numArguments = new int[batch.size()];
        int[] numIndexingArguments = new int[batch.size()];
        int[] numRealArguments = new int[batch.size()];
        PointerPointer argumentsPointer = new PointerPointer(batch.size());
        PointerPointer shapesPointer = new PointerPointer(batch.size());
        PointerPointer indexingPointer = new PointerPointer(batch.size());
        PointerPointer realPointer = new PointerPointer(batch.size());

        List<INDArray> arraysHolder = new ArrayList<>();
        List<Pointer> pointersHolder = new ArrayList<>();
        List<double[]> realsHolder = new ArrayList<>();

        for (int e = 0; e < batch.size(); e++) {
            Aggregate op = batch.getAggregates().get(e);
            ops[e] = op.opNum();
            numArguments[e] = op.getArguments().size();
            numIndexingArguments[e] = op.getIndexingArguments().size();
            numRealArguments[e] = op.getRealArguments().size();

            long[] arguments = new long[numArguments[e]];

            for (int x = 0; x < numArguments[e]; x++ ) {
                arguments[x] = op.getArguments().get(x).data().addressPointer().address();
            }

            argumentsPointer.put(e, new LongPointer(arguments));

            long[] shapes = new long[numShapes[e]];
            for (int x = 0; x < numShapes[e]; x++ ) {
                shapes[x] = op.getShapes().get(x).addressPointer().address();
            }

            shapesPointer.put(e, new LongPointer(shapes));

            int[] indexes = new int[numIndexingArguments[e]];
            for (int x = 0; x < numIndexingArguments[e]; x++) {
                indexes[x] = op.getIndexingArguments().get(x);
            }

            IntPointer idxPointer = new IntPointer(indexes);
            indexingPointer.put(e, idxPointer);
            pointersHolder.add(idxPointer);

            double[] reals = new double[numRealArguments[e]];
            for (int x = 0; x < numRealArguments[e]; x++) {
                reals[x] = op.getRealArguments().get(x);
            }

            INDArray realsBuffer = Nd4j.create(reals);
            arraysHolder.add(realsBuffer);

            realsHolder.add(reals);

            realPointer.put(e, realsBuffer.data().addressPointer());
        }

        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            loop.execAggregateBatchFloat(
                    null,
                    batch.size(),
                    new IntPointer(ops),
                    argumentsPointer,
                    new IntPointer(numArguments),
                    shapesPointer,
                    new IntPointer(numShapes),
                    indexingPointer,
                    new IntPointer(numIndexingArguments),
                    realPointer,
                    new IntPointer(numRealArguments)
            );
        } else {
            throw new RuntimeException("DOUBLE not implemented here yet");
        }

        argumentsPointer.sizeof();
        arraysHolder.size();
        indexingPointer.sizeof();
        realPointer.sizeof();
        batch.size();
    }

    @Override
    public void exec(Aggregate op) {

        int numArguments = op.getArguments().size();
        int numIndexArguments = op.getIndexingArguments().size();
        int numRealArguments = op.getRealArguments().size();
        int numShapes = op.getShapes().size();

        PointerPointer arguments = new PointerPointer(numArguments);

        for (int x = 0; x < numArguments; x++ ) {
            arguments.put(x, op.getArguments().get(x).data().addressPointer());
        }

        PointerPointer shapes = new PointerPointer(numShapes);

        for (int x = 0; x < numShapes; x++ ) {
            if (op.getShapes().get(x).dataType() != DataBuffer.Type.INT)
                throw new RuntimeException("ShapeBuffers should have INT data type");

            shapes.put(x, op.getShapes().get(x).addressPointer());
        }

        int[] indexes = new int[numIndexArguments];
        for (int x = 0; x < numIndexArguments; x++) {
            indexes[x] = op.getIndexingArguments().get(x);
        }

        IntPointer pointer = new IntPointer(indexes);

        double[] reals = new double[numRealArguments];
        for (int x = 0; x < numRealArguments; x++) {
            reals[x] = op.getRealArguments().get(x);
        }

        INDArray realsBuffer = Nd4j.create(reals);


        if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            loop.execAggregateFloat(null, op.opNum(),
                    arguments,
                    numArguments,
                    shapes,
                    numShapes,
                    pointer,
                    numIndexArguments,
                    (FloatPointer) realsBuffer.data().addressPointer(),
                    numRealArguments
                    );
        } else if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {

        }
    }

    /**
     * This method return set of key/value and key/key/value objects, describing current environment
     *
     * @return
     */
    @Override
    public Properties getEnvironmentInformation() {
        Properties properties = super.getEnvironmentInformation();

        properties.put("backend","CPU");
        properties.put("omp.threads", loop.ompGetMaxThreads());
        properties.put("blas.threads", NativeOpsHolder.getInstance().getDeviceNativeBlas().getMaxThreads());
        properties.put("blas.vendor", NativeOpsHolder.getInstance().getDeviceNativeBlas().getBlasVendor().toString());

        return properties;
    }
}
