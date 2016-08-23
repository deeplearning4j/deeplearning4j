package org.nd4j.linalg.jcublas.ops.executioner;

import org.apache.commons.math3.util.Pair;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;
import org.nd4j.linalg.api.ops.impl.meta.InvertedPredicateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.PostulateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.PredicateMetaOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Deque;

/**
 * mGRID implementation for CUDA
 *
 * PLEASE NOTE: WORK IN PROGRESS, DO NOT EVER USE THIS EXECUTIONER IN PRODUCTION
 * @author raver119@gmail.com
 */
public class CudaGridExecutioner extends CudaExecutioner implements GridExecutioner {
    protected enum MetaType {
        NOT_APPLICABLE,
        PREDICATE,
        INVERTED_PREDICATE,
        POSTULATE,
    }

    // general queues
    //private List<Deque<OpDescriptor>> deviceQueues = new ArrayList<>();

    // last op
    private ThreadLocal<OpDescriptor> lastOp = new ThreadLocal<>();
    private ThreadLocal<PointerPointer> extraz = new ThreadLocal<>();
    private ThreadLocal<Deque<OpDescriptor>> deviceQueues = new ThreadLocal<>();
    private PointerPointer exxtrazz = new PointerPointer(4);

    private static Logger logger = LoggerFactory.getLogger(CudaGridExecutioner.class);

    public CudaGridExecutioner() {
        extraz.set(new PointerPointer(4));
    }

    /**
     * This is one of the main entry points for ops that are executed without respect to dimension.
     *
     * Developers note: For CudaGridExecutioner that's also the MetaOp/GridOp creation point.
     *
     * @param op
     * @return
     */
    @Override
    public Op exec(Op op) {
        /*
            We pass this op to GridProcessor through check for possible MetaOp concatenation
            Also, it's the GriOp entry point
         */

        if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            exec(acc, new int[]{Integer.MAX_VALUE});
        } else if (op instanceof IndexAccumulation) {
            IndexAccumulation acc = (IndexAccumulation) op;
            exec(acc, new int[]{Integer.MAX_VALUE});
        } else if (op instanceof ScalarOp){
            // the only entry place for TADless ops
            processAsGridOp(op);
        } else if (op instanceof BroadcastOp) {
            invoke((BroadcastOp) op);

        }

        return op;
    }

    /**
     * This method adds op into GridOp queue
     *
     * @return
     */
    protected void pushToGrid(OpDescriptor descriptor) {

        // we should just add op to queue here
        //deviceQueues.get().add(descriptor);

        // FIXME: following code should be removed, since it's just executing supers instead of batching
        Op op = descriptor.getOp();
        int[] dimensions = descriptor.getDimensions();

        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;

            super.exec(t);
        } else if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;

            super.exec(acc, dimensions);
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;

            super.exec(sc);
        } else if(op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;

            super.exec(broadcastOp, dimensions);
        }
        else if(op instanceof IndexAccumulation) {
            IndexAccumulation indexAccumulation = (IndexAccumulation) op;

            super.exec(indexAccumulation, dimensions);
        } else if (op instanceof MetaOp) {

            exec((MetaOp) op);
        } else if (op instanceof GridOp) {
            exec((GridOp) op);
        }
    }

    protected void processAsGridOp(Op op, int... dimension) {
        /*
            We have multiple options here:
                1) Op has no relation to lastOp
                2) Op has SOME relation to lastOp
                3) Op is supposed to blocking

            So we either should append this op to future GridOp, form MetaOp, or immediately execute everything
            But we don't expect this method called for blocking ops ever, so it's either
        */

        OpDescriptor last = lastOp.get();
        if (last != null) {
            MetaType type = getMetaOpType(op, dimension);
            switch (type) {
                case NOT_APPLICABLE: {
                    /*
                        If we can't form MetaOp with new Op here, we should move lastOp to GridOp queue, and update lastOp with current Op
                    */
                    lastOp.set(new OpDescriptor(op, dimension));

                    if (last != null)
                        pushToGrid(last);
                }
                break;
                case PREDICATE: {
                    lastOp.remove();

                    MetaOp metaOp = new PredicateMetaOp(last, new OpDescriptor(op, dimension));
                    pushToGrid(new OpDescriptor(metaOp));
                }
                break;
                case INVERTED_PREDICATE: {
                    lastOp.remove();

                    MetaOp metaOp = new InvertedPredicateMetaOp(last, new OpDescriptor(op, dimension));
                    pushToGrid(new OpDescriptor(metaOp));
                }
                break;
                case POSTULATE: {
                    lastOp.remove();

                    MetaOp metaOp = new PostulateMetaOp(last, new OpDescriptor(op, dimension));
                    pushToGrid(new OpDescriptor(metaOp));
                }
                break;
                default:
                    throw new UnsupportedOperationException("Not supported MetaType: [" + type + "]");
            }
        } else {
            lastOp.set(new OpDescriptor(op, dimension));
        }

        //return op;
    }

    protected MetaType getMetaOpType(Op op, int... dimension) {
        OpDescriptor last = lastOp.get();

        if (last == null) {
            logger.info("last is NULL");
            return MetaType.NOT_APPLICABLE;
        } else {
            // TODO: it's still possible to use InvertedPredicates on op.Y, but it requires investigation

            if (last.getOp() instanceof ScalarOp || last.getOp() instanceof TransformOp) {
                /*
                    Predicate logic is simple:
                        1) LastOp is one of following op types: Scalar, Transform, PairwiseTransform
                        2) LastOp isn't specialOp
                        3) LastOp op.x() == op.z()
                        4) currentOp op.x() == op.z(), and matches lastOp op.z()
                */

                return isMatchingZX(last.getOp(), op) ? MetaType.PREDICATE: MetaType.NOT_APPLICABLE;
            } else if (last.getOp() instanceof Accumulation) {
                /*
                    InvertedMetaOp, aka Postulate logic

                    Postulate logic is simple too:
                        1) LastOp is type of Reduce or Reduce3
                        2) LastOp op.z() isn't scalar
                        3) currentOp is one of the following op types: Scalar, Transform
                 */
                if ((op instanceof ScalarOp || op instanceof TransformOp) && op.y() == null)
                    return isMatchingZX(last.getOp(), op) ? MetaType.POSTULATE : MetaType.NOT_APPLICABLE;
            }
        }

        return MetaType.NOT_APPLICABLE;
    }

    /**
     * This method checks, if opA and opB are sharing the same operands
     *
     * @param opA
     * @param opB
     * @return
     */
    protected boolean isMatchingZX(Op opA, Op opB) {
        if (opA.z() == opB.x() && opA.x() == opB.z())
            return true;

        return false;
    }

    /**
     * This method is additional check, basically it qualifies possibility of InvertedPredicate MetaOp
     *
     * @param opA
     * @param opB
     * @return
     */
    protected boolean isMatchingZXY(Op opA, Op opB) {
        if (opA.z() == opB.x() || opA.z() == opB.y())
            return true;

        return false;
    }

    /**
     * This method returns Op as set of required pointers for it
     * @param op
     * @param dimensions
     * @return
     */
    protected GridPointers pointerizeOp(Op op, int... dimensions) {
        GridPointers pointers = new GridPointers(op, dimensions);

        AtomicAllocator allocator = AtomicAllocator.getInstance();

        // FIXME: do not leave it as is
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

        pointers.setX(allocator.getPointer(op.x(), context));
        pointers.setXShapeInfo(allocator.getPointer(op.x().shapeInfoDataBuffer(), context));
        pointers.setZ(allocator.getPointer(op.z(), context));
        pointers.setZShapeInfo(allocator.getPointer(op.z().shapeInfoDataBuffer(), context));
        pointers.setZLength(op.z().length());

        if (op.y() != null) {
            pointers.setY(allocator.getPointer(op.y(), context));
            pointers.setYShapeInfo(allocator.getPointer(op.y().shapeInfoDataBuffer(), context));
        }

        if (dimensions != null && dimensions.length > 0) {
            DataBuffer dimensionBuffer = Nd4j.getConstantHandler().getConstantBuffer(dimensions);
            pointers.setDimensions(allocator.getPointer(dimensionBuffer, context));
            pointers.setDimensionsLength(dimensions.length);
        }


        // we build TADs
        if (dimensions != null && dimensions.length > 0) {
            Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimensions);

            Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);
            Pointer devTadOffsets = tadBuffers.getSecond() == null ? null :AtomicAllocator.getInstance().getPointer(tadBuffers.getSecond(), context);

            // we don't really care, if tadOffsets will be nulls
            pointers.setTadShape(devTadShapeInfo);
            pointers.setTadOffsets(devTadOffsets);
        }


        return pointers;
    }

    /**
     * This method returns Op queue lengths for current device
     *
     * PLEASE NOTE: This value do NOT includes variative lastOp
     *
     * @return
     */
    protected int getQueueLength() {
        return deviceQueues.get().size();
    }

    /**
     * This method returns Op queue length for specified device
     *
     * @param deviceId
     * @return
     */
    @Deprecated
    protected int getQueueLength(int deviceId) {
        return -1;
    }

    /**
     * This method bundless all ops available in queue into single GridOp
     * @return
     */
    protected GridOp buildGrid() {
        return null;
    }

    protected void buildZ(IndexAccumulation op, int... dimension) {
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


        INDArray ret = null;
        if (Math.abs(op.zeroDouble()) < Nd4j.EPS_THRESHOLD) {
            ret= Nd4j.zeros(retShape);
        } else {
            ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
        }
        op.setZ(ret);
    }

    protected void buildZ(Accumulation op, int... dimension) {
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

/*
        if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return op.noOp();
*/

        INDArray ret = null;
        if (Math.abs(op.zeroDouble()) < Nd4j.EPS_THRESHOLD) {
            ret= Nd4j.zeros(retShape);
        } else {
            ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
        }
        op.setZ(ret);
    }

    @Override
    public Op exec(Op op, int... dimension) {
        // FIXME: make sure we're not going this route
        if (1>0) throw new UnsupportedOperationException("Bad execution route");
        return super.exec(op, dimension);
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        buildZ(op, dimension);

        // we should check, if this op returns scalar or not
        // if op.Z is scalar, we can't use GridOp here
        if (op.z().isScalar()) {
            // So, that's scalar. We'll have to flush queue
        } else {
            processAsGridOp(op, dimension);
        }

        return op.z();
    }


    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        buildZ(op, dimension);

        if (op.z().isScalar()) {
            // So, that's scalar. We'll have to flush queue
        } else {
            processAsGridOp(op, dimension);
        }

        return op.z();
    }

    @Override
    public INDArray exec(BroadcastOp op, int... dimension) {
        processAsGridOp(op, dimension);

        return op.z();
    }

    // FIXME: remove CudaContext return type. We just don't need it
    @Override
    protected CudaContext invoke(BroadcastOp op) {
        processAsGridOp(op, op.getDimension());

        return null;
    }

    // FIXME: remove CudaContext return type. We just don't need it
    @Override
    protected CudaContext invoke(ScalarOp op) {
        processAsGridOp(op, null);

        processAsGridOp(op, null);

        return null;
    }

    // FIXME: remove CudaContext return type. We just don't need it
    @Override
    protected CudaContext invoke(TransformOp op) {
        if (op.isExecSpecial()) {
            super.invoke(op);
        } else {
            processAsGridOp(op, null);
        }
        return null;
    }

    protected void prepareGrid(MetaOp op) {
        GridPointers ptrA = pointerizeOp(op.getFirstOp());
        GridPointers ptrB = pointerizeOp(op.getSecondOp());

        op.setFirstPointers(ptrA);
        op.setSecondPointers(ptrB);
    }

    @Override
    public void exec(MetaOp op) {
        prepareGrid(op);

        //CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z());

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        PointerPointer extras = extraz.get().put(1, context.getOldStream());

        double scalarA = 0.0;
        double scalarB = 0.0;

        if (op.getFirstOp() instanceof ScalarOp)
            scalarA = ((ScalarOp) op.getFirstOp()).scalar().doubleValue();

        if (op.getSecondOp() instanceof ScalarOp)
            scalarB = ((ScalarOp) op.getSecondOp()).scalar().doubleValue();

        GridPointers first = op.getGridDescriptor().getGridPointers().get(0);
        GridPointers second = op.getGridDescriptor().getGridPointers().get(1);

        /*
            TODO: launch can be either strided, or shapeInfo-based, it doesn't really matters for us.
            We just need to pass all pointers.

            TODO: obviously, execMetaPredicateElementwiseFloat should be renamed to execMetaPredicateStridedFloat
         */

        if (first.getDtype() == DataBuffer.Type.FLOAT) {

            nativeOps.execMetaPredicateStridedFloat(extras,
                    first.getType().ordinal(),
                    first.getOpNum(),
                    second.getType().ordinal(),
                    second.getOpNum(),
                    first.getXLength(),
                    first.getX(),
                    first.getXStride(),
                    second.getY(), // can be null
                    second.getYStride(), // cane be -1
                    second.getZ(),
                    second.getZStride(),
                    first.getExtraArgs(),
                    second.getExtraArgs(),
                    (float) scalarA,
                    (float) scalarB
            );
        } else if (first.getDtype() == DataBuffer.Type.DOUBLE) {
            nativeOps.execMetaPredicateStridedFloat(extras,
                    first.getType().ordinal(),
                    first.getOpNum(),
                    second.getType().ordinal(),
                    second.getOpNum(),
                    first.getXLength(),
                    first.getX(),
                    first.getXStride(),
                    second.getY(), // can be null
                    second.getYStride(), // cane be -1
                    second.getZ(),
                    second.getZStride(),
                    first.getExtraArgs(),
                    second.getExtraArgs(),
                    (float) scalarA,
                    (float) scalarB
            );
        } else if (first.getDtype() == DataBuffer.Type.HALF) {
            nativeOps.execMetaPredicateStridedFloat(extras,
                    first.getType().ordinal(),
                    first.getOpNum(),
                    second.getType().ordinal(),
                    second.getOpNum(),
                    first.getXLength(),
                    first.getX(),
                    first.getXStride(),
                    second.getY(), // can be null
                    second.getYStride(), // cane be -1
                    second.getZ(),
                    second.getZStride(),
                    first.getExtraArgs(),
                    second.getExtraArgs(),
                    (float) scalarA,
                    (float) scalarB
            );
        }

/*

         //
         //   Initial draft for MetaOps
         //

        if (op.getGridDescriptor().getGridPointers().get(0).getType() == Op.Type.SCALAR) {

            nativeOps.execMetaStridedFloat(extras,
                    op.getGridDescriptor().getGridPointers().get(0).getType().ordinal(),
                    op.getGridDescriptor().getGridPointers().get(0).getOpNum(),
                    op.getGridDescriptor().getGridPointers().get(1).getType().ordinal(),
                    op.getGridDescriptor().getGridPointers().get(1).getOpNum(),
                    op.getGridDescriptor().getGridPointers().get(0).getXLength(),
                    ((ScalarOp) op.getFirstOp()).scalar().floatValue(),
                    op.getGridDescriptor().getGridPointers().get(0).getX(),
                    op.getGridDescriptor().getGridPointers().get(0).getXStride(),
                    op.getGridDescriptor().getGridPointers().get(1).getExtraArgs(),
                    op.getGridDescriptor().getGridPointers().get(1).getZ(),
                    op.getGridDescriptor().getGridPointers().get(1).getZStride()
            );

        } else if (op.getGridDescriptor().getGridPointers().get(1).getType() == Op.Type.SCALAR) {
            nativeOps.execMetaStridedFloat(extras,
                    op.getGridDescriptor().getGridPointers().get(0).getType().ordinal(),
                    op.getGridDescriptor().getGridPointers().get(0).getOpNum(),
                    op.getGridDescriptor().getGridPointers().get(1).getType().ordinal(),
                    op.getGridDescriptor().getGridPointers().get(1).getOpNum(),
                    op.getGridDescriptor().getGridPointers().get(0).getXLength(),
                    ((ScalarOp) op.getSecondOp()).scalar().floatValue(),
                    op.getGridDescriptor().getGridPointers().get(0).getX(),
                    op.getGridDescriptor().getGridPointers().get(0).getXStride(),
                    op.getGridDescriptor().getGridPointers().get(0).getExtraArgs(),
                    op.getGridDescriptor().getGridPointers().get(1).getZ(),
                    op.getGridDescriptor().getGridPointers().get(1).getZStride()
            );
        }
*/
    }

    @Override
    public void exec(GridOp op) {
        // TODO: to be implemented
    }

    /**
     * This method forces all currently enqueued ops to be executed immediately
     *
     * PLEASE NOTE: This call IS non-blocking
     */
    public void flushQueue() {
        /*
            Basically we just want to form GridOp and pass it to native executioner
            But since we don't have GridOp interface yet, we'll send everything to underlying CudaExecutioner.
         */
        // TODO: proper implementation for GridOp creation required here
        Deque<OpDescriptor> currentQueue = deviceQueues.get();

        OpDescriptor op = currentQueue.pollFirst();
        while (op != null) {
            if (op.getDimensions() == null) {
                super.exec(op.getOp());
            } else {
                if (op.getOp() instanceof IndexAccumulation) {
                    super.exec((IndexAccumulation) op.getOp(), op.getDimensions());
                } else if (op.getOp() instanceof Accumulation) {
                    super.exec((Accumulation) op.getOp(), op.getDimensions());
                } else if (op.getOp() instanceof BroadcastOp) {
                    super.exec((BroadcastOp) op.getOp(), op.getDimensions());
                }
            }

            op = currentQueue.pollFirst();

            // if no more ops in queue - we force lastOp to be executed here as well
            if (op == null) {
                op = lastOp.get();
                if (op != null)
                    lastOp.remove();
            }
        }

    }

    /**
     * This method forces all currently enqueued ops to be executed immediately
     *
     * PLEASE NOTE: This call is always blocking, until all queued operations are finished
     */
    @Override
    public void flushQueueBlocking() {
        flushQueue();

        ((CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext()).syncOldStream();
    }
}
