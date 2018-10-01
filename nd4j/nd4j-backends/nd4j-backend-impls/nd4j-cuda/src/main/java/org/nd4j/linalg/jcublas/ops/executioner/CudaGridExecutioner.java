/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.jcublas.ops.executioner;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.val;
import org.nd4j.linalg.primitives.Pair;
import org.bytedeco.javacpp.*;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.impl.meta.InvertedPredicateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.PostulateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.PredicateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.ReduceMetaOp;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMax;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMin;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.ops.executioner.aggregates.AggregateDescriptor;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.LongPointerWrapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

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
        POSTULATE
    }

    // general queues
    //private List<Deque<OpDescriptor>> deviceQueues = new ArrayList<>();

    // last op
    private ThreadLocal<OpDescriptor> lastOp = new ThreadLocal<>();
    //    private ThreadLocal<PointerPointer> extraz = new ThreadLocal<>();
    private ThreadLocal<Deque<OpDescriptor>> deviceQueues = new ThreadLocal<>();

    private ThreadLocal<AtomicLong> opCounter = new ThreadLocal<>();

    private AtomicLong metaCounter = new AtomicLong(0);
    private AtomicLong execCounter = new AtomicLong(0);

    private List<WatchdogPair> watchdog = new CopyOnWriteArrayList<>();

    private List<Queue<AggregateDescriptor>> aggregates = new ArrayList<>();

    private static Logger logger = LoggerFactory.getLogger(CudaGridExecutioner.class);

    private AtomicBoolean experimental = new AtomicBoolean(false);

    public CudaGridExecutioner() {
        //        extraz.set(new PointerPointer(10));
        deviceQueues.set(new ArrayDeque<OpDescriptor>());

        int numDevices = nativeOps.getAvailableDevices();

        for (int x = 0; x < numDevices; x++) {
            aggregates.add(new ConcurrentLinkedQueue<AggregateDescriptor>());
        }

        experimental.set(nativeOps.isExperimentalEnabled());
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
        checkForCompression(op);

        invokeWatchdog(op);

        if (op instanceof GradientOp) {
            commit();
            op.exec();
        } else if (op instanceof ReduceOp) {
            exec((ReduceOp) op, new int[] {Integer.MAX_VALUE});
        } else if (op instanceof IndexAccumulation) {
            exec((IndexAccumulation) op, new int[] {Integer.MAX_VALUE});
        } else if (op instanceof ScalarOp || op instanceof TransformOp) {
            // the only entry place for TADless ops
            processAsGridOp(op);
        } else if (op instanceof BroadcastOp) {
            invoke((BroadcastOp) op);
        } else {
            //logger.info("Random op: {}", op.getClass().getSimpleName());
            pushToGrid(new OpDescriptor(op));
        }

        return op;
    }


    protected void pushToGrid(OpDescriptor descriptor) {
        pushToGrid(descriptor, true);
    }


    protected void invokeWatchdog(Op op) {

        if (watchdog.size() > 0)
            for (WatchdogPair pair : watchdog) {
                if (compareArrays(pair.getArray(), op)) {
                    //    logger.info("WATCHDOG: Invoked {} op on {} using JVM eq", op.getClass().getSimpleName(), pair.getTag());
                    continue;
                }

                if (compareDevicePointers(pair.getArray(), op)) {
                    //  logger.info("WATCHDOG: Invoked {} op on {} using device PTR; Thread ID: {}; deviceId: {}", op.getClass().getSimpleName(), pair.getTag(), Thread.currentThread().getId(), Nd4j.getAffinityManager().getDeviceForCurrentThread());
                    throw new RuntimeException();
                }

                if (compareHostPointers(pair.getArray(), op)) {
                    //    logger.info("WATCHDOG: Invoked {} op on {} using host PTR", op.getClass().getSimpleName(), pair.getTag());
                    continue;
                }
            }
    }

    protected boolean compareDevicePointers(INDArray array, Op op) {
        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        Pointer pointer = AtomicAllocator.getInstance().getPointer(array, context);

        long opZ = AtomicAllocator.getInstance().getPointer(op.z(), context).address();
        long opX = AtomicAllocator.getInstance().getPointer(op.x(), context).address();

        long opY = op.y() == null ? 0 : AtomicAllocator.getInstance().getPointer(op.y(), context).address();

        if (opZ == pointer.address()) {
            //logger.error("op.Z matched: {}", pointer.address());
            return true;
        }

        if (opY == pointer.address()) {
            //logger.error("op.Y matched: {}", pointer.address());
            return true;
        }

        if (opX == pointer.address()) {
            //logger.error("op.X matched: {}", pointer.address());
            return true;
        }

        return false;
    }


    protected boolean compareHostPointers(INDArray array, Op op) {
        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        Pointer pointer = AtomicAllocator.getInstance().getPointer(array, context);

        long opZ = AtomicAllocator.getInstance().getHostPointer(op.z()).address();
        long opX = AtomicAllocator.getInstance().getHostPointer(op.x()).address();

        long opY = op.y() == null ? 0 : AtomicAllocator.getInstance().getHostPointer(op.y()).address();

        if (opZ == pointer.address() || opY == pointer.address() || opX == pointer.address())
            return true;

        return false;
    }

    protected boolean compareArrays(INDArray array, Op op) {
        if (op.x() == array || op.y() == array || op.z() == array)
            return true;

        return false;
    }

    /**
     * This method adds op into GridOp queue
     *
     * @return
     */
    protected void pushToGrid(OpDescriptor descriptor, boolean flush) {

        // we should just add op to queue here
        //deviceQueues.get().add(descriptor);

        // FIXME: following code should be removed, since it's just executing supers instead of batching

        execCounter.incrementAndGet();

        Op op = descriptor.getOp();
        int[] dimensions = descriptor.getDimensions();

        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            if (flush)
                flushQueue();

            //logger.info("Sending TransformOp to CudaExecutioner");
            super.invoke(t);
        } else if (op instanceof Variance) {
            Variance acc = (Variance) op;
            if (flush)
                flushQueue();

            super.naiveExec(acc, dimensions);
        } else if (op instanceof ReduceOp) {
            ReduceOp acc = (ReduceOp) op;
            if (flush)
                flushQueue();

            //logger.info("Sending AccumulationOp to CudaExecutioner: {}", Arrays.toString(dimensions));
            super.naiveExec(acc, dimensions);
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;
            if (flush)
                flushQueue();

            //logger.info("Sending ScalarOp to CudaExecutioner");
            super.invoke(sc);
        } else if (op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;
            if (flush)
                flushQueue();

            //logger.info("Sending BroadcastOp to CudaExecutioner");
            if (dimensions != null) {
                super.exec(broadcastOp, dimensions);
            } else {
                super.invoke(broadcastOp);
            }
        } else if (op instanceof IndexAccumulation) {
            IndexAccumulation indexAccumulation = (IndexAccumulation) op;
            if (flush)
                flushQueue();

            //logger.info("Sending IndexAccumulationOp to CudaExecutioner");
            super.exec(indexAccumulation, dimensions);
        } else if (op instanceof MetaOp) {
            //     logger.info("Executing MetaOp");
            metaCounter.incrementAndGet();
            exec((MetaOp) op);
        } else if (op instanceof GridOp) {
            //    logger.info("Executing GridOp");
            exec((GridOp) op);
        }
    }



    public long getMetaCounter() {
        return metaCounter.get();
    }

    public long getExecutionCounter() {
        return execCounter.get();
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
        // CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());


        OpDescriptor last = lastOp.get();
        if (last != null) {
            MetaType type = getMetaOpType(op, dimension);
            lastOp.remove();
            switch (type) {
                case NOT_APPLICABLE: {
                    /*
                        If we can't form MetaOp with new Op here, we should move lastOp to GridOp queue, and update lastOp with current Op
                    */
                    dequeueOp(last);
                    pushToGrid(last, false);

                    //|| op instanceof ScalarOp
                    if ((op instanceof TransformOp && op.y() != null) && onCurrentDeviceXYZ(op)) {
                        enqueueOp(new OpDescriptor(op, dimension));
                    } else {
                        pushToGrid(new OpDescriptor(op, dimension), false);
                    }
                }
                break;
                case PREDICATE: {
                    MetaOp metaOp = new PredicateMetaOp(last, new OpDescriptor(op, dimension));
                    pushToGrid(new OpDescriptor(metaOp), false);
                }
                break;
                case INVERTED_PREDICATE: {
                    OpDescriptor currentOp = new OpDescriptor(op, dimension);

                    //          logger.info("Calling for Meta: {}+{}", last.getOp().getClass().getSimpleName(), currentOp.getOp().getClass().getSimpleName());
                    dequeueOp(last);
                    dequeueOp(currentOp);

                    MetaOp metaOp = new InvertedPredicateMetaOp(last, currentOp);
                    pushToGrid(new OpDescriptor(metaOp), false);
                }
                break;
                case POSTULATE: {
                    MetaOp metaOp = new PostulateMetaOp(last, new OpDescriptor(op, dimension));
                    pushToGrid(new OpDescriptor(metaOp), false);
                }
                break;
                default:
                    throw new UnsupportedOperationException("Not supported MetaType: [" + type + "]");
            }
        } else {
            //&& Nd4j.dataType() != DataBuffer.Type.HALF
            if ((op instanceof TransformOp && op.y() != null && onCurrentDeviceXYZ(op))) {
                enqueueOp(new OpDescriptor(op, dimension));
            } else {
                pushToGrid(new OpDescriptor(op, dimension), false);
            }
        }

        //   AtomicAllocator.getInstance().getFlowController().registerAction(context, op.z(), op.x(), op.y());

        //return op;
    }

    protected boolean onCurrentDeviceXYZ(Op op) {
        int deviceId = AtomicAllocator.getInstance().getDeviceId();
        int deviceX = AtomicAllocator.getInstance().getDeviceId(op.x());
        int deviceY = AtomicAllocator.getInstance().getDeviceId(op.y());
        int deviceZ = AtomicAllocator.getInstance().getDeviceId(op.y());

        return deviceId == deviceX && deviceY == deviceZ && deviceZ == deviceX;
    }

    protected void enqueueOp(OpDescriptor descriptor) {
        AtomicAllocator.getInstance().getAllocationPoint(descriptor.getOp().x()).markEnqueued(true);
        AtomicAllocator.getInstance().getAllocationPoint(descriptor.getOp().z()).markEnqueued(true);

        if (descriptor.getOp().y() != null)
            AtomicAllocator.getInstance().getAllocationPoint(descriptor.getOp().y()).markEnqueued(true);

        //   logger.info("Enqueued op: " + descriptor.getOp().getClass().getSimpleName());

        lastOp.set(descriptor);
    }

    protected void dequeueOp(OpDescriptor descriptor) {

        AtomicAllocator.getInstance().getAllocationPoint(descriptor.getOp().x()).markEnqueued(false);
        AtomicAllocator.getInstance().getAllocationPoint(descriptor.getOp().z()).markEnqueued(false);

        if (descriptor.getOp().y() != null)
            AtomicAllocator.getInstance().getAllocationPoint(descriptor.getOp().y()).markEnqueued(false);

        //   logger.info("Dequeued op: " + descriptor.getOp().getClass().getSimpleName());
    }

    protected MetaType getMetaOpType(Op op, int... dimension) {

        //if (1 > 0) return MetaType.NOT_APPLICABLE;

        OpDescriptor last = lastOp.get();

        if (last == null) {
            return MetaType.NOT_APPLICABLE;
        } else {
            // Experimental native compilation required for full MIMD support
            if (experimental.get()) {
                logger.info("Experimental hook");
                if (last.getOp() instanceof ScalarOp || last.getOp() instanceof TransformOp) {
                    /*
                    Predicate logic is simple:
                        1) LastOp is one of following op types: Scalar, Transform, PairwiseTransform
                        2) LastOp isn't specialOp
                        3) LastOp op.x() == op.z()
                        4) currentOp op.x() == op.z(), and matches lastOp op.z()
                    */

                    return isMatchingZX(last.getOp(), op) ? MetaType.PREDICATE : MetaType.NOT_APPLICABLE;
                } else if (last.getOp() instanceof ReduceOp) {
                    /*
                    InvertedMetaOp, aka Postulate logic
                    
                    Postulate logic is simple too:
                        1) LastOp is opType of Reduce or Reduce3
                        2) LastOp op.z() isn't scalar
                        3) currentOp is one of the following op types: Scalar, Transform
                     */
                    if ((op instanceof ScalarOp || op instanceof TransformOp) && op.y() == null)
                        return isMatchingZX(last.getOp(), op) ? MetaType.POSTULATE : MetaType.NOT_APPLICABLE;
                }
            } else {
                // TODO: extend non-experimental support for MetaOps
                // we enable this only for PairwisetTransforms.Set followed by scalar
                if (last.getOp() instanceof TransformOp && last.getOp().y() != null) {
                    // FIXME: get rid of those instanceof
                    if (op instanceof ScalarOp && ((ScalarOp) op).getDimension() == null && !(op instanceof ScalarMax)
                            && !(op instanceof ScalarMin) && !(op.opNum() >= 7 && op.opNum() <= 11)
                            && op.opNum() != 16 && op.opNum() != 13
                            && !(op.opNum() >= 56 && op.opNum() <= 59)) {
                        return isMatchingZX(last.getOp(), op) ? MetaType.INVERTED_PREDICATE : MetaType.NOT_APPLICABLE;
                    }
                }
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
        if (opA.x() == opB.x() && opA.z() == opB.z() && opA.x() == opB.z())
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

    protected GridPointers pointerizeOp(OpDescriptor descriptor) {
        return pointerizeOp(descriptor.getOp(), descriptor.getDimensions());
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

        //        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(op.z(), op.x(), op.y());
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
            Pointer devTadOffsets = tadBuffers.getSecond() == null ? null
                    : AtomicAllocator.getInstance().getPointer(tadBuffers.getSecond(), context);

            // we don't really care, if tadOffsets will be nulls
            pointers.setTadShape(devTadShapeInfo);
            pointers.setTadOffsets(devTadOffsets);
        }


        return pointers;
    }

    /**
     * This method returns Op queue lengths for current device
     *
     * PLEASE NOTE: This value also includes variative lastOp
     *
     * @return
     */
    public int getQueueLength() {
        //return deviceQueues.get().size() + (lastOp.get() == null ? 0 : 1);
        return (lastOp.get() == null ? 0 : 1);
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

        for (int i = 0; i < dimension.length; i++) {
            if (dimension[i] < 0)
                dimension[i] += op.x().rank();
        }

        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        long[] retShape = Shape.wholeArrayDimension(dimension) ? new long[] {1, 1}
                : ArrayUtil.removeIndex(op.x().shape(), dimension);
        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new long[] {1, retShape[0]};
            else
                retShape = new long[] {retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new long[] {1, 1};
        }

        if(op.z() == null || op.z() == op.x()){
            INDArray ret = null;
            if (Math.abs(op.zeroDouble()) < Nd4j.EPS_THRESHOLD) {
                ret = Nd4j.zeros(retShape);
            } else {
                ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
            }

            op.setZ(ret);
        } else if(!Arrays.equals(retShape, op.z().shape())){
            throw new IllegalStateException("Z array shape does not match expected return type for op " + op
                    + ": expected shape " + Arrays.toString(retShape) + ", z.shape()=" + Arrays.toString(op.z().shape()));
        }
    }

    protected void buildZ(ReduceOp op, int... dimension) {
        Arrays.sort(dimension);

        for (int i = 0; i < dimension.length; i++) {
            if (dimension[i] < 0)
                dimension[i] += op.x().rank();
        }

        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        long[] retShape = Shape.wholeArrayDimension(dimension) ? new long[] {1, 1}
                : ArrayUtil.removeIndex(op.x().shape(), dimension);
        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new long[] {1, retShape[0]};
            else
                retShape = new long[] {retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new long[] {1, 1};
        }

        /*
        if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return op.noOp();
        */

        INDArray ret = null;
        if (op.z() == null || op.z() == op.x()) {
            if (op.isComplexAccumulation()) {
                val xT = op.x().tensorssAlongDimension(dimension);
                val yT = op.y().tensorssAlongDimension(dimension);

                ret = Nd4j.create(xT, yT);
            } else {
                if (Math.abs(op.zeroDouble()) < Nd4j.EPS_THRESHOLD) {
                    ret = Nd4j.zeros(retShape);
                } else {
                    ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
                }
            }

            op.setZ(ret);
        } else {
            // compare length
            if (op.z().lengthLong() != ArrayUtil.prodLong(retShape))
                throw new ND4JIllegalStateException("Shape of target array for reduction [" + Arrays.toString(op.z().shape()) + "] doesn't match expected [" + Arrays.toString(retShape) + "]");

            if (op.x().data().dataType() == DataBuffer.Type.DOUBLE) {
                op.z().assign(op.zeroDouble());
            } else if (op.x().data().dataType() == DataBuffer.Type.FLOAT) {
                op.z().assign(op.zeroFloat());
            } else if (op.x().data().dataType() == DataBuffer.Type.HALF) {
                op.z().assign(op.zeroHalf());
            }

            ret = op.z();
        }
    }

    @Override
    public Op exec(Op op, int... dimension) {
        // FIXME: make sure we're not going this route
        // if (1>0) throw new UnsupportedOperationException("Bad execution route");
        flushQueue();

        return super.exec(op, dimension);
    }

    @Override
    public INDArray exec(ReduceOp op, int... dimension) {


        // we should check, if this op returns scalar or not
        // if op.Z is scalar, we can't use GridOp here
        if (dimension == null || dimension.length == 0 || dimension[0] == Integer.MAX_VALUE) {
            // So, that's scalar. We'll have to flush queue
            // processAsGridOp(op, dimension);
            flushQueue();

            super.exec(op, new int[] {Integer.MAX_VALUE});
        } else {
            buildZ(op, dimension);
            processAsGridOp(op, dimension);
        }

        return op.z();
    }


    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        //        buildZ(op, dimension);

        if (dimension == null || dimension.length == 0 || dimension[0] == Integer.MAX_VALUE) {
            // So, that's scalar. We'll have to flush queue
            flushQueue();

            buildZ(op, new int[] {Integer.MAX_VALUE});
            super.invoke(op, new int[] {Integer.MAX_VALUE});
        } else {
            buildZ(op, dimension);
            processAsGridOp(op, dimension);
        }

        return op.z();
    }

    @Override
    public INDArray exec(BroadcastOp op, int... dimension) {
        processAsGridOp(op, dimension);

        return op.z();
    }

    // FIXME: remove CudaContext return opType. We just don't need it
    @Override
    protected CudaContext invoke(BroadcastOp op) {
        processAsGridOp(op, op.getDimension());

        return null;
    }

    // FIXME: remove CudaContext return opType. We just don't need it
    @Override
    protected CudaContext invoke(ScalarOp op) {
        processAsGridOp(op, null);

        return null;
    }

    // FIXME: remove CudaContext return opType. We just don't need it
    @Override
    protected CudaContext invoke(TransformOp op) {
        if (op.isExecSpecial()) {
            flushQueue();
            super.invoke(op);
        } else {
            processAsGridOp(op, null);
        }
        return null;
    }

    protected void prepareGrid(MetaOp op) {
        GridPointers ptrA = pointerizeOp(op.getFirstOpDescriptor());
        GridPointers ptrB = pointerizeOp(op.getSecondOpDescriptor());



        op.setFirstPointers(ptrA);
        op.setSecondPointers(ptrB);
    }

    @Override
    public void exec(MetaOp op) {
        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        prepareGrid(op);

        GridPointers first = op.getGridDescriptor().getGridPointers().get(0);
        GridPointers second = op.getGridDescriptor().getGridPointers().get(1);

        // we need to use it only for first op, since for MetaOps second op shares the same X & Z by definition
        CudaContext context =
                AtomicAllocator.getInstance().getFlowController().prepareAction(first.getOpZ(), first.getOpY());

        //        AtomicAllocator.getInstance().getFlowController().prepareAction(second.getOpX(), second.getOpY(), second.getOpZ());


        //CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();


        PointerPointer extras = extraz.get().put(null, context.getOldStream());

        double scalarA = 0.0;
        double scalarB = 0.0;

        if (op.getFirstOp() instanceof ScalarOp)
            scalarA = ((ScalarOp) op.getFirstOp()).scalar().doubleValue();

        if (op.getSecondOp() instanceof ScalarOp)
            scalarB = ((ScalarOp) op.getSecondOp()).scalar().doubleValue();



        //logger.info("FirstOp: {}, SecondOp: {}", op.getFirstOp().getClass().getSimpleName(), op.getSecondOp().getClass().getSimpleName());

        /*
            TODO: launch can be either strided, or shapeInfo-based, it doesn't really matters for us.
            We just need to pass all pointers.
        
            TODO: obviously, execMetaPredicateElementwiseFloat should be renamed to execMetaPredicateStridedFloat
         */

        // FIXME: this is bad hack, reconsider this one
        GridPointers yGrid = first;

        if (op.getSecondOp().y() != null) {
            yGrid = second;
        }


        if (op instanceof PredicateMetaOp || op instanceof InvertedPredicateMetaOp) {
            if (first.getDtype() == DataBuffer.Type.FLOAT) {
                if (yGrid.getYOrder() == yGrid.getXOrder() && yGrid.getXStride() >= 1 && yGrid.getYStride() >= 1) {
                    nativeOps.execMetaPredicateStridedFloat(extras, first.getType().ordinal(), first.getOpNum(),
                            second.getType().ordinal(), second.getOpNum(), first.getXLength(),
                            (FloatPointer) first.getX(), first.getXStride(), (FloatPointer) yGrid.getY(), // can be null
                            yGrid.getYStride(), // cane be -1
                            (FloatPointer) second.getZ(), second.getZStride(),
                            (FloatPointer) first.getExtraArgs(), (FloatPointer) second.getExtraArgs(),
                            (float) scalarA, (float) scalarB);
                } else {
                    nativeOps.execMetaPredicateShapeFloat(extras, first.getType().ordinal(), first.getOpNum(),
                            second.getType().ordinal(), second.getOpNum(), first.getXLength(),
                            (FloatPointer) first.getX(), (LongPointer) first.getXShapeInfo(),
                            (FloatPointer) yGrid.getY(), // can be null
                            (LongPointer) yGrid.getYShapeInfo(), // cane be -1
                            (FloatPointer) second.getZ(), (LongPointer) second.getZShapeInfo(),
                            (FloatPointer) first.getExtraArgs(), (FloatPointer) second.getExtraArgs(),
                            (float) scalarA, (float) scalarB);
                }
            } else if (first.getDtype() == DataBuffer.Type.DOUBLE) {
                if (yGrid.getYOrder() == yGrid.getXOrder() && yGrid.getXStride() >= 1 && yGrid.getYStride() >= 1) {
                    nativeOps.execMetaPredicateStridedDouble(extras, first.getType().ordinal(), first.getOpNum(),
                            second.getType().ordinal(), second.getOpNum(), first.getXLength(),
                            (DoublePointer) first.getX(), first.getXStride(), (DoublePointer) yGrid.getY(), // can be null
                            yGrid.getYStride(), // cane be -1
                            (DoublePointer) second.getZ(), second.getZStride(),
                            (DoublePointer) first.getExtraArgs(), (DoublePointer) second.getExtraArgs(),
                            scalarA, scalarB);
                } else {
                    nativeOps.execMetaPredicateShapeDouble(extras, first.getType().ordinal(), first.getOpNum(),
                            second.getType().ordinal(), second.getOpNum(), first.getXLength(),
                            (DoublePointer) first.getX(), (LongPointer) first.getXShapeInfo(),
                            (DoublePointer) yGrid.getY(), // can be null
                            (LongPointer) yGrid.getYShapeInfo(), // cane be -1
                            (DoublePointer) second.getZ(), (LongPointer) second.getZShapeInfo(),
                            (DoublePointer) first.getExtraArgs(), (DoublePointer) second.getExtraArgs(),
                            scalarA, scalarB);
                }
            } else {
                if (yGrid.getYOrder() == yGrid.getXOrder() && yGrid.getXStride() >= 1 && yGrid.getYStride() >= 1) {
                    nativeOps.execMetaPredicateStridedHalf(extras, first.getType().ordinal(), first.getOpNum(),
                            second.getType().ordinal(), second.getOpNum(), first.getXLength(),
                            (ShortPointer) first.getX(), first.getXStride(), (ShortPointer) yGrid.getY(), // can be null
                            yGrid.getYStride(), // cane be -1
                            (ShortPointer) second.getZ(), second.getZStride(),
                            (ShortPointer) first.getExtraArgs(), (ShortPointer) second.getExtraArgs(),
                            (float) scalarA, (float) scalarB);
                } else {
                    nativeOps.execMetaPredicateShapeHalf(extras, first.getType().ordinal(), first.getOpNum(),
                            second.getType().ordinal(), second.getOpNum(), first.getXLength(),
                            (ShortPointer) first.getX(), (LongPointer) first.getXShapeInfo(),
                            (ShortPointer) yGrid.getY(), // can be null
                            (LongPointer) yGrid.getYShapeInfo(), // cane be -1
                            (ShortPointer) second.getZ(), (LongPointer) second.getZShapeInfo(),
                            (ShortPointer) first.getExtraArgs(), (ShortPointer) second.getExtraArgs(),
                            (float) scalarA, (float) scalarB);
                }
            }
        } else if (op instanceof ReduceMetaOp) {
            if (first.getDtype() == DataBuffer.Type.FLOAT) {

                nativeOps.execMetaPredicateReduceFloat(extras, first.getType().ordinal(), first.getOpNum(),
                        second.getType().ordinal(), second.getOpNum(), (FloatPointer) first.getX(),
                        (LongPointer) first.getXShapeInfo(), (FloatPointer) second.getY(),
                        (LongPointer) second.getYShapeInfo(), (FloatPointer) second.getZ(),
                        (LongPointer) second.getZShapeInfo(), (IntPointer) second.getDimensions(),
                        second.getDimensionsLength(), (LongPointer) second.getTadShape(),
                        new LongPointerWrapper(second.getTadOffsets()), (FloatPointer) first.getExtraArgs(),
                        (FloatPointer) second.getExtraArgs(), (float) scalarA, 0.0f, false);
            }
        }

        AtomicAllocator.getInstance().getFlowController().registerAction(context, first.getOpZ(), first.getOpY());
        //        AtomicAllocator.getInstance().getFlowController().registerAction(context, second.getOpX(), second.getOpY(), second.getOpZ());
    }

    @Override
    public void exec(GridOp op) {
        // TODO: to be implemented
    }

    protected void purgeQueue() {
        lastOp.remove();
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
        //    logger.info("Non-Blocking flush");
        // TODO: proper implementation for GridOp creation required here
        /*
        Deque<OpDescriptor> currentQueue = deviceQueues.get();
        if (currentQueue == null)
            return;
        
        OpDescriptor op = currentQueue.pollFirst();
        while (op != null) {
            pushToGrid(op, false);
        
            op = currentQueue.pollFirst();
        }
        */

        // we need to check,
        OpDescriptor op = lastOp.get();
        if (op != null) {
            if (!experimental.get()) {
                //if (!nativeOps.isExperimentalEnabled()) {
                // it might be only pairwise transform here for now
                //          logger.info("Flushing existing lastOp");
                lastOp.remove();
                dequeueOp(op);
                pushToGrid(op, false);
            } else {
                throw new UnsupportedOperationException("Experimental flush isn't supported yet");
            }
        } else {
            //      logger.info("Queue is empty");

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

        val context =((CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext());

        context.syncSpecialStream();
        context.syncOldStream();
    }

    public void addToWatchdog(INDArray array, String tag) {
        watchdog.add(new WatchdogPair(array, tag));
    }


    /**
     * This method executes specified RandomOp using default RNG available via Nd4j.getRandom()
     *
     * @param op
     */
    @Override
    public INDArray exec(RandomOp op) {
        return exec(op, Nd4j.getRandom());
    }


    @Override
    public void exec(List<Aggregate> batch) {
        flushQueue();

        super.exec(batch);
    }

    @Override
    public void exec(Aggregate op) {
        flushQueue();

        super.exec(op);
    }

    /**
     * This method enqueues aggregate op for future invocation with respect to thread and op order
     * This method uses current thread Id as aggregation key.
     *
     * @param op
     */
    @Override
    public void aggregate(Aggregate op) {
        aggregate(op, Thread.currentThread().getId());
    }

    /**
     * This method enqueues aggregate op for future invocation.
     * Key value will be used to batch individual ops
     *
     * @param op
     * @param key
     */
    @Override
    public void aggregate(Aggregate op, long key) {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        if (opCounter.get() == null)
            opCounter.set(new AtomicLong(0));

        // we enqueue op for specific device here
        aggregates.get(deviceId).add(new AggregateDescriptor(op, key, opCounter.get().getAndIncrement()));
    }

    @Override
    public INDArray exec(RandomOp op, Random rng) {
        flushQueue();

        return super.exec(op, rng);
    }

    protected void buildAggregation() {

    }

    /*
    @Override
    public INDArray execAndReturn(BroadcastOp op) {
        flushQueue();
        execCounter.incrementAndGet();
    
        return super.execAndReturn(op);
    }
    
    @Override
    public INDArray execAndReturn(Op op) {
        flushQueue();
        execCounter.incrementAndGet();
    
        return super.execAndReturn(op);
    }
    
    @Override
    public INDArray execAndReturn(ScalarOp op) {
        flushQueue();
        execCounter.incrementAndGet();
    
        super.invoke(op);
        return op.z();
    }
    
    @Override
    public INDArray execAndReturn(TransformOp op) {
        flushQueue();
        execCounter.incrementAndGet();
    
        super.invoke(op);
    
        return op.z();
    }
    */



    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    private static class WatchdogPair {
        private INDArray array;
        private String tag;
    }


    @Override
    public void push() {
        flushQueue();
    }

    @Override
    public void commit() {
        flushQueueBlocking();
    }
}
