package org.nd4j.jita.flow.impl;

import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.context.ContextPack;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.jita.allocator.time.TimeProvider;
import org.nd4j.jita.allocator.time.providers.OperativeProvider;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class AsynchronousFlowController implements FlowController{
    private volatile Allocator allocator;

    private static final Configuration configuration = CudaEnvironment.getInstance().getConfiguration();

    private static Logger log = LoggerFactory.getLogger(AsynchronousFlowController.class);

    protected NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    private transient TimeProvider timeProvider = new OperativeProvider();

    protected AtomicLong asyncHit = new AtomicLong(0);
    protected AtomicLong asyncMiss = new AtomicLong(0);

    protected Map<Integer, AtomicLong> lanesCounter = new ConcurrentHashMap<>();

    private AtomicLong totalHits = new AtomicLong(0);

    protected static final int MAX_EXECUTION_QUEUE =  configuration.getCommandQueueLength();

    protected static final AtomicLong eventCounts = new AtomicLong(0);

    @Override
    public void init(Allocator allocator) {
        this.allocator = allocator;
    }

    @Override
    public void synchronizeToHost(AllocationPoint point) {
        if (!point.isActualOnHostSide()) {

            if (!point.isConstant())
                waitTillFinished(point);

            //  log.info("Synchronization started... " + point.getShape());

            // if this piece of memory is device-dependant, we'll also issue copyback once
            if (point.getAllocationStatus() == AllocationStatus.DEVICE && !point.isActualOnHostSide()) {
                CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

                if (nativeOps.memcpyAsync(
                        point.getHostPointer().address(),
                        point.getDevicePointer().address(),
                        AllocationUtils.getRequiredMemory(point.getShape()),
                        CudaConstants.cudaMemcpyDeviceToHost,
                        context.getSpecialStream().address()) == 0)
                    throw new IllegalStateException("MemcpyAsync failed");

                context.syncSpecialStream();
            }// else log.info("Not [DEVICE] memory, skipping...");

            // updating host read timer
            point.tickHostRead();
            //log.info("After sync... isActualOnHostSide: {}", point.isActualOnHostSide());
        }
    }

    @Override
    public void waitTillFinished(AllocationPoint point) {
        cudaEvent_t event = point.getWriteLane();
        if (event != null) {
            event.synchronize();
            event.destroy();
        }
    }

    public void waitTillReleased(AllocationPoint point) {
        waitTillFinished(point);

        cudaEvent_t event;
        while ((event = point.getReadLane().poll()) != null) {
                event.synchronize();
                event.destroy();
        }
    }

    @Override
    public void registerAction(CudaContext context, INDArray result, INDArray... operands) {
        // TODO: this should be lane-dependant context

        if (totalHits.incrementAndGet() % 25000 == 0) {
            log.debug("AsyncHit ratio: [{}]", getAsyncHitRatio());
/*
            for (int lane = 0; lane < allocator.getContextPool().acquireContextPackForDevice(0).getAvailableLanes(); lane++) {
                log.debug("Lane [{}]: {} ", lane, lanesCounter.get(lane).get());
            }
            */
        }

        cudaEvent_t event = new cudaEvent_t(nativeOps.createEvent());
        event.setLaneId(context.getLaneId());
        nativeOps.registerEvent(event.address(), context.getOldStream().address());

        if (result != null) {
            setWriteLane(result, event);
            allocator.tickDeviceWrite(result);
        }

        for (INDArray operand: operands) {
            if (operand == null) continue;

            setReadLane(operand, event);
        }
    }

    protected void setWriteLane(INDArray array, cudaEvent_t event) {
        AllocationPoint point = allocator.getAllocationPoint(array);

        point.setWriteLane(event);
    }

    protected void setReadLane(INDArray array, cudaEvent_t event) {
        AllocationPoint point = allocator.getAllocationPoint(array);

        point.addReadLane(event);
    }

    protected Queue<cudaEvent_t> getReadLanes(INDArray array) {
        AllocationPoint point = allocator.getAllocationPoint(array);

        return point.getReadLane();
    }

    protected cudaEvent_t getWriteLane(INDArray array) {
        AllocationPoint point = allocator.getAllocationPoint(array);

        return point.getWriteLane();
    }

    protected int hasActiveWrite(INDArray array) {
        if (array == null) return -1;

        cudaEvent_t event = getWriteLane(array);
        if (event == null || event.isDestroyed()) return -1;

        return event.getLaneId();
    }

    protected boolean hasActiveReads(AllocationPoint point) {
        Queue<cudaEvent_t> events = point.getReadLane();

        if (events.size() == 0) return false;

        AtomicBoolean result = new AtomicBoolean(false);
        List<cudaEvent_t> asList = new ArrayList<>(events);
        for (cudaEvent_t event: asList) {
            if (event == null) continue;

            // we mark this AllocationPoint is pending read, if at least one event isn't destroyed yet
            result.compareAndSet(false, !event.isDestroyed());
        }

        return result.get();
    }

    protected boolean hasActiveReads(INDArray array) {
        if (array == null) return false;

        AllocationPoint point = allocator.getAllocationPoint(array);

        return hasActiveReads(point);
    }

    protected boolean isMatchingLanes(int[] lanes) {
        if (lanes[0] == lanes[1] || lanes[1] == -1 || lanes[0] == -1)
            return true;
        return false;
    }

    protected boolean isMatchingLanes(int zLane, int[] lanes) {
        if ((zLane == lanes[0] || zLane == lanes[1]) && isMatchingLanes(lanes))
            return true;
        return false;
    }

    protected void synchronizeReadLanes(AllocationPoint point) {
        cudaEvent_t event;
        int cnt = 0;
        while ((event = point.getReadLane().poll()) != null) {
            event.synchronize();
            event.destroy();
            cnt++;
        }
      //  log.info("Events synchronized: [{}]", cnt);
    }

    protected void synchronizeReadLanes(INDArray array) {
        if (array == null) return;

        AllocationPoint point = allocator.getAllocationPoint(array);
        synchronizeReadLanes(point);
    }

    @Override
    public void registerAction(CudaContext context, AllocationPoint result, AllocationPoint... operands) {
        cudaEvent_t event = new cudaEvent_t(nativeOps.createEvent());
        event.setLaneId(context.getLaneId());
        nativeOps.registerEvent(event.address(), context.getOldStream().address());

        result.setWriteLane(event);
    }

    @Override
    public CudaContext prepareAction(AllocationPoint result, AllocationPoint... operands) {
        if (hasActiveReads(result))
            synchronizeReadLanes(result);

        ContextPack pack = allocator.getContextPool().acquireContextPackForDevice(allocator.getDeviceId());

        return pack.getContextForLane(pack.nextRandomLane());
    }


    protected int pickFirstLane(int[] lanes) {
        if (lanes[0] >= 0)
            return lanes[0];
        else if (lanes[1] >= 0)
            return lanes[1];

        return 0;
    }

    @Override
    public CudaContext prepareAction(INDArray result, INDArray... operands) {

        /**
         * This method should decide, which CUDA stream should be used for execution, based on data affinity
         * Decision is made based on data affinity, at INDArray level solely.
         */

        ContextPack pack = allocator.getContextPool().acquireContextPackForDevice(allocator.getDeviceId());

        // for result holding lane do not really matters, only depending lanes to matter, because they are used to read
        // default lane is lane_0
        int newLane = 0;
        int zLane = hasActiveWrite(result);
        boolean zReads = hasActiveReads(result);

        if (result != null && (zReads || zLane >= 0)) {
            // we send this op to the same lane as active read/write event


            // but we still have to check, if op.X and op.Y has pending writes on other lanes
          //  log.info("Busy Z dep: [{}], hasReads: [{}]", zLane, zReads);

            AtomicInteger cnt = new AtomicInteger(0);
            AtomicInteger holdersCount = new AtomicInteger(0);
            int lastLane = -1;
            int pendingLanes[] = new int[]{-1, -1};
            for (INDArray operand: operands) {
                if (operand == null) continue;

                int lane = hasActiveWrite(operand);
                if (lane >= 0) {
                    // at least one operand has pendingWrite. And we don't care about pending reads.
                    pendingLanes[cnt.get()] = lane;
                    holdersCount.incrementAndGet();
                    lastLane = lane;
                }
                cnt.incrementAndGet();
            }

            if (zReads) {
          //      log.info("Synchronizing zReads");
                synchronizeReadLanes(result);
            }

            if (holdersCount.get() > 0) {
                asyncMiss.incrementAndGet();

                if (isMatchingLanes(zLane, pendingLanes)) {
                 //   log.info("All matching lanes additional deps in [{}] -> [{}, {}]", zLane, pendingLanes[0], pendingLanes[1]);
                    if (zLane >= 0)
                        newLane = zLane;
                    else
                        newLane = pickFirstLane(pendingLanes);
                } else {
                 //   log.info("Mismatching lanes additional deps in [{}] -> [{}, {}]", zLane, pendingLanes[0], pendingLanes[1]);
                    // now we must sync on both pendingLanes and pass data to zLane
                    if (zLane >= 0)
                        newLane = zLane;
                    else newLane = pickFirstLane(pendingLanes);

                    for (INDArray operand: operands) {
                        if (operand == null) continue;

                        waitTillFinished(allocator.getAllocationPoint(operand));
                    }
                }
            } else {
          //      log.info("Only Z is holder: [{}]", zLane);

                asyncHit.incrementAndGet();

                if (zLane < 0)
                    zLane = pack.nextRandomLane();

                newLane = zLane;
            }
        } else {
            // we go and check op.X and op.Y
            AtomicInteger cnt = new AtomicInteger(0);
            AtomicInteger holdersCount = new AtomicInteger(0);
            int lastLane = -1;
            // FIXME: this array must be dynamic, bound to configuration
            int pendingLanes[] = new int[]{-1, -1, -1, -1};
            for (INDArray operand: operands) {
                if (operand == null) continue;

                int lane = hasActiveWrite(operand);
                if (lane >= 0) {
                    // at least one operand has pendingWrite. And we don't care about pending reads.
                    pendingLanes[cnt.get()] = lane;
                    holdersCount.incrementAndGet();
                    lastLane = lane;
                }
                cnt.incrementAndGet();
            }

            if (holdersCount.get() > 0) {
                // we have some holders here
                asyncMiss.incrementAndGet();
                if (isMatchingLanes(pendingLanes)) {
                    // if op.X and/or op.Y has pending write in same lane - just throw op to that lane, and enjoy
                    newLane = lastLane;
               //     log.info("Paired dependencies: [{}]", newLane);
                } else {
                    // we have different lanes for op.X and op.Y with pending write. We need to synchronize somewhere to become free.
                    // basically - synchronize on one lane, and throw task to another one
                    //log.info("Unpaired dependencies: [{}, {}]", pendingLanes[0], pendingLanes[1]);
                    if (pendingLanes[0] >= 0) {
                        waitTillFinished(allocator.getAllocationPoint(operands[0]));
                        newLane = pendingLanes[1];
                    } else if (pendingLanes[1] >= 0) {
                        waitTillFinished(allocator.getAllocationPoint(operands[1]));
                        newLane = pendingLanes[0];
                    }
                }
            } else {
                // we don't have any holders here. Totally free execution here
                asyncHit.incrementAndGet();

                newLane = pack.nextRandomLane();

             //   log.info("Free pass here: [{}]", newLane);
            }
        }



        CudaContext context = pack.getContextForLane(newLane);

        if (result != null)
            allocator.getAllocationPoint(result).setCurrentContext(context);

        for (INDArray operand: operands) {
            if (operand == null) continue;

            allocator.getAllocationPoint(operand).setCurrentContext(context);
        }

        if (!lanesCounter.containsKey(newLane)) {
            lanesCounter.put(newLane, new AtomicLong(0));
        }

        lanesCounter.get(newLane).incrementAndGet();

        if (context == null)
            throw new IllegalStateException("Context shouldn't be null: " + newLane);

        return context;
    }

    private float getAsyncHitRatio() {
        long totalHits = asyncHit.get() + asyncMiss.get();
        float cacheRatio = asyncHit.get() * 100 / (float) totalHits;
        return cacheRatio;
    }
}
