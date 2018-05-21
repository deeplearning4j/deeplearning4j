package org.nd4j.jita.allocator.context.impl;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.jita.allocator.context.ContextPack;
import org.nd4j.jita.allocator.context.ContextPool;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
@Deprecated
@Slf4j
public class PackedContextPool extends BasicContextPool implements ContextPool {

    protected static final int LANES_PER_THREAD =
                    CudaEnvironment.getInstance().getConfiguration().getCommandLanesNumber();

    private volatile Map<Long, ContextPack> contextsPool = new ConcurrentHashMap<>();

    @Override
    public CudaContext acquireContextForDevice(Integer deviceId) {
        return acquireContextPackForDevice(deviceId).getContextForLane(0);
    }

    @Override
    public ContextPack acquireContextPackForDevice(Integer deviceId) {
        Long threadId = Thread.currentThread().getId();
        if (!contextsPool.containsKey(threadId)) {
            try {
                lock.acquire();

                ContextPack pack = new ContextPack(LANES_PER_THREAD);
                for (int c = 0; c < LANES_PER_THREAD; c++) {
                    CudaContext context = createNewStream(deviceId);

                    getDeviceBuffers(context, deviceId);

                    if (cublasPool.get(deviceId) == null) {
                        // if we have no contexts created - it's just awesome time to attach cuBLAS handle here
                        log.debug("Creating new cuBLAS handle for device [{}]", deviceId);

                        cudaStream_t cublasStream = createNewStream(deviceId).getOldStream();

                        cublasHandle_t handle = createNewCublasHandle(cublasStream);
                        context.setHandle(handle);
                        context.setCublasStream(cublasStream);

                        cublasPool.put(deviceId, handle);

                        log.debug("Creating new cuSolver handle for device [{}]...", deviceId);

                        cudaStream_t solverStream = createNewStream(deviceId).getOldStream();

                        cusolverDnHandle_t solverhandle = createNewSolverHandle(solverStream);
                        context.setSolverHandle(solverhandle);
                        context.setSolverStream(solverStream);

                        solverPool.put(deviceId, solverhandle);

                    } else {
                        // just pick handle out there
                        log.debug("Reusing cuBLAS handle for device [{}]", deviceId);
                        cublasHandle_t handle = cublasPool.get(deviceId);
                        context.setHandle(handle);

                        log.debug("Reusing solver here...");
                        cusolverDnHandle_t solverHandle = solverPool.get(deviceId);
                        context.setSolverHandle(solverHandle);
                    }

                    pack.addLane(c, context);
                }

                contextsPool.put(threadId, pack);


            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                lock.release();
            }
        }

        return contextsPool.get(threadId);
    }
}
