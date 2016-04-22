package org.nd4j.jita.allocator.context;

import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 *
 * @author raver119@gmail.com
 */
public interface ContextPool {
    CudaContext acquireContextForDevice(Integer deviceId);

    ContextPack acquireContextPackForDevice(Integer deviceId);
}
