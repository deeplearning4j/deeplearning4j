package org.nd4j.jita.allocator.context;

import lombok.Data;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 * @author raver119@gmail.com
 */
@Data
public class ExternalCudaContext extends ExternalContext {
    private CudaContext context;
}
