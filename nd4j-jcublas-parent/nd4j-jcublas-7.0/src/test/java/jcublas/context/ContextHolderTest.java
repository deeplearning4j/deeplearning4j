package jcublas.context;


import static org.junit.Assume.*;

import jcuda.driver.CUcontext;
import org.junit.Test;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;

/**
 * @author Adam Gibson
 */
public class ContextHolderTest {
    @Test
    public void testContextHolder() {
        SimpleJCublas.init();
        ContextHolder holder = ContextHolder.getInstance();
        CUcontext ctx = holder.getContext();
        assumeNotNull(ctx);
        assumeTrue(holder.getDeviceIDContexts().size() == 1);

    }

    @Test
    public void testLoadFunction() {
        assumeNotNull(KernelFunctionLoader.launcher("std_strided","double"));
    }

}
