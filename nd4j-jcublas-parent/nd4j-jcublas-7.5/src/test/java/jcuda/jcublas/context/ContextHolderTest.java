/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package jcuda.jcublas.context;


import static org.junit.Assume.*;

import jcuda.driver.CUcontext;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;

/**
 * @author Adam Gibson
 */
public class ContextHolderTest {
    @Test
    public void testContextHolder() throws Exception  {
        ContextHolder holder = ContextHolder.getInstance();
        CUcontext ctx = holder.getContext();
        assumeNotNull(ctx);
        assumeTrue(holder.getDeviceIDContexts().size() == 1);

    }

    @Test
    public void testLoadFunction() {
        assumeNotNull(KernelFunctionLoader.launcher("std_strided", DataBuffer.Type.DOUBLE));
    }

}
