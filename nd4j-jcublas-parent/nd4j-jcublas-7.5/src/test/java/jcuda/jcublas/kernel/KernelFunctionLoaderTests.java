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

package jcuda.jcublas.kernel;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.Properties;


/**
 * Created by agibsonccc on 2/27/15.
 */
public class KernelFunctionLoaderTests {

    @Test
    public void testLoader() throws Exception {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        KernelFunctionLoader loader = KernelFunctionLoader.getInstance();
        loader.load();
        ClassPathResource res = new ClassPathResource("/cudafunctions.properties", KernelFunctionLoader.class.getClassLoader());
        if (!res.exists())
            throw new IllegalStateException("Please put a cudafunctions.properties in your class path");
        Properties props = new Properties();
        props.load(res.getInputStream());
        loader.unload();

    }



    @Test
    public void testKernelElementWise() {

    }

}
