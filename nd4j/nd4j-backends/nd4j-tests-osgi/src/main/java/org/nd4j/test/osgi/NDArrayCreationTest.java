package org.nd4j.test.osgi;
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


import static org.junit.Assert.assertEquals;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This test is based on the TestNDArrayCreation test from the nd4j-tests module
 */
public class NDArrayCreationTest {

	private static Logger log = LoggerFactory.getLogger(NDArrayCreationTest.class);
	
    @Before
    public void before() throws Exception {
        Nd4j nd4j = new Nd4j();
        nd4j.initWithBackend(Nd4jBackend.load());
        Nd4j.factory().setOrder('c');
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    @After
    public void after() throws Exception {
        Nd4j.getMemoryManager().purgeCaches();

        //Attempt to keep workspaces isolated between tests
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        MemoryWorkspace currWS = Nd4j.getMemoryManager().getCurrentWorkspace();
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        if(currWS != null){
            //Not really safe to continue testing under this situation... other tests will likely fail with obscure
            // errors that are hard to track back to this
            log.error("Open workspace leaked from test! Exiting - {}, isOpen = {} - {}", currWS.getId(), currWS.isScopeActive(), currWS);
            System.exit(1);
        }
    }

    @Test
    // We only run with CPU for now
    //@Ignore("AB 2019/05/23 - Failing on linux-x86_64-cuda-9.2 - see issue #7657")
    public void testBufferCreation() {
        DataBuffer dataBuffer = Nd4j.createBuffer(new float[] {1, 2});
        Pointer pointer = dataBuffer.pointer();
        FloatPointer floatPointer = new FloatPointer(pointer);
        DataBuffer dataBuffer1 = Nd4j.createBuffer(floatPointer, 2, DataType.FLOAT);

        assertEquals(2, dataBuffer.length());
        assertEquals(1.0, dataBuffer.getDouble(0), 1e-1);
        assertEquals(2.0, dataBuffer.getDouble(1), 1e-1);

        assertEquals(2, dataBuffer1.length());
        assertEquals(1.0, dataBuffer1.getDouble(0), 1e-1);
        assertEquals(2.0, dataBuffer1.getDouble(1), 1e-1);
        INDArray arr = Nd4j.create(dataBuffer1);
        System.out.println(arr);
    }

}
