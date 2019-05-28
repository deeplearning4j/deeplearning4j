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

package org.deeplearning4j.integration;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.rules.TestName;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;
import java.util.Properties;

@Slf4j
public class BaseDL4JTest {

    @Rule
    public TestName name = new TestName();

    protected long startTime;

    /**
     * Override this to set the profiling mode for the tests defined in the child class
     */
    public OpExecutioner.ProfilingMode getProfilingMode(){
        return OpExecutioner.ProfilingMode.SCOPE_PANIC;
    }

    /**
     * Override this to set the datatype of the tests defined in the child class
     */
    public DataType getDataType(){
        return DataType.FLOAT;
    }

    @Before
    public void beforeTest(){
        log.info("{}.{}", getClass().getSimpleName(), name.getMethodName());
        Nd4j.getExecutioner().setProfilingMode(getProfilingMode());
        Nd4j.setDataType(getDataType());
    }

    @After
    public void afterTest(){
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

        StringBuilder sb = new StringBuilder();
        long maxPhys = Pointer.maxPhysicalBytes();
        long maxBytes = Pointer.maxBytes();
        long currPhys = Pointer.physicalBytes();
        long currBytes = Pointer.totalBytes();

        long jvmTotal = Runtime.getRuntime().totalMemory();
        long jvmMax = Runtime.getRuntime().maxMemory();

        long duration = System.currentTimeMillis() - startTime;
        sb.append(getClass().getSimpleName()).append(".").append(name.getMethodName())
                .append(": ").append(duration).append(" ms")
                .append(", jvmTotal=").append(jvmTotal)
                .append(", jvmMax=").append(jvmMax)
                .append(", totalBytes=").append(currBytes).append(", maxBytes=").append(maxBytes)
                .append(", currPhys=").append(currPhys).append(", maxPhys=").append(maxPhys);

        List<MemoryWorkspace> ws = Nd4j.getWorkspaceManager().getAllWorkspacesForCurrentThread();
        if(ws != null && ws.size() > 0){
            long currSize = 0;
            for(MemoryWorkspace w : ws){
                currSize += w.getCurrentSize();
            }
            if(currSize > 0){
                sb.append(", threadWSSize=").append(currSize)
                        .append(" (").append(ws.size()).append(" WSs)");
            }
        }


        Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
        Object o = p.get("cuda.devicesInformation");
        if(o instanceof List){
            List<Map<String,Object>> l = (List<Map<String, Object>>) o;
            if(l.size() > 0) {

                sb.append(" [").append(l.size())
                        .append(" GPUs: ");

                for (int i = 0; i < l.size(); i++) {
                    Map<String,Object> m = l.get(i);
                    if(i > 0)
                        sb.append(",");
                    sb.append("(").append(m.get("cuda.freeMemory")).append(" free, ")
                            .append(m.get("cuda.totalMemory")).append(" total)");
                }
                sb.append("]");
            }
        }
        log.info(sb.toString());
    }

}
