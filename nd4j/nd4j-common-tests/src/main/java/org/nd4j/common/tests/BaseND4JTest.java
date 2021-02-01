/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.common.tests;

import ch.qos.logback.classic.LoggerContext;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.junit.*;
import org.junit.rules.TestName;
import org.junit.rules.Timeout;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.ProfilerConfig;
import org.slf4j.ILoggerFactory;
import org.slf4j.LoggerFactory;

import java.lang.management.ManagementFactory;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import static org.junit.Assume.assumeTrue;

@Slf4j
public abstract class BaseND4JTest {

    @Rule
    public TestName name = new TestName();
    @Rule
    public Timeout timeout = Timeout.millis(getTimeoutMilliseconds());

    protected long startTime;
    protected int threadCountBefore;

    /**
     * Override this method to set the default timeout for methods in the test class
     */
    public long getTimeoutMilliseconds(){
        return 90_000;
    }

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
        return DataType.DOUBLE;
    }

    /**
     * Override this to set the datatype of the tests defined in the child class
     */
    public DataType getDefaultFPDataType(){
        return getDataType();
    }

    private final int DEFAULT_THREADS = Runtime.getRuntime().availableProcessors();

    /**
     * Override this to specify the number of threads for C++ execution, via
     * {@link org.nd4j.linalg.factory.Environment#setMaxMasterThreads(int)}
     * @return Number of threads to use for C++ op execution
     */
    public int numThreads(){
        return DEFAULT_THREADS;
    }

    protected Boolean integrationTest;

    /**
     * @return True if integration tests maven profile is enabled, false otherwise.
     */
    public boolean isIntegrationTests(){
        if(integrationTest == null){
            String prop = System.getenv("DL4J_INTEGRATION_TESTS");
            integrationTest = Boolean.parseBoolean(prop);
        }
        return integrationTest;
    }

    /**
     * Call this as the first line of a test in order to skip that test, only when the integration tests maven profile is not enabled.
     * This can be used to dynamically skip integration tests when the integration test profile is not enabled.
     * Note that the integration test profile is not enabled by default - "integration-tests" profile
     */
    public void skipUnlessIntegrationTests(){
        assumeTrue("Skipping integration test - integration profile is not enabled", isIntegrationTests());
    }

    @Before
    public void beforeTest(){
        log.info("{}.{}", getClass().getSimpleName(), name.getMethodName());
        //Suppress ND4J initialization - don't need this logged for every test...
        System.setProperty(ND4JSystemProperties.LOG_INITIALIZATION, "false");
        System.setProperty(ND4JSystemProperties.ND4J_IGNORE_AVX, "true");
        Nd4j.getExecutioner().setProfilingMode(getProfilingMode());
        Nd4j.getExecutioner().setProfilingConfig(ProfilerConfig.builder().build());
        Nd4j.setDefaultDataTypes(getDataType(), getDefaultFPDataType());
        Nd4j.getExecutioner().setProfilingConfig(ProfilerConfig.builder().build());
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
        int numThreads = numThreads();
        Preconditions.checkState(numThreads > 0, "Number of threads must be > 0");
        if(numThreads != Nd4j.getEnvironment().maxMasterThreads()) {
            Nd4j.getEnvironment().setMaxMasterThreads(numThreads);
        }
        startTime = System.currentTimeMillis();
        threadCountBefore = ManagementFactory.getThreadMXBean().getThreadCount();
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
            System.out.println("Open workspace leaked from test! Exiting - " + currWS.getId() + ", isOpen = " + currWS.isScopeActive() + " - " + currWS);
            System.out.flush();
            //Try to flush logs also:
            try{ Thread.sleep(1000); } catch (InterruptedException e){ }
            ILoggerFactory lf = LoggerFactory.getILoggerFactory();
            if( lf instanceof LoggerContext){
                ((LoggerContext)lf).stop();
            }
            try{ Thread.sleep(1000); } catch (InterruptedException e){ }
            System.exit(1);
        }

        StringBuilder sb = new StringBuilder();
        long maxPhys = Pointer.maxPhysicalBytes();
        long maxBytes = Pointer.maxBytes();
        long currPhys = Pointer.physicalBytes();
        long currBytes = Pointer.totalBytes();

        long jvmTotal = Runtime.getRuntime().totalMemory();
        long jvmMax = Runtime.getRuntime().maxMemory();

        int threadsAfter = ManagementFactory.getThreadMXBean().getThreadCount();

        long duration = System.currentTimeMillis() - startTime;
        sb.append(getClass().getSimpleName()).append(".").append(name.getMethodName())
                .append(": ").append(duration).append(" ms")
                .append(", threadCount: (").append(threadCountBefore).append("->").append(threadsAfter).append(")")
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
