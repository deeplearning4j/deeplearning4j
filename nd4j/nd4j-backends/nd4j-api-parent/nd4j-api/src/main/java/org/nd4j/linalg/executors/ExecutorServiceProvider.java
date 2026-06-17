/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.executors;

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.config.SystemPropertyUtils;
import java.util.concurrent.*;

public class ExecutorServiceProvider {

    // Centralized in ND4JSystemProperties; kept here as aliases for backward compatibility
    public static final String EXEC_THREADS = ND4JSystemProperties.PARALLEL_THREADS;
    public final static String ENABLED = ND4JSystemProperties.PARALLEL_ENABLED;

    private static final int nThreads;
    private static ExecutorService executorService;
    private static ForkJoinPool forkJoinPool;

    static {
        int defaultThreads = Runtime.getRuntime().availableProcessors();
        boolean enabled = SystemPropertyUtils.getBooleanProperty(ND4JSystemProperties.PARALLEL_ENABLED, true);
        if (!enabled)
            nThreads = 1;
        else
            nThreads = SystemPropertyUtils.getIntProperty(ND4JSystemProperties.PARALLEL_THREADS, defaultThreads);
    }

    public static synchronized ExecutorService getExecutorService() {
        if (executorService != null)
            return executorService;

        executorService = new ThreadPoolExecutor(nThreads, nThreads, 60L, TimeUnit.SECONDS,
                        new LinkedTransferQueue<Runnable>(), new ThreadFactory() {
                            @Override
                            public Thread newThread(Runnable r) {
                                Thread t = Executors.defaultThreadFactory().newThread(r);
                                t.setDaemon(true);
                                return t;
                            }
                        });
        return executorService;
    }

    public static synchronized ForkJoinPool getForkJoinPool() {
        if (forkJoinPool != null)
            return forkJoinPool;
        forkJoinPool = new ForkJoinPool(nThreads, ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
        return forkJoinPool;
    }

}
