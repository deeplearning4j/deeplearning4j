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

package org.nd4j.linalg.executors;

import java.util.concurrent.*;

public class ExecutorServiceProvider {

    public static final String EXEC_THREADS = "org.nd4j.parallel.threads";
    public final static String ENABLED = "org.nd4j.parallel.enabled";

    private static final int nThreads;
    private static ExecutorService executorService;
    private static ForkJoinPool forkJoinPool;

    static {
        int defaultThreads = Runtime.getRuntime().availableProcessors();
        boolean enabled = Boolean.parseBoolean(System.getProperty(ENABLED, "true"));
        if (!enabled)
            nThreads = 1;
        else
            nThreads = Integer.parseInt(System.getProperty(EXEC_THREADS, String.valueOf(defaultThreads)));
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
