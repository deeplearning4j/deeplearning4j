/*-
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
 */

package org.deeplearning4j.clustering.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.*;

public class MultiThreadUtils {

    private static Logger log = LoggerFactory.getLogger(MultiThreadUtils.class);

    private static ExecutorService instance;

    private MultiThreadUtils() {}

    public static synchronized ExecutorService newExecutorService() {
        int nThreads = Runtime.getRuntime().availableProcessors();
        return new ThreadPoolExecutor(nThreads, nThreads, 60L, TimeUnit.SECONDS, new LinkedTransferQueue<Runnable>(),
                        new ThreadFactory() {
                            @Override
                            public Thread newThread(Runnable r) {
                                Thread t = Executors.defaultThreadFactory().newThread(r);
                                t.setDaemon(true);
                                return t;
                            }
                        });
    }

    public static void parallelTasks(final List<Runnable> tasks, ExecutorService executorService) {
        int tasksCount = tasks.size();
        final CountDownLatch latch = new CountDownLatch(tasksCount);
        for (int i = 0; i < tasksCount; i++) {
            final int taskIdx = i;
            executorService.execute(new Runnable() {
                public void run() {
                    try {
                        tasks.get(taskIdx).run();
                    } catch (Throwable e) {
                        log.info("Unchecked exception thrown by task", e);
                    } finally {
                        latch.countDown();
                    }
                }
            });
        }

        try {
            latch.await();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
