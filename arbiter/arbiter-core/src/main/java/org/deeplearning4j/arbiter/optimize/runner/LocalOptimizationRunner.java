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

package org.deeplearning4j.arbiter.optimize.runner;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import lombok.Setter;
import org.deeplearning4j.arbiter.optimize.api.*;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * LocalOptimizationRunner: execute hyperparameter optimization
 * locally (on current machine, in current JVM).
 *
 * @author Alex Black
 */
public class LocalOptimizationRunner extends BaseOptimizationRunner {

    public static final int DEFAULT_MAX_CONCURRENT_TASKS = 1;

    private final int maxConcurrentTasks;

    private TaskCreator taskCreator;
    private ListeningExecutorService executor;
    @Setter
    private long shutdownMaxWaitMS = 2L * 24 * 60 * 60 * 1000;

    public LocalOptimizationRunner(OptimizationConfiguration config){
        this(config, null);
    }

    public LocalOptimizationRunner(OptimizationConfiguration config, TaskCreator taskCreator) {
        this(DEFAULT_MAX_CONCURRENT_TASKS, config, taskCreator);
    }

    public LocalOptimizationRunner(int maxConcurrentTasks, OptimizationConfiguration config){
        this(maxConcurrentTasks, config, null);
    }

    public LocalOptimizationRunner(int maxConcurrentTasks, OptimizationConfiguration config, TaskCreator taskCreator) {
        super(config);
        if (maxConcurrentTasks <= 0)
            throw new IllegalArgumentException("maxConcurrentTasks must be > 0 (got: " + maxConcurrentTasks + ")");
        this.maxConcurrentTasks = maxConcurrentTasks;

        if(taskCreator == null){
            Class<? extends ParameterSpace> psClass = config.getCandidateGenerator().getParameterSpace().getClass();
            taskCreator = TaskCreatorProvider.defaultTaskCreatorFor(psClass);
            if(taskCreator == null){
                throw new IllegalStateException("No TaskCreator was provided and a default TaskCreator cannot be " +
                        "inferred for ParameterSpace class " + psClass.getName() + ". Please provide a TaskCreator " +
                        "via the LocalOptimizationRunner constructor");
            }
        }

        this.taskCreator = taskCreator;

        ExecutorService exec = Executors.newFixedThreadPool(maxConcurrentTasks, new ThreadFactory() {
            private AtomicLong counter = new AtomicLong(0);

            @Override
            public Thread newThread(Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);
                t.setDaemon(true);
                t.setName("LocalCandidateExecutor-" + counter.getAndIncrement());
                return t;
            }
        });
        executor = MoreExecutors.listeningDecorator(exec);

        init();
    }

    @Override
    protected int maxConcurrentTasks() {
        return maxConcurrentTasks;
    }

    @Override
    protected ListenableFuture<OptimizationResult> execute(Candidate candidate, DataProvider dataProvider,
                    ScoreFunction scoreFunction) {
        return execute(Collections.singletonList(candidate), dataProvider, scoreFunction).get(0);
    }

    @Override
    protected List<ListenableFuture<OptimizationResult>> execute(List<Candidate> candidates, DataProvider dataProvider,
                    ScoreFunction scoreFunction) {
        List<ListenableFuture<OptimizationResult>> list = new ArrayList<>(candidates.size());
        for (Candidate candidate : candidates) {
            Callable<OptimizationResult> task =
                            taskCreator.create(candidate, dataProvider, scoreFunction, statusListeners, this);
            list.add(executor.submit(task));
        }
        return list;
    }

    @Override
    public void shutdown(boolean awaitTermination) {
        if(awaitTermination){
            try {
                executor.shutdown();
                executor.awaitTermination(shutdownMaxWaitMS, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e){
                throw new RuntimeException(e);
            }
        } else {
            executor.shutdownNow();
        }
    }
}
