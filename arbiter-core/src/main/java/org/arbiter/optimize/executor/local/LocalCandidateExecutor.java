/*
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.arbiter.optimize.executor.local;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.TaskCreator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.arbiter.optimize.executor.CandidateExecutor;
import org.arbiter.optimize.runner.listener.candidate.UICandidateStatusListenerImpl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;


public class LocalCandidateExecutor<T, M, D, A> implements CandidateExecutor<T, M, D, A> {

    private TaskCreator<T, M, D, A> taskCreator;
    private ListeningExecutorService executor;
    private final int nThreads;
    private final boolean reportResults;

    public LocalCandidateExecutor(TaskCreator<T, M, D, A> taskCreator, boolean reportResults) {
        this(taskCreator, reportResults, 1);
    }

    /**
     *
     * @param taskCreator
     * @param reportResults If true: report results to UI by adding a UICandidateStatusListener to each candidate
     * @param nThreads
     */
    public LocalCandidateExecutor(TaskCreator<T, M, D, A> taskCreator, boolean reportResults, int nThreads) {
        this.taskCreator = taskCreator;
        this.reportResults = reportResults;
        this.nThreads = nThreads;

        ExecutorService exec = Executors.newFixedThreadPool(nThreads, new ThreadFactory() {
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
    }


    @Override
    public ListenableFuture<OptimizationResult<T, M, A>> execute(Candidate<T> candidate, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction) {
        return execute(Collections.singletonList(candidate),dataProvider,scoreFunction).get(0);
    }

    @Override
    public List<ListenableFuture<OptimizationResult<T, M, A>>> execute(List<Candidate<T>> candidates, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction) {
        List<ListenableFuture<OptimizationResult<T, M, A>>> list = new ArrayList<>(candidates.size());
        for (Candidate<T> candidate : candidates) {
            Callable<OptimizationResult<T, M, A>> task = taskCreator.create(candidate, dataProvider, scoreFunction,
                    (reportResults ? new UICandidateStatusListenerImpl(candidate.getIndex()) : null));
            list.add(executor.submit(task));
        }
        return list;
    }

    @Override
    public int maxConcurrentTasks() {
        return nThreads;
    }

    @Override
    public void shutdown() {
        executor.shutdownNow();
    }

    @AllArgsConstructor
    private class Job {
        private Candidate<T> candidate;
        private DataProvider<D> dataProvider;
    }
}
