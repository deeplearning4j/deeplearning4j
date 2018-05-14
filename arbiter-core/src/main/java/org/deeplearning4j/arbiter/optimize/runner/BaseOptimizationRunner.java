/*-
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
package org.deeplearning4j.arbiter.optimize.runner;

import com.google.common.util.concurrent.ListenableFuture;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * BaseOptimization runner: responsible for scheduling tasks, saving results using the result saver, etc.
 *
 * @author Alex Black
 */
@Slf4j
public abstract class BaseOptimizationRunner implements IOptimizationRunner {
    private static final int POLLING_FREQUENCY = 1;
    private static final TimeUnit POLLING_FREQUENCY_UNIT = TimeUnit.SECONDS;

    protected OptimizationConfiguration config;
    protected Queue<Future<OptimizationResult>> queuedFutures = new ConcurrentLinkedQueue<>();
    protected BlockingQueue<Future<OptimizationResult>> completedFutures = new LinkedBlockingQueue<>();
    protected AtomicInteger totalCandidateCount = new AtomicInteger();
    protected AtomicInteger numCandidatesCompleted = new AtomicInteger();
    protected AtomicInteger numCandidatesFailed = new AtomicInteger();
    protected Double bestScore = null;
    protected Long bestScoreTime = null;
    protected AtomicInteger bestScoreCandidateIndex = new AtomicInteger(-1);
    protected List<ResultReference> allResults = new ArrayList<>();

    protected Map<Integer, CandidateInfo> currentStatus = new ConcurrentHashMap<>(); //TODO: better design possible?

    protected ExecutorService futureListenerExecutor;

    protected List<StatusListener> statusListeners = new ArrayList<>();


    protected BaseOptimizationRunner(OptimizationConfiguration config) {
        this.config = config;

        if (config.getTerminationConditions() == null || config.getTerminationConditions().size() == 0) {
            throw new IllegalArgumentException("Cannot create BaseOptimizationRunner without TerminationConditions ("
                    + "termination conditions are null or empty)");
        }

    }

    protected void init() {
        futureListenerExecutor = Executors.newFixedThreadPool(maxConcurrentTasks(), new ThreadFactory() {
            private AtomicLong counter = new AtomicLong(0);

            @Override
            public Thread newThread(Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);
                t.setDaemon(true);
                t.setName("ArbiterOptimizationRunner-" + counter.getAndIncrement());
                return t;
            }
        });
    }

    /**
     *
     */
    @Override
    public void execute() {
        log.info("{}: execution started", this.getClass().getSimpleName());
        config.setExecutionStartTime(System.currentTimeMillis());
        for (StatusListener listener : statusListeners) {
            listener.onInitialization(this);
        }

        //Initialize termination conditions (start timers, etc)
        for (TerminationCondition c : config.getTerminationConditions()) {
            c.initialize(this);
        }

        //Queue initial tasks:
        List<Future<OptimizationResult>> tempList = new ArrayList<>(100);
        while (true) {
            //Otherwise: add tasks if required
            Future<OptimizationResult> future = null;
            try {
                future = completedFutures.poll(POLLING_FREQUENCY, POLLING_FREQUENCY_UNIT);
            } catch (InterruptedException e) {
                //No op?
            }
            if (future != null) {
                tempList.add(future);
            }
            completedFutures.drainTo(tempList);

            //Process results (if any)
            for (Future<OptimizationResult> f : tempList) {
                queuedFutures.remove(f);
                processReturnedTask(f);
            }

            if (tempList.size() > 0) {
                for (StatusListener sl : statusListeners) {
                    sl.onRunnerStatusChange(this);
                }
            }
            tempList.clear();

            //Check termination conditions:
            if (terminate()) {
                shutdown();
                break;
            }

            //Add additional tasks
            while (config.getCandidateGenerator().hasMoreCandidates() && queuedFutures.size() < maxConcurrentTasks()) {
                Candidate candidate = config.getCandidateGenerator().getCandidate();
                CandidateInfo status;
                if (candidate.getException() != null) {
                    //Failed on generation...
                    status = processFailedCandidates(candidate);
                } else {
                    ListenableFuture<OptimizationResult> f =
                            execute(candidate, config.getDataProvider(), config.getScoreFunction());
                    f.addListener(new OnCompletionListener(f), futureListenerExecutor);
                    queuedFutures.add(f);
                    totalCandidateCount.getAndIncrement();

                    status = new CandidateInfo(candidate.getIndex(), CandidateStatus.Created, null,
                            System.currentTimeMillis(), null, null, candidate.getFlatParameters(), null);
                    currentStatus.put(candidate.getIndex(), status);
                }

                for (StatusListener listener : statusListeners) {
                    listener.onCandidateStatusChange(status, this, null);
                }
            }
        }

        //Process any final (completed) tasks:
        completedFutures.drainTo(tempList);
        for (Future<OptimizationResult> f : tempList) {
            queuedFutures.remove(f);
            processReturnedTask(f);
        }
        tempList.clear();

        log.info("Optimization runner: execution complete");
        for (StatusListener listener : statusListeners) {
            listener.onShutdown(this);
        }
    }


    private CandidateInfo processFailedCandidates(Candidate<?> candidate) {
        //In case the candidate fails during the creation of the candidate

        long time = System.currentTimeMillis();
        String stackTrace = ExceptionUtils.getStackTrace(candidate.getException());
        CandidateInfo newStatus = new CandidateInfo(candidate.getIndex(), CandidateStatus.Failed, null, time, time,
                time, candidate.getFlatParameters(), stackTrace);
        currentStatus.put(candidate.getIndex(), newStatus);

        return newStatus;
    }

    /**
     * Process returned task (either completed or failed
     */
    private void processReturnedTask(Future<OptimizationResult> future) {
        long currentTime = System.currentTimeMillis();
        OptimizationResult result;
        try {
            result = future.get(100, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            throw new RuntimeException("Unexpected InterruptedException thrown for task", e);
        } catch (ExecutionException e) {
            //Note that most of the time, an OptimizationResult is returned even for an exception
            //This is just to handle any that are missed there (or, by implementations that don't properly do this)
            log.warn("Task failed", e);

            numCandidatesFailed.getAndIncrement();
            return;
        } catch (TimeoutException e) {
            throw new RuntimeException(e); //TODO
        }

        //Update internal status:
        CandidateInfo status = currentStatus.get(result.getIndex());
        CandidateInfo newStatus = new CandidateInfo(result.getIndex(), result.getCandidateInfo().getCandidateStatus(),
                result.getScore(), status.getCreatedTime(), result.getCandidateInfo().getStartTime(),
                currentTime, status.getFlatParams(), result.getCandidateInfo().getExceptionStackTrace());
        currentStatus.put(result.getIndex(), newStatus);

        //Listeners (on complete, etc) should be executed in underlying task


        if (result.getCandidateInfo().getCandidateStatus() == CandidateStatus.Failed) {
            log.info("Task {} failed during execution: {}", result.getIndex(), result.getCandidateInfo().getExceptionStackTrace());
            numCandidatesFailed.getAndIncrement();
        } else {

            //Report completion to candidate generator
            config.getCandidateGenerator().reportResults(result);

            Double score = result.getScore();
            log.info("Completed task {}, score = {}", result.getIndex(), result.getScore());

            boolean minimize = config.getScoreFunction().minimize();
            if (score != null && (bestScore == null
                    || ((minimize && score < bestScore) || (!minimize && score > bestScore)))) {
                if (bestScore == null) {
                    log.info("New best score: {} (first completed model)", score);
                } else {
                    int idx = result.getIndex();
                    int lastBestIdx = bestScoreCandidateIndex.get();
                    log.info("New best score: {}, model {} (prev={}, model {})", score, idx, bestScore, lastBestIdx);
                }
                bestScore = score;
                bestScoreTime = System.currentTimeMillis();
                bestScoreCandidateIndex.set(result.getIndex());
            }
            numCandidatesCompleted.getAndIncrement();

            //Model saving is done in the optimization tasks, to avoid CUDA threading issues
            ResultReference resultReference = result.getResultReference();

            if (resultReference != null)
                allResults.add(resultReference);
        }
    }

    @Override
    public int numCandidatesTotal() {
        return totalCandidateCount.get();
    }

    @Override
    public int numCandidatesCompleted() {
        return numCandidatesCompleted.get();
    }

    @Override
    public int numCandidatesFailed() {
        return numCandidatesFailed.get();
    }

    @Override
    public int numCandidatesQueued() {
        return queuedFutures.size();
    }

    @Override
    public Double bestScore() {
        return bestScore;
    }

    @Override
    public Long bestScoreTime() {
        return bestScoreTime;
    }

    @Override
    public int bestScoreCandidateIndex() {
        return bestScoreCandidateIndex.get();
    }

    @Override
    public List<ResultReference> getResults() {
        return new ArrayList<>(allResults);
    }

    @Override
    public OptimizationConfiguration getConfiguration() {
        return config;
    }


    @Override
    public void addListeners(StatusListener... listeners) {
        for (StatusListener l : listeners) {
            if (!statusListeners.contains(l)) {
                statusListeners.add(l);
            }
        }
    }

    @Override
    public void removeListeners(StatusListener... listeners) {
        for (StatusListener l : listeners) {
            if (statusListeners.contains(l)) {
                statusListeners.remove(l);
            }
        }
    }

    @Override
    public void removeAllListeners() {
        statusListeners.clear();
    }

    @Override
    public List<CandidateInfo> getCandidateStatus() {
        List<CandidateInfo> list = new ArrayList<>();
        list.addAll(currentStatus.values());
        return list;
    }

    private boolean terminate() {
        for (TerminationCondition c : config.getTerminationConditions()) {
            if (c.terminate(this)) {
                log.info("BaseOptimizationRunner global termination condition hit: {}", c);
                return true;
            }
        }
        return false;
    }

    @AllArgsConstructor
    @Data
    private class FutureDetails {
        private final Future<OptimizationResult> future;
        private final long startTime;
        private final int index;
    }

    @AllArgsConstructor
    private class OnCompletionListener implements Runnable {
        private Future<OptimizationResult> future;

        @Override
        public void run() {
            completedFutures.add(future);
        }
    }


    protected abstract int maxConcurrentTasks();

    protected abstract ListenableFuture<OptimizationResult> execute(Candidate candidate, DataProvider dataProvider,
                                                                    ScoreFunction scoreFunction);

    protected abstract List<ListenableFuture<OptimizationResult>> execute(List<Candidate> candidates,
                                                                          DataProvider dataProvider, ScoreFunction scoreFunction);

    protected abstract void shutdown();
}
