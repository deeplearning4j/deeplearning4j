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
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.listener.runner.OptimizationRunnerStatusListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * BaseOptimization runner: responsible for scheduling tasks, saving results using the result saver, etc.
 *
 * @param <C> Type of configuration
 * @param <M> Type of model learned
 * @param <D> Type of data used to train model
 * @param <A> Type of additional results
 * @author Alex Black
 */
@Slf4j
public abstract class BaseOptimizationRunner<C, M, D, A> implements IOptimizationRunner<C, M, A> {
    private static final int POLLING_FREQUENCY = 1;
    private static final TimeUnit POLLING_FREQUENCY_UNIT = TimeUnit.SECONDS;

    private OptimizationConfiguration<C, M, D, A> config;
    //    private CandidateExecutor<C, M, D, A> executor;
    private Queue<Future<OptimizationResult<C, M, A>>> queuedFutures = new ConcurrentLinkedQueue<>();
    private BlockingQueue<Future<OptimizationResult<C, M, A>>> completedFutures = new LinkedBlockingQueue<>();
    private int totalCandidateCount = 0;
    private int numCandidatesCompleted = 0;
    private int numCandidatesFailed = 0;
    private Double bestScore = null;
    private Long bestScoreTime = null;
    private int bestScoreCandidateIndex = -1;
    private List<ResultReference<C, M, A>> allResults = new ArrayList<>();

    private Map<Integer, CandidateStatus> currentStatus = new ConcurrentHashMap<>(); //TODO: better design possible?

    private ExecutorService futureListenerExecutor;

    private List<OptimizationRunnerStatusListener> statusListeners = new ArrayList<>();


    protected BaseOptimizationRunner(OptimizationConfiguration<C, M, D, A> config) {
        this.config = config;

        if (config.getTerminationConditions() == null || config.getTerminationConditions().size() == 0) {
            throw new IllegalArgumentException("Cannot create BaseOptimizationRunner without TerminationConditions (" +
                    "termination conditions are null or empty)");
        }

    }

    protected void init() {
        futureListenerExecutor = Executors.newFixedThreadPool(maxConcurrentTasks(),
                new ThreadFactory() {
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
        log.info("BaseOptimizationRunner: execution started");
        for (OptimizationRunnerStatusListener listener : statusListeners) listener.onInitialization(this);

        //Initialize termination conditions (start timers, etc)
        for (TerminationCondition c : config.getTerminationConditions()) {
            c.initialize(this);
        }

        //Queue initial tasks:


        List<Future<OptimizationResult<C, M, A>>> tempList = new ArrayList<>(100);
        while (true) {
            boolean statusChange = false;

            //Otherwise: add tasks if required
            Future<OptimizationResult<C, M, A>> future = null;
            try {
                future = completedFutures.poll(POLLING_FREQUENCY, POLLING_FREQUENCY_UNIT);
            } catch (InterruptedException e) {
                //No op?
            }
            if (future != null) tempList.add(future);
            completedFutures.drainTo(tempList);

            //Process results (if any)
            for (Future<OptimizationResult<C, M, A>> f : tempList) {
                queuedFutures.remove(f);
                processReturnedTask(f);
                statusChange = true;
            }
            tempList.clear();


            //Check termination conditions:
            if (terminate()) {
                shutdown();
                break;
            }

            //Add additional tasks
            while (config.getCandidateGenerator().hasMoreCandidates() && queuedFutures.size() < maxConcurrentTasks()) {
                Candidate<C> candidate = config.getCandidateGenerator().getCandidate();
                ListenableFuture<OptimizationResult<C, M, A>> f = execute(candidate, config.getDataProvider(), config.getScoreFunction());
                f.addListener(new OnCompletionListener(f), futureListenerExecutor);
                queuedFutures.add(f);
                totalCandidateCount++;
                statusChange = true;

                CandidateStatus status = new CandidateStatus(
                        candidate.getIndex(),
                        Status.Created,
                        null,
                        System.currentTimeMillis(),
                        null,
                        null);
                currentStatus.put(candidate.getIndex(), status);
            }

            if (statusChange) {
                for (OptimizationRunnerStatusListener listener : statusListeners) {
                    listener.onStatusChange(this);
                }
            }
        }

        //Process any final (completed) tasks:
        completedFutures.drainTo(tempList);
        for (Future<OptimizationResult<C, M, A>> f : tempList) {
            queuedFutures.remove(f);
            processReturnedTask(f);
        }
        tempList.clear();

        log.info("Optimization runner: execution complete");
        for (OptimizationRunnerStatusListener listener : statusListeners) listener.onShutdown(this);
    }

    /**
     * Process returned task (either completed or failed
     */
    private void processReturnedTask(Future<OptimizationResult<C, M, A>> future) {
        long currentTime = System.currentTimeMillis();
        //TODO: track and log execution time
        OptimizationResult<C, M, A> result;
        try {
            result = future.get(100, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            throw new RuntimeException("Unexpected InterruptedException thrown for task", e);
        } catch (ExecutionException e) {
            log.warn("Task failed", e);

            numCandidatesFailed++;
            return;
        } catch (TimeoutException e) {
            throw new RuntimeException(e);  //TODO
        }

        //Update internal status:
        CandidateStatus status = currentStatus.get(result.getIndex());
        CandidateStatus newStatus = new CandidateStatus(
                result.getIndex(),
                Status.Complete,
                result.getScore(),
                status.getCreatedTime(),
                null,       //TODO: how to know when execution actually started?
                currentTime);
        currentStatus.put(result.getIndex(), newStatus);

        //Listeners:
        for (OptimizationRunnerStatusListener listener : statusListeners) listener.onCompletion(result);

        //Report completion to candidate generator
        config.getCandidateGenerator().reportResults(result);

        Double score = result.getScore();
        log.info("Completed task {}, score = {}", result.getIndex(), result.getScore());

        //TODO handle minimization vs. maximization
        boolean minimize = config.getScoreFunction().minimize();
        if (score != null && (bestScore == null || ((minimize && score < bestScore) || (!minimize && score > bestScore)))) {
            if (bestScore == null) {
                log.info("New best score: {} (first completed model)", score);
            } else {
                log.info("New best score: {} (prev={})", score, bestScore);
            }
            bestScore = score;
            bestScoreTime = System.currentTimeMillis();
            bestScoreCandidateIndex = result.getIndex();
        }
        numCandidatesCompleted++;

        //TODO: In general, we don't want to save EVERY model, only the best ones
        ResultSaver<C, M, A> saver = config.getResultSaver();
        ResultReference<C, M, A> resultReference = null;
        if (saver != null) {
            try {
                resultReference = saver.saveModel(result);
            } catch (IOException e) {
                //TODO: Do we want ta warn or fail on IOException?
                log.warn("Error saving model (id={}): IOException thrown. ", result.getIndex(), e);
            }
        }

        if (resultReference != null) allResults.add(resultReference);
    }

    @Override
    public int numCandidatesTotal() {
        return totalCandidateCount;
    }

    @Override
    public int numCandidatesCompleted() {
        return numCandidatesCompleted;
    }

    @Override
    public int numCandidatesFailed() {
        return numCandidatesFailed;
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
        return bestScoreCandidateIndex;
    }

    @Override
    public List<ResultReference<C, M, A>> getResults() {
        return new ArrayList<>(allResults);
    }

    @Override
    public OptimizationConfiguration<C, M, ?, A> getConfiguration() {
        return config;
    }


    @Override
    public void addListeners(OptimizationRunnerStatusListener... listeners) {
        for (OptimizationRunnerStatusListener l : listeners) {
            if (!statusListeners.contains(l)) statusListeners.add(l);
        }
    }

    @Override
    public void removeListeners(OptimizationRunnerStatusListener... listeners) {
        for (OptimizationRunnerStatusListener l : listeners) {
            if (statusListeners.contains(l)) statusListeners.remove(l);
        }
    }

    @Override
    public void removeAllListeners() {
        statusListeners.clear();
    }

    @Override
    public List<CandidateStatus> getCandidateStatus() {
        List<CandidateStatus> list = new ArrayList<>();
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
        private final Future<OptimizationResult<C, M, A>> future;
        private final long startTime;
        private final int index;
    }

    @AllArgsConstructor
    private class OnCompletionListener implements Runnable {
        private Future<OptimizationResult<C, M, A>> future;

        @Override
        public void run() {
            completedFutures.add(future);
        }
    }


    protected abstract int maxConcurrentTasks();

    protected abstract ListenableFuture<OptimizationResult<C, M, A>> execute(Candidate<C> candidate, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction);

    protected abstract List<ListenableFuture<OptimizationResult<C, M, A>>> execute(List<Candidate<C>> candidates, DataProvider<D> dataProvider, ScoreFunction<M, D> scoreFunction);

    protected abstract void shutdown();
}
