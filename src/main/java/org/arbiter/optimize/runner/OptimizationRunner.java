package org.arbiter.optimize.runner;

import com.google.common.util.concurrent.ListenableFuture;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.saving.ResultReference;
import org.arbiter.optimize.api.saving.ResultSaver;
import org.arbiter.optimize.api.termination.TerminationCondition;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.executor.CandidateExecutor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/** Optimization runner: responsible for scheduling tasks on executor, saving results, etc.
 *
 * @param <T> Type of configuration
 * @param <M> Type of model learned
 * @param <D> Type of data used to train model
 */
public class OptimizationRunner<T, M, D> implements IOptimizationRunner<T,M> {

    private static final int POLLING_FREQUENCY = 10;
    private static final TimeUnit POLLING_FREQUENCY_UNIT = TimeUnit.SECONDS;
    private static Logger log = LoggerFactory.getLogger(OptimizationRunner.class);

    private OptimizationConfiguration<T, M, D> config;
    private CandidateExecutor<T, M, D> executor;
    private Queue<Future<OptimizationResult<T,M>>> queuedFutures = new ConcurrentLinkedQueue<>();
    private BlockingQueue<Future<OptimizationResult<T,M>>> completedFutures = new LinkedBlockingQueue<>();
    private int totalCandidateCount = 0;
    private int numCandidatesCompleted = 0;
    private int numCandidatesFailed = 0;
    private double bestScore = Double.MAX_VALUE;
    private long bestScoreTime = 0;
    private List<ResultReference<T,M>> allResults = new ArrayList<>();

    private ExecutorService listenerExecutor;


    public OptimizationRunner(OptimizationConfiguration<T, M, D> config, CandidateExecutor<T, M, D> executor) {
        this.config = config;
        this.executor = executor;

        if(config.getTerminationConditions() == null || config.getTerminationConditions().size() == 0 ){
            throw new IllegalArgumentException("Cannot create OptimizationRunner without TerminationConditions ("+
                "termination conditions are null or empty)");
        }

        listenerExecutor = Executors.newFixedThreadPool(executor.maxConcurrentTasks(),
                new ThreadFactory() {
                    private AtomicLong counter = new AtomicLong(0);
                    @Override
                    public Thread newThread(Runnable r) {
                        Thread t = Executors.defaultThreadFactory().newThread(r);
                        t.setDaemon(true);
                        t.setName("ArbiterOptimizationRunner-"+counter.getAndIncrement());
                        return t;
                    }
                });
    }

    public void execute() {

        log.info("OptimizationRunner: execution started");

        //Initialize termination conditions (start timers, etc)
        for(TerminationCondition c : config.getTerminationConditions()){
            c.initialize(this);
        }

        //Queue initial tasks:


        List<Future<OptimizationResult<T,M>>> tempList = new ArrayList<>(100);
        while(true){


            //Otherwise: add tasks if required
            Future<OptimizationResult<T,M>> future = null;
            try{
                future = completedFutures.poll(POLLING_FREQUENCY, POLLING_FREQUENCY_UNIT);
            }catch(InterruptedException e){
                //No op?
            }
            if(future != null) tempList.add(future);
            completedFutures.drainTo(tempList);

            //Process results (if any)
            for(Future<OptimizationResult<T,M>> f : tempList){
                queuedFutures.remove(f);
                processReturnedTask(f);
            }
            tempList.clear();


            //Check termination conditions:
            if(terminate()){
                executor.shutdown();
                break;
            }

            //Add additional tasks
            while(queuedFutures.size() < executor.maxConcurrentTasks()){
                Candidate<T> candidate = config.getCandidateGenerator().getCandidate();
                ListenableFuture<OptimizationResult<T, M>> f = executor.execute(candidate, config.getDataProvider(), config.getScoreFunction());
                f.addListener(new OnCompletionListener(f),listenerExecutor);
                queuedFutures.add(f);
                totalCandidateCount++;
            }
        }

        //Process any final (completed) tasks:
        completedFutures.drainTo(tempList);
        for(Future<OptimizationResult<T,M>> f : tempList){
            queuedFutures.remove(f);
            processReturnedTask(f);
        }
        tempList.clear();

        log.info("Optimization runner: execution complete");
    }

    /** Process returned task (either completed or failed */
    private void processReturnedTask(Future<OptimizationResult<T,M>> future){

        //TODO: track and log execution time
        OptimizationResult<T,M> result;
        try{
            result = future.get(100,TimeUnit.MILLISECONDS);
        }catch(InterruptedException e ){
            throw new RuntimeException("Unexpected InterruptedException thrown for task",e);
        } catch( ExecutionException e ){
            log.warn("Task failed",e);

            numCandidatesFailed++;
            return;
        } catch (TimeoutException e) {
            throw new RuntimeException(e);  //TODO
        }

        double score = result.getScore();
        log.info("Completed task {}, score = {}",result.getIndex(),result.getScore());

        //TODO handle minimization vs. maximization
        if(score < bestScore){
            if(bestScore == Double.MAX_VALUE){
                log.info("New best score: {} (first completed model)", score);
            } else {
                log.info("New best score: {} (prev={})", score, bestScore);
            }
            bestScore = score;
        }
        numCandidatesCompleted++;

        //TODO: In general, we don't want to save EVERY model, only the best ones
        ResultSaver<T,M> saver = config.getResultSaver();
        ResultReference<T,M> resultReference = null;
        if(saver != null){
            try{
                resultReference = saver.saveModel(result);
            }catch(IOException e){
                //TODO: Do we want ta warn or fail on IOException?
                log.warn("Error saving model (id={}): IOException thrown. ",result.getIndex(),e);
            }
        }

        if(resultReference != null) allResults.add(resultReference);
    }

    @Override
    public int numCandidatesScheduled() {
        return totalCandidateCount;
    }

    @Override
    public int numCandidatesCompleted() {
        return numCandidatesCompleted;
    }

    @Override
    public int numCandidatesFailed(){
        return numCandidatesFailed;
    }

    @Override
    public double bestScore() {
        return bestScore;
    }

    @Override
    public long bestScoreTime() {
        return bestScoreTime;
    }

    @Override
    public List<ResultReference<T, M>> getResults() {
        return new ArrayList<>(allResults);
    }

    private boolean terminate(){
        for(TerminationCondition c : config.getTerminationConditions() ){
            if(c.terminate(this)){
                log.info("OptimizationRunner global termination condition hit: {}", c);
                return true;
            }
        }
        return false;
    }

    @AllArgsConstructor @Data
    private class FutureDetails {
        private final Future<OptimizationResult<T,M>> future;
        private final long startTime;
        private final int index;
    }

    @AllArgsConstructor
    private class OnCompletionListener implements Runnable {
        private Future<OptimizationResult<T,M>> future;

        @Override
        public void run() {
            completedFutures.add(future);
        }
    }

}
