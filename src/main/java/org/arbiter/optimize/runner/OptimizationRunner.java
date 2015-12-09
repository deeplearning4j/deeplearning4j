package org.arbiter.optimize.runner;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
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

/** Optimization runner: responsible for scheduling tasks on executor, saving results, etc.
 *
 * @param <T> Type of configuration
 * @param <M> Type of model learned
 * @param <D> Type of data used to train model
 */
public class OptimizationRunner<T, M, D> implements IOptimizationRunner<T,M> {

    private static Logger log = LoggerFactory.getLogger(OptimizationRunner.class);

    private OptimizationConfiguration<T, M, D> config;
    private CandidateExecutor<T, M, D> executor;
    private Queue<FutureDetails> futures = new ConcurrentLinkedQueue<>();
    private int totalCandidateCount = 0;
    private int numCandidatesCompleted = 0;
    private int numCandidatesFailed = 0;
    private double bestScore = Double.MAX_VALUE;
    private long bestScoreTime = 0;
    //TODO: Work out a proper way to handle results. This is basically a memory leak right here -> can't be used in general
    //Better approach: store a _reference_ to the results, which might be on disk, in database, or (rarely) in memory, etc
    // with a method to get the result from whetever it is stored
    private List<OptimizationResult<T,M>> allResults = new ArrayList<>();


    public OptimizationRunner(OptimizationConfiguration<T, M, D> config, CandidateExecutor<T, M, D> executor) {
        this.config = config;
        this.executor = executor;

        if(config.getTerminationConditions() == null || config.getTerminationConditions().size() == 0 ){
            throw new IllegalArgumentException("Cannot create OptimizationRunner without TerminationConditions ("+
                "termination conditions are null or empty)");
        }

        if(config.getMaxConcurrentJobs() <= 0){
            throw new IllegalArgumentException("Invalid setting for maxConcurrentJobs (" + config.getMaxConcurrentJobs() + " <= 0)");
        }
    }

    public void execute() {

        log.info("OptimizationRunner: execution started");

        //Initialize termination conditions (start timers, etc)
        for(TerminationCondition c : config.getTerminationConditions()){
            c.initialize(this);
        }

        while(!terminate()){

            if(futures.size() >= config.getMaxConcurrentJobs() ){

                //Wait on ANY of the current futures to complete...
                //Design problem: How to best implement this?
                //Option 1: Queue + ListenableFuture (Guava)
                //      1a) Possibly utilizing JDKFutureAdaptor. But: that requires 1 thread per future :(
                //      1b) Change interface to return ListenableFuture
                //Option 2: polling approach (i.e., check + sleep, etc; very inelegant)

                //Bad solution that is good enough for now: just wait on first

                FutureDetails futureDetails = futures.remove();

                try{
                    futureDetails.getFuture().get();
                    processReturnedTask(futureDetails); //Process on success
                }catch(InterruptedException | ExecutionException e){
                    processReturnedTask(futureDetails); //Process on failure
                }
            }

            //Schedule new tasks
            if(futures.size() < config.getMaxConcurrentJobs() ){

                //TODO how to determine number of concurrent jobs to pass to executor?
                //      -> Might be better to run N, wait for results, then generate new N based on those (i.e., for
                //          Bayesian optimization procedures) rather than 100 right up front...
                //TODO how to handle cancelling of jobs after time (etc) limit is exceeded?
                Candidate<T> candidate = config.getCandidateGenerator().getCandidate();
                Future<OptimizationResult<T, M>> future = executor.execute(candidate, config.getDataProvider(), config.getScoreFunction());

                FutureDetails fd = new FutureDetails(future,System.currentTimeMillis(),candidate.getIndex());
                futures.add(fd);

                log.info("Scheduled new candidate (id={}) for execution", fd.getIndex());
                totalCandidateCount++;
            }
        }

        //Wait on final tasks to complete (TODO: specify how long to wait + cancel if exceeds this?)
        while(futures.size() > 0){
            FutureDetails futureDetails = futures.remove();

            try{
                futureDetails.getFuture().get();
                processReturnedTask(futureDetails); //Process on success
            }catch(InterruptedException | ExecutionException e){
                processReturnedTask(futureDetails); //Process on failure
            }
        }

        log.info("Optimization runner: execution complete");
    }

    /** Process returned task (either completed or failed */
    private void processReturnedTask(FutureDetails completed){

        long execDuration = (System.currentTimeMillis() - completed.getStartTime())/1000;
        OptimizationResult<T,M> result;
        try{
            result = completed.getFuture().get();
        }catch(InterruptedException e ){
            throw new RuntimeException("Unexpected InterruptedException thrown for task " + completed.getIndex() + " initialized at time "
                    + completed.getStartTime(),e);
        } catch( ExecutionException e ){
            log.warn("Task {} failed after {} seconds with ExecutionException: ",completed.getIndex(),execDuration,e);

            numCandidatesFailed++;
            return;
        }

        double score = result.getScore();
        log.info("Completed task {}, score = {}, executionTime = {} seconds",completed.getIndex(),result.getScore(), execDuration);

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

        //TODO: In general, we don't want to save EVERY model, only the best ones. How to implement this?
        ResultSaver<T,M> saver = config.getResultSaver();
        if(saver != null){
            try{
                saver.saveModel(result);
            }catch(IOException e){
                //TODO: Do we want ta warn or fail on IOException?
                log.warn("Error saving model (id={}): IOException thrown. ",result.getIndex(),e);
            }
        }

        allResults.add(result);
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
    public List<OptimizationResult<T, M>> getResults() {
        return new ArrayList<>(allResults);
    }

    private boolean terminate(){
        for(TerminationCondition c : config.getTerminationConditions() ){
            if(c.terminate(this)){
                log.info("Termination condition hit: {}", c);
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

}
