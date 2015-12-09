package org.arbiter.optimize.runner;

import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.executor.CandidateExecutor;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.*;


public class OptimizationRunner<T, M, D> {

    private OptimizationConfiguration<T, M, D> config;
    private CandidateExecutor<T, M, D> executor;
//    private List<Future<OptimizationResult<T, M>>> futures = new ArrayList<>();  //TODO: use threadsafe list?
    private BlockingQueue<Future<OptimizationResult<T, M>>> futures = new LinkedBlockingQueue<>();
    private int totalCount = 0;
    private final int MAX_CONCURRENT_JOBS = 5;      //PLACEHOLDER for


    public OptimizationRunner(OptimizationConfiguration<T, M, D> config, CandidateExecutor<T, M, D> executor) {
        this.config = config;
        this.executor = executor;
    }

    public void execute() {

        while(!terminate()){

            if(futures.size() >= MAX_CONCURRENT_JOBS ){

                //Wait on ANY of the futures to complete...
                //Design problem: How to best implement this?
                //Option 1: Queue + ListenableFuture (Guava)
                //      1a) Possibly utilizing JDKFutureAdaptor. But: that requires 1 thread per future :(
                //      1b) Change interface to return ListenableFuture
                //Option 2: polling approach (i.e., check + sleep, etc; very inelegant)

                //Bad solution that is good enough for now: just wait on first
                Future<OptimizationResult<T,M>> future;
                OptimizationResult<T,M> result;
                try{
                    future = futures.take();
                    result = future.get();
                } catch(InterruptedException | ExecutionException e ){
                    throw new RuntimeException(e);
                }

                System.out.println("RESULT COMPLETE");

            } else {

                //TODO how to determine number of concurrent jobs to pass to executor?
                //      -> Might be better to run N, wait for results, then generate new N based on those (i.e., for
                //          Bayesian optimization procedures) rather than 100 right up front...
                //TODO how to handle cancelling of jobs after time (etc) limit is exceeded?
                Candidate<T> candidate = config.getCandidateGenerator().getCandidate();
                Future<OptimizationResult<T, M>> future = executor.execute(candidate, config.getDataProvider());

                futures.add(future);
                totalCount++;
            }
        }

        //Wait on final tasks to complete
        while(futures.size() > 0){
            Future<OptimizationResult<T,M>> future;
            OptimizationResult<T,M> result;
            try{
                future = futures.take();
                result = future.get();
            } catch(InterruptedException | ExecutionException e ){
                throw new RuntimeException(e);
            }

            System.out.println("RESULT COMPLETE");
        }
    }

    private boolean terminate(){
        return totalCount >= 5;
    }

}
