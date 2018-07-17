package org.deeplearning4j.spark.impl.evaluation;

import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
public class EvaluationRunner {

    private static final EvaluationRunner INSTANCE = new EvaluationRunner();

    public static EvaluationRunner getInstance(){
        return INSTANCE;
    }



    private final AtomicBoolean isFirst = new AtomicBoolean(false);
    private ParallelInference pi = null;

    private Queue<Eval> queue = new ConcurrentLinkedQueue<>();

    private long lastModelBroadcastId = 0;  //Use ID of Broadcast<INDArray> to know when to update model

    private EvaluationRunner(){ }

    public Future<IEvaluation[]> execute(IEvaluation[] evals, int evalWorkers, Iterator<DataSet> ds, Iterator<MultiDataSet> mds,
                                        boolean isCG, Broadcast<String> json, Broadcast<INDArray> params, int evalBatchSize){

        if(isFirst.compareAndSet(false, true)){
            //Create PI instance
            if(pi == null || lastModelBroadcastId != params.id()){

                if(pi == null) {
                    log.info("Starting ParallelInference at thread {}", Thread.currentThread().getId());
                } else {
                    log.info("Re-creating ParallelInference at thread {} for new model", Thread.currentThread().getId());
                }

                Model m;
                if(isCG){
                    ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(json.getValue());
                    ComputationGraph cg = new ComputationGraph(conf);
                    cg.init(params.getValue(), false);
                    m = cg;
                } else {
                    MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(json.getValue());
                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init(params.getValue(), false);
                    m = net;
                }

                pi = new ParallelInference.Builder(m)
                        .inferenceMode(InferenceMode.BATCHED)
                        .workers(evalWorkers)
                        .build();
            }

            SimpleFuture f = new SimpleFuture();
            queue.add(new Eval(ds, mds, evals, f));

            //Perform evaluations:
            IEvaluation[] e = evals;
            while(!queue.isEmpty()){


                List<Eval> toProcess = new ArrayList<>();
                while(!queue.isEmpty()){
                    toProcess.add(queue.remove());
                }

                MultiDataSetIterator iter;
                if(toProcess.get(0).getDs() != null){
                    List<Iterator<DataSet>> list = new ArrayList<>();
                    for(Eval eval : toProcess){
                        list.add(eval.getDs());
                    }
                    Iterator<DataSet> merged = null;    //TODO new IteratorMerger<DataSet>(list);
                    iter = new AsyncMultiDataSetIterator(new MultiDataSetIteratorAdapter(new IteratorDataSetIterator(merged, evalBatchSize)));
                } else {
                    List<Iterator<MultiDataSet>> list = new ArrayList<>();
                    for(Eval eval : toProcess){
                        list.add(eval.getMds());
                    }
                    Iterator<MultiDataSet> merged = new IteratorMerger<>();
                    iter = new AsyncMultiDataSetIterator(new IteratorMultiDataSetIterator(merged, evalBatchSize));
                }

                while(iter.hasNext()){
                    MultiDataSet next = iter.next();

                    //Submit this (with feature masks) to PI

                    //TODO we sort of want to submit say 2 * nWorkers to PI, and then process them...
                    //Or better yet: add a ParallelInference.output(Iterable<Pair<INDArray[],INDArray[]>>) -> List<Future<INDArray[]>>
                    //Problem is we don't want to simply dump the contents of the iterator into the output methods,
                    // as the iterator could be lazy

                }


            }

            f.setCompleted(true);
            isFirst.set(false);
            return f;

        } else {
            SimpleFuture f = new SimpleFuture();
            queue.add(new Eval(ds, mds, evals, f));
            return f;
        }
    }



    @AllArgsConstructor
    @Data
    private static class Eval {
        private Iterator<DataSet> ds;
        private Iterator<MultiDataSet> mds;
        private IEvaluation[] evaluations;
        private SimpleFuture future;
    }

    @Setter
    private static class SimpleFuture implements Future<IEvaluation[]> {

        private boolean completed;
        private Evaluation[] result;
        private Exception exception;

        @Override
        public boolean cancel(boolean mayInterruptIfRunning) {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public boolean isCancelled() {
            return false;
        }

        @Override
        public boolean isDone() {
            return false;
        }

        @Override
        public IEvaluation[] get() throws InterruptedException, ExecutionException {
            while (!completed && exception == null){
                Thread.sleep(5);
            }
            if(exception != null){
                throw new ExecutionException(exception);
            }
            return result;
        }

        @Override
        public IEvaluation[] get(long timeout, @NotNull TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
            throw new UnsupportedOperationException();
        }
    }

    //TODO let's move this to ND4J common. And call it something better :)
    @AllArgsConstructor
    private static class IteratorMerger<T> implements Iterator<T> {

        private final Queue<Iterator<? extends T>> queue;
        private Iterator<? extends T> current;
        @Getter
        private int iteratorNumber;
        @Getter
        private int numItersCompleted;

        public IteratorMerger(Iterator<? extends T>... iters){
            queue = new LinkedList<>();
            queue.addAll(Arrays.asList(iters));
            iteratorNumber = -1;
            numItersCompleted = 0;
        }

        public IteratorMerger(Collection<Iterator<? extends T>> c){
            queue = new LinkedList<>(c);
            iteratorNumber = -1;
            numItersCompleted = 0;
        }

        @Override
        public boolean hasNext() {
            if((current != null && current.hasNext()) || (queue.size() > 0 && queue.peek().hasNext())){
                return true;
            }
            //Only way to get here: non-empty queue, but empty first iterator in queue
            while(!queue.isEmpty() && !queue.peek().hasNext()){
                queue.remove();
                iteratorNumber++;
                numItersCompleted++;
            }

            return !queue.isEmpty();
        }

        @Override
        public T next() {
            if(!hasNext())
                throw new NoSuchElementException();

            if(current == null) {
                current = queue.remove();
                iteratorNumber++;
            }
            if(!current.hasNext()){
                current = queue.remove();
                iteratorNumber++;
                numItersCompleted++;
            }

            T ret = current.next();

            if(!current.hasNext()){
                numItersCompleted++;
                current = null;
            }
            return ret;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported");
        }
    }
}
