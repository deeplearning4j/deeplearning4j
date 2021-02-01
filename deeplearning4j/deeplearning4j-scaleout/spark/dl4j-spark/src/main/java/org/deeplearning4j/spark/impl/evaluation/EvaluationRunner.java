/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark.impl.evaluation;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.io.ByteArrayInputStream;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Singleton evaluation hrunner class for performing evaluation on Spark.
 * Allows fewer evaluation networks (and hence memory/cache thrashing) than one network per spark thread
 *
 * @author Alex Black
 */
@Slf4j
public class EvaluationRunner {

    private static final EvaluationRunner INSTANCE = new EvaluationRunner();

    public static EvaluationRunner getInstance(){
        return INSTANCE;
    }

    private final AtomicInteger workerCount = new AtomicInteger(0);
    private Queue<Eval> queue = new ConcurrentLinkedQueue<>();
    //parameters map for device local parameters for a given broadcast
    //Note: byte[] doesn't override Object.equals hence this is effectively an *identity* weak hash map, which is what we want here
    //i.e., DeviceLocal<INDArray> can be GC'd once the Broadcast<byte[]> is no longer referenced anywhere
    //This approach relies on the fact that a single Broadcast object's *content* will be shared by all of Spark's threads,
    // even though the Broadcast object itself mayb not be
    //Also by storing params as a byte[] (i.e., in serialized form), we sidestep a lot of the thread locality issues
    private Map<byte[],DeviceLocalNDArray> paramsMap = new WeakHashMap<>();


    private EvaluationRunner(){ }

    /**
     * Evaluate the data using the specified evaluations
     * @param evals         Evaluations to perform
     * @param evalWorkers   Number of concurrent workers
     * @param evalBatchSize Evaluation batch size to use
     * @param ds            DataSet iterator
     * @param mds           MultiDataSet iterator
     * @param isCG          True if ComputationGraph, false otherwise
     * @param json          JSON for the network
     * @param params        Parameters for the network
     * @return Future for the results
     */
    public Future<IEvaluation[]> execute(IEvaluation[] evals, int evalWorkers, int evalBatchSize, Iterator<DataSet> ds, Iterator<MultiDataSet> mds,
                                         boolean isCG, Broadcast<String> json, Broadcast<byte[]> params){
        Preconditions.checkArgument(evalWorkers > 0, "Invalid number of evaluation workers: must be > 0. Got: %s", evalWorkers);
        Preconditions.checkState(ds != null || mds != null, "No data provided - both DataSet and MultiDataSet iterators were null");

        //For multi-GPU we'll use a round robbin approach for worker thread/GPU affinity
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        if(numDevices <= 0)
            numDevices = 1;

        //Create the device local params if required
        DeviceLocalNDArray deviceLocalParams;
        synchronized (this){
            if(!paramsMap.containsKey(params.getValue())){
                //Due to singleton pattern, this block should execute only once (first thread)
                //Initially put on device 0. For CPU, this means we only have a single copy of the params INDArray shared by
                // all threads, which is both safe and uses the least amount of memory
                //For CUDA, we can't share threads otherwise arrays will be continually relocated, causing a crash
                //Nd4j.getMemoryManager().releaseCurrentContext();
                //NativeOpsHolder.getInstance().getDeviceNativeOps().setDevice(0);
                //Nd4j.getAffinityManager().attachThreadToDevice(Thread.currentThread(), 0);
                byte[] pBytes = params.getValue();
                INDArray p;
                try{
                    p = Nd4j.read(new ByteArrayInputStream(pBytes));
                } catch (RuntimeException e){
                    throw new RuntimeException(e);  //Should never happen
                }
                DeviceLocalNDArray dlp = new DeviceLocalNDArray(p);
                paramsMap.put(params.getValue(), dlp);
                //log.info("paramsMap: size {}", paramsMap.size());
            }
            deviceLocalParams = paramsMap.get(params.getValue());
        }

        int currentWorkerCount;
        while((currentWorkerCount = workerCount.get()) < evalWorkers){
            //For load balancing: we're relying on the fact that threads are mapped to devices in a round-robbin approach
            // the first time they touch an INDArray. If we assume this method is called by new threads,
            // then the first N workers will be distributed evenly across available devices.

                if (workerCount.compareAndSet(currentWorkerCount, currentWorkerCount + 1)) {
                    log.debug("Starting evaluation in thread {}", Thread.currentThread().getId());
                    //This thread is now a worker
                    EvaluationFuture f = new EvaluationFuture();
                    f.setResult(evals);
                    try {
                        Model m;
                        if (isCG) {
                            ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(json.getValue());
                            ComputationGraph cg = new ComputationGraph(conf);
                            cg.init(deviceLocalParams.get(), false);
                            m = cg;
                        } else {
                            MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(json.getValue());
                            MultiLayerNetwork net = new MultiLayerNetwork(conf);
                            net.init(deviceLocalParams.get(), false);
                            m = net;
                        }

                        //Perform eval on this thread's data
                        try {
                            doEval(m, evals, ds, mds, evalBatchSize);
                        } catch (Throwable t) {
                            f.setException(t);
                        } finally {
                            f.getSemaphore().release(1);
                        }

                        //Perform eval on other thread's data
                        while (!queue.isEmpty()) {
                            Eval e = queue.poll();  //Use poll not remove to avoid race condition on last element
                            if (e == null)
                                continue;
                            try {
                                doEval(m, evals, e.getDs(), e.getMds(), evalBatchSize);
                            } catch (Throwable t) {
                                e.getFuture().setException(t);
                            } finally {
                                e.getFuture().getSemaphore().release(1);
                            }
                        }
                    } finally {
                        workerCount.decrementAndGet();
                        log.debug("Finished evaluation in thread {}", Thread.currentThread().getId());
                    }

                    Nd4j.getExecutioner().commit();
                    return f;
                }
        }

        //At this point: not a worker thread (otherwise, would have returned already)
        log.debug("Submitting evaluation from thread {} for processing in evaluation thread", Thread.currentThread().getId());
        EvaluationFuture f = new EvaluationFuture();
        queue.add(new Eval(ds, mds, evals, f));
        return f;
    }

    private static void doEval(Model m, IEvaluation[] e, Iterator<DataSet> ds, Iterator<MultiDataSet> mds, int evalBatchSize){
        if(m instanceof MultiLayerNetwork){
            MultiLayerNetwork mln = (MultiLayerNetwork)m;
            if(ds != null){
                mln.doEvaluation(new IteratorDataSetIterator(ds, evalBatchSize), e);
            } else {
                mln.doEvaluation(new IteratorMultiDataSetIterator(mds, evalBatchSize), e);
            }
        } else {
            ComputationGraph cg = (ComputationGraph)m;
            if(ds != null){
                cg.doEvaluation(new IteratorDataSetIterator(ds, evalBatchSize), e);
            } else {
                cg.doEvaluation(new IteratorMultiDataSetIterator(mds, evalBatchSize), e);
            }
        }
    }



    @AllArgsConstructor
    @Data
    private static class Eval {
        private Iterator<DataSet> ds;
        private Iterator<MultiDataSet> mds;
        private IEvaluation[] evaluations;
        private EvaluationFuture future;
    }

    @Setter
    @Getter
    private static class EvaluationFuture implements Future<IEvaluation[]> {

        private Semaphore semaphore = new Semaphore(0);
        private IEvaluation[] result;
        private Throwable exception;

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
            return semaphore.availablePermits() > 0;
        }

        @Override
        public IEvaluation[] get() throws InterruptedException, ExecutionException {
            if(result == null && exception == null)
                semaphore.acquire();    //Block until completion (or failure) is reported
            if(exception != null){
                throw new ExecutionException(exception);
            }
            return result;
        }

        @Override
        public IEvaluation[] get(long timeout, @NonNull TimeUnit unit) {
            throw new UnsupportedOperationException();
        }
    }
}
