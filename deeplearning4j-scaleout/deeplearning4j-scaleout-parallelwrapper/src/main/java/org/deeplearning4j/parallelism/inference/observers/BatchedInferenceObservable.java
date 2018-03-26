package org.deeplearning4j.parallelism.inference.observers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.parallelism.inference.InferenceObservable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This class holds reference input, and implements second use case: BATCHED inference
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class BatchedInferenceObservable extends BasicInferenceObservable implements InferenceObservable {
    private List<INDArray[]> inputs = new ArrayList<>();
    private List<INDArray[]> outputs = new ArrayList<>();
    private AtomicInteger counter = new AtomicInteger(0);
    private ThreadLocal<Integer> position = new ThreadLocal<>();

    private final Object locker = new Object();

    private ReentrantReadWriteLock realLocker = new ReentrantReadWriteLock();
    private AtomicBoolean isLocked = new AtomicBoolean(false);
    private AtomicBoolean isReadLocked = new AtomicBoolean(false);

    public BatchedInferenceObservable() {

    }

    @Override
    public void addInput(INDArray... input) {
        synchronized (locker) {
            inputs.add(input);
            position.set(counter.getAndIncrement());

            if (isReadLocked.get())
                realLocker.readLock().unlock();
        }
    }

    @Override
    public List<INDArray[]> getInputBatches() {
        realLocker.writeLock().lock();
        isLocked.set(true);

        // this method should pile individual examples into single batch

        if (counter.get() > 1) {

            int pos = 0;
            List<INDArray[]> out = new ArrayList<>();
            while(pos < inputs.size()) {
                INDArray[] result = new INDArray[inputs.get(0).length];

                //First: determine which we can actually batch...
                int lastPossible = pos;
                for (int i = pos+1; i < inputs.size(); i++) {
                    if (canBatch(inputs.get(pos), inputs.get(i))) {
                        lastPossible = i;
                    } else {
                        break;
                    }
                }

                for (int i = 0; i < result.length; i++) {
                    INDArray[] examples = new INDArray[lastPossible-pos+1];
                    int toStackPos = 0;
                    for (int e = pos; e <= lastPossible; e++) {
                        examples[toStackPos++] = inputs.get(e)[i];
                    }
//                    result[i] = Nd4j.pile(examples);
                    result[i] = Nd4j.concat(0, examples);
                }

                out.add(result);
                pos = lastPossible+1;
            }
            realLocker.writeLock().unlock();
            return out;
        } else {
            realLocker.writeLock().unlock();
            return Collections.singletonList(inputs.get(0));
        }
    }

    private static boolean canBatch(INDArray[] first, INDArray[] candidate){
        //For now: let's simply require that the inputs have the same shape
        //In the future: we'll intelligently handle the RNN variable length case
        for(int i=0; i<first.length; i++ ){
            if(!Arrays.equals(first[i].shape(), candidate[i].shape())){
                return false;
            }
        }
        return true;
    }

    @Override
    public void setOutputBatches(List<INDArray[]> output) {
        //this method should split batched output INDArray[] into multiple separate INDArrays
        // pre-create outputs
        int countAllBatches = 0;
        for (INDArray[] currBatchOutputs : output) {
            int exampleCount = currBatchOutputs[0].size(0);
            for (int i = 0; i < exampleCount; i++) {
                outputs.add(new INDArray[currBatchOutputs.length]);
            }

            // pull back results for individual examples
            for (int outputNumber = 0; outputNumber < currBatchOutputs.length; outputNumber++) {
                INDArray[] split = splitExamples(currBatchOutputs[outputNumber]);

                for (int exInBatch = 0; exInBatch < exampleCount; exInBatch++) {
                    outputs.get(countAllBatches++)[outputNumber] = split[exInBatch];
                }
            }
        }

        this.setChanged();
        notifyObservers();
    }

    private static INDArray[] splitExamples(INDArray input){
        //This isn't pretty, but Nd4j.tear() changes dimensionality...
        INDArrayIndex[] indices = new INDArrayIndex[input.rank()];
        for(int i=1; i<indices.length; i++ ){
            indices[i] = NDArrayIndex.all();
        }
        int nEx = input.size(0);
        INDArray[] out = new INDArray[nEx];
        for( int i=0; i<nEx; i++ ){
            indices[0] = NDArrayIndex.interval(i, i, true); //Can't use point, as it reduces # dimensions
            out[i] = input.get(indices);
        }
        return out;
    }

    /**
     * PLEASE NOTE: This method is for tests only
     *
     * @return
     */
    protected List<INDArray[]> getOutputs() {
        return outputs;
    }

    protected void setCounter(int value) {
        counter.set(value);
    }

    public void setPosition(int pos) {
        position.set(pos);
    }

    public int getCounter() {
        return counter.get();
    }



    public boolean isLocked() {
        boolean lck = !realLocker.readLock().tryLock();

        boolean result = lck || isLocked.get();

        if (!result)
            isReadLocked.set(true);

        return result;
    }


    @Override
    public INDArray[] getOutput() {
        // basically we should take care of splits here: each client should get its own part of output, wrt order number

        int pos = position.get();
        return outputs.get(position.get());
    }
}
