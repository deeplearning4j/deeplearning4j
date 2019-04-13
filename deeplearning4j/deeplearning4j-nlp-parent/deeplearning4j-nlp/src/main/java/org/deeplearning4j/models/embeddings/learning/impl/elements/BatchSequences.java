package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class BatchSequences<T extends SequenceElement> {

    private int batches;

    List<BatchItem<T>> buffer = new ArrayList<>();

    public BatchSequences(int batches) {
        this.batches = batches;
    }

    public void put(T word, T lastWord, long randomValue, double alpha) {
        BatchItem<T> newItem = new BatchItem<>(word, lastWord, randomValue, alpha);
        buffer.add(newItem);
    }

    public void put(T word, int[] windowWords, boolean[] wordStatuses, long randomValue, double alpha) {
        BatchItem<T> newItem = new BatchItem<>(word, windowWords, wordStatuses, randomValue, alpha);
        buffer.add(newItem);
    }

    public void put(T word, int[] windowWords, boolean[] wordStatuses, long randomValue, double alpha, int numLabels) {
        BatchItem<T> newItem = new BatchItem<>(word, windowWords, wordStatuses, randomValue, alpha, numLabels);
        buffer.add(newItem);
    }

    public List<BatchItem<T>> get(int chunkNo) {
        List<BatchItem<T>> retVal = new ArrayList<>();

        for (int i = 0 + chunkNo * batches; (i < batches + chunkNo * batches) && (i < buffer.size()); ++i) {
            BatchItem<T> value = buffer.get(i);
            retVal.add(value);
        }
        return retVal;
    }

    public int size() {
        return buffer.size();
    }

    public void clear() {
        buffer.clear();
    }


}
