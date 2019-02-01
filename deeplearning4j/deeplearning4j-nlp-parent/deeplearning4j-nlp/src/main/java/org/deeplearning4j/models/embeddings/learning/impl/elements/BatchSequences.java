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

    public void put(T word, T lastWord, AtomicLong randomValue, double alpha) {
        BatchItem<T> newItem = new BatchItem<>(word, lastWord, randomValue, alpha);
        buffer.add(newItem);
    }

    public List<BatchItem<T>> get() {
        List<BatchItem<T>> retVal = new ArrayList<>();

        for (int i = 0; i < batches; ++i) {
            try {
                BatchItem<T> value = buffer.get(i);
            } catch (IndexOutOfBoundsException e) {
                //log.error(e.getMessage());
                break;
            }
            retVal.add(buffer.get(i));
        }
        return retVal;
    }

    public int size() {
        return buffer.size();
    }
}
