package org.nd4j.linalg.dataset.curation.batching;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.function.ToIntFunction;

/**
 * Groups sequences by length into buckets to minimize padding waste.
 * Within each bucket, sequences are shuffled and batched together.
 *
 * @param <T> the type of data items
 */
public class LengthBucketingIterator<T> implements Iterator<List<T>>, Serializable {
    private static final long serialVersionUID = 1L;

    private final int[] bucketBoundaries;
    private final int tokenBudget;
    private final int fixedBatchSize;
    private final ToIntFunction<T> lengthFn;
    private final List<List<T>> buckets;
    private int currentBucket;
    private int posInBucket;

    private LengthBucketingIterator(Builder<T> builder, List<T> data) {
        this.bucketBoundaries = builder.bucketBoundaries;
        this.tokenBudget = builder.tokenBudget;
        this.fixedBatchSize = builder.fixedBatchSize;
        this.lengthFn = builder.lengthFn;

        // Initialize buckets
        this.buckets = new ArrayList<>();
        for (int i = 0; i <= bucketBoundaries.length; i++) {
            buckets.add(new ArrayList<>());
        }

        // Assign data to buckets
        for (T item : data) {
            int len = lengthFn.applyAsInt(item);
            int bucketIdx = findBucket(len);
            buckets.get(bucketIdx).add(item);
        }

        // Shuffle within buckets
        if (builder.seed != null) {
            Random rng = new Random(builder.seed);
            for (List<T> bucket : buckets) {
                Collections.shuffle(bucket, rng);
            }
        }

        this.currentBucket = 0;
        this.posInBucket = 0;
        advanceToNonEmptyBucket();
    }

    private int findBucket(int length) {
        for (int i = 0; i < bucketBoundaries.length; i++) {
            if (length <= bucketBoundaries[i]) return i;
        }
        return bucketBoundaries.length;
    }

    private void advanceToNonEmptyBucket() {
        while (currentBucket < buckets.size() && posInBucket >= buckets.get(currentBucket).size()) {
            currentBucket++;
            posInBucket = 0;
        }
    }

    @Override
    public boolean hasNext() {
        return currentBucket < buckets.size();
    }

    @Override
    public List<T> next() {
        if (!hasNext()) throw new NoSuchElementException();

        List<T> bucket = buckets.get(currentBucket);
        int batchSize;

        if (tokenBudget > 0) {
            // Dynamic batch size: token budget / max length in this bucket
            int maxLen = bucketBoundaries.length > 0 && currentBucket < bucketBoundaries.length
                ? bucketBoundaries[currentBucket]
                : (currentBucket > 0 ? bucketBoundaries[currentBucket - 1] * 2 : 512);
            batchSize = Math.max(1, tokenBudget / maxLen);
        } else {
            batchSize = fixedBatchSize;
        }

        int end = Math.min(posInBucket + batchSize, bucket.size());
        List<T> batch = new ArrayList<>(bucket.subList(posInBucket, end));
        posInBucket = end;
        advanceToNonEmptyBucket();
        return batch;
    }

    public static <T> Builder<T> builder() {
        return new Builder<>();
    }

    public static class Builder<T> {
        private int[] bucketBoundaries = {128, 256, 512, 1024, 2048};
        private int tokenBudget = 0;
        private int fixedBatchSize = 32;
        private ToIntFunction<T> lengthFn;
        private Long seed;

        public Builder<T> bucketBoundaries(int... boundaries) {
            this.bucketBoundaries = boundaries;
            return this;
        }

        public Builder<T> tokenBudget(int budget) {
            this.tokenBudget = budget;
            return this;
        }

        public Builder<T> fixedBatchSize(int size) {
            this.fixedBatchSize = size;
            return this;
        }

        public Builder<T> lengthFunction(ToIntFunction<T> fn) {
            this.lengthFn = fn;
            return this;
        }

        public Builder<T> shuffle(long seed) {
            this.seed = seed;
            return this;
        }

        public LengthBucketingIterator<T> build(List<T> data) {
            if (lengthFn == null) {
                throw new IllegalStateException("lengthFunction must be set");
            }
            return new LengthBucketingIterator<>(this, data);
        }
    }
}
