package org.nd4j.linalg.dataset.curation.curriculum;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * Difficulty-ordered iteration for curriculum learning.
 * Supports easy-to-hard, hard-to-easy, and mixed ordering strategies.
 *
 * @param <T> the type of training examples
 */
public class CurriculumIterator<T> implements Iterator<T>, Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Ordering strategy for curriculum learning.
     */
    public enum Strategy {
        EASY_TO_HARD,
        HARD_TO_EASY,
        MIXED
    }

    /**
     * Pacing function that controls how quickly to ramp up difficulty.
     */
    public enum Pacing {
        LINEAR,
        EXPONENTIAL,
        STEP
    }

    private final List<T> orderedData;
    private int position;

    public CurriculumIterator(List<T> data, DifficultyScorer<T> scorer, Strategy strategy) {
        this(data, scorer, strategy, Pacing.LINEAR, 42L);
    }

    public CurriculumIterator(List<T> data, DifficultyScorer<T> scorer, Strategy strategy,
                               Pacing pacing, long seed) {
        // Score and sort
        List<ScoredItem<T>> scored = new ArrayList<>();
        for (T item : data) {
            scored.add(new ScoredItem<>(item, scorer.score(item)));
        }

        switch (strategy) {
            case EASY_TO_HARD:
                scored.sort(Comparator.comparingDouble(s -> s.score));
                break;
            case HARD_TO_EASY:
                scored.sort(Comparator.<ScoredItem<T>, Double>comparing(s -> s.score).reversed());
                break;
            case MIXED:
                // Interleave easy and hard: sort then interleave from both ends
                scored.sort(Comparator.comparingDouble(s -> s.score));
                List<ScoredItem<T>> interleaved = new ArrayList<>();
                int lo = 0, hi = scored.size() - 1;
                boolean pickLow = true;
                while (lo <= hi) {
                    if (pickLow) {
                        interleaved.add(scored.get(lo++));
                    } else {
                        interleaved.add(scored.get(hi--));
                    }
                    pickLow = !pickLow;
                }
                scored = interleaved;
                break;
        }

        // Apply pacing (reorder based on pacing function)
        if (pacing == Pacing.STEP && strategy != Strategy.MIXED) {
            // Step: divide into thirds and shuffle within each third
            Random rng = new Random(seed);
            int third = scored.size() / 3;
            if (third > 0) {
                List<ScoredItem<T>> first = new ArrayList<>(scored.subList(0, third));
                List<ScoredItem<T>> second = new ArrayList<>(scored.subList(third, 2 * third));
                List<ScoredItem<T>> last = new ArrayList<>(scored.subList(2 * third, scored.size()));
                Collections.shuffle(first, rng);
                Collections.shuffle(second, rng);
                Collections.shuffle(last, rng);
                scored = new ArrayList<>();
                scored.addAll(first);
                scored.addAll(second);
                scored.addAll(last);
            }
        }

        this.orderedData = new ArrayList<>();
        for (ScoredItem<T> si : scored) {
            orderedData.add(si.item);
        }
        this.position = 0;
    }

    @Override
    public boolean hasNext() {
        return position < orderedData.size();
    }

    @Override
    public T next() {
        if (!hasNext()) throw new NoSuchElementException();
        return orderedData.get(position++);
    }

    public List<T> getOrderedData() {
        return Collections.unmodifiableList(orderedData);
    }

    public void reset() {
        position = 0;
    }

    /**
     * Built-in scorer: score by string length.
     */
    public static DifficultyScorer<String> lengthScorer() {
        return String::length;
    }

    private static class ScoredItem<T> {
        final T item;
        final double score;

        ScoredItem(T item, double score) {
            this.item = item;
            this.score = score;
        }
    }
}
