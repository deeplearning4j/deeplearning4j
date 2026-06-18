package org.nd4j.linalg.dataset.curation.mixing;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * Weighted sampling across multiple data sources for domain mixing.
 * Samples proportionally by weight with optional temperature scaling.
 *
 * @param <T> the type of data items
 */
public class WeightedDataMixer<T> implements Iterator<T>, Serializable {
    private static final long serialVersionUID = 1L;

    private final List<String> sourceNames;
    private final List<Iterator<T>> iterators;
    private final double[] normalizedWeights;
    private final Map<String, Long> consumptionCounts;
    private final Random rng;

    public WeightedDataMixer(Map<String, WeightedSource<T>> sources, double temperature, long seed) {
        this.sourceNames = new ArrayList<>(sources.keySet());
        this.iterators = new ArrayList<>();
        this.consumptionCounts = new HashMap<>();
        this.rng = new Random(seed);

        double[] rawWeights = new double[sourceNames.size()];
        for (int i = 0; i < sourceNames.size(); i++) {
            String name = sourceNames.get(i);
            WeightedSource<T> ws = sources.get(name);
            iterators.add(ws.iterator);
            rawWeights[i] = ws.weight;
            consumptionCounts.put(name, 0L);
        }

        // Apply temperature scaling
        if (temperature != 1.0) {
            for (int i = 0; i < rawWeights.length; i++) {
                rawWeights[i] = Math.pow(rawWeights[i], 1.0 / temperature);
            }
        }

        // Normalize
        double sum = 0;
        for (double w : rawWeights) sum += w;
        this.normalizedWeights = new double[rawWeights.length];
        for (int i = 0; i < rawWeights.length; i++) {
            normalizedWeights[i] = rawWeights[i] / sum;
        }
    }

    public WeightedDataMixer(Map<String, WeightedSource<T>> sources) {
        this(sources, 1.0, 42L);
    }

    @Override
    public boolean hasNext() {
        for (Iterator<T> it : iterators) {
            if (it.hasNext()) return true;
        }
        return false;
    }

    @Override
    public T next() {
        if (!hasNext()) throw new NoSuchElementException();

        // Build available weights for sources that still have data
        double[] availWeights = new double[normalizedWeights.length];
        double totalAvail = 0;
        for (int i = 0; i < iterators.size(); i++) {
            if (iterators.get(i).hasNext()) {
                availWeights[i] = normalizedWeights[i];
                totalAvail += normalizedWeights[i];
            }
        }

        // Multinomial sampling
        double r = rng.nextDouble() * totalAvail;
        double cumulative = 0;
        int selected = -1;
        for (int i = 0; i < availWeights.length; i++) {
            if (availWeights[i] <= 0) continue;
            cumulative += availWeights[i];
            if (r <= cumulative) {
                selected = i;
                break;
            }
        }
        // Fallback to last available source (handles floating point edge case)
        if (selected < 0) {
            for (int i = availWeights.length - 1; i >= 0; i--) {
                if (availWeights[i] > 0) {
                    selected = i;
                    break;
                }
            }
        }

        String sourceName = sourceNames.get(selected);
        consumptionCounts.merge(sourceName, 1L, Long::sum);
        return iterators.get(selected).next();
    }

    public Map<String, Long> getStats() {
        return new HashMap<>(consumptionCounts);
    }

    /**
     * A data source with an associated weight.
     */
    public static class WeightedSource<T> implements Serializable {
        private static final long serialVersionUID = 1L;
        public final Iterator<T> iterator;
        public final double weight;

        public WeightedSource(Iterator<T> iterator, double weight) {
            this.iterator = iterator;
            this.weight = weight;
        }
    }
}
