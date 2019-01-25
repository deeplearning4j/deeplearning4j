package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import org.apache.commons.math3.random.RandomGenerator;

import java.util.*;

public class CrossoverPointsGenerator {
    private final int chromosomeLength;
    private final int minCrossovers;
    private final int maxCrossovers;
    private final RandomGenerator rng;
    private List<Integer> parameterIndexes;

    public CrossoverPointsGenerator(int chromosomeLength, int minCrossovers, int maxCrossovers, RandomGenerator rng) {
        this.chromosomeLength = chromosomeLength;
        this.minCrossovers = minCrossovers;
        this.maxCrossovers = maxCrossovers;
        this.rng = rng;
        parameterIndexes = new ArrayList<Integer>();
        for(int i = 0; i < chromosomeLength; ++i) {
            parameterIndexes.add(i);
        }
    }

    public Deque<Integer> getCrossoverPoints() {
        Collections.shuffle(parameterIndexes);
        List<Integer> crossoverPointLists =  parameterIndexes.subList(0, rng.nextInt(maxCrossovers - minCrossovers) + minCrossovers);
        Collections.sort(crossoverPointLists);
        Deque<Integer> crossoverPoints = new ArrayDeque<Integer>(crossoverPointLists);
        crossoverPoints.add(Integer.MAX_VALUE);

        return crossoverPoints;
    }
}