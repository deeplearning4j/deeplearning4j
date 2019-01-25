package org.deeplearning4j.arbiter.optimize.generator;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.ChromosomeFactory;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.SinglePointCrossover;
import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.MutationOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.mutation.RandomMutationOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.EmptyPopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationInitializer;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.GeneticSelectionOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.SelectionOperator;

import java.util.Map;

public class GeneticSearchCandidateGenerator extends BaseCandidateGenerator {



    public static class Builder {
        protected final ParameterSpace<?> parameterSpace;

        protected Map<String, Object> dataParameters;
        protected boolean initDone;
        protected boolean minimizeScore;
        protected PopulationModel populationModel;
        protected ChromosomeFactory chromosomeFactory;
        protected SelectionOperator selectionOperator;

        public Builder(ParameterSpace<?> parameterSpace, ScoreFunction scoreFunction) {
            this.parameterSpace = parameterSpace;
            this.minimizeScore = scoreFunction.minimize();
        }

        public GeneticSearchCandidateGenerator.Builder populationModel(PopulationModel populationModel) {
            this.populationModel = populationModel;
            return this;
        }

        public GeneticSearchCandidateGenerator.Builder selectionOperator(SelectionOperator selectionOperator) {
            this.selectionOperator = selectionOperator;
            return this;
        }

        public GeneticSearchCandidateGenerator.Builder dataParameters(Map<String, Object> dataParameters) {

            this.dataParameters = dataParameters;
            return this;
        }

        public GeneticSearchCandidateGenerator.Builder initDone(boolean initDone) {
            this.initDone = initDone;
            return this;
        }

        public GeneticSearchCandidateGenerator.Builder chromosomeFactory(ChromosomeFactory chromosomeFactory) {
            this.chromosomeFactory = chromosomeFactory;
            return this;
        }

        public GeneticSearchCandidateGenerator build() {
            if(populationModel == null) {
                PopulationInitializer defaultPopulationInitializer = new EmptyPopulationInitializer();
                populationModel = new PopulationModel.Builder(defaultPopulationInitializer).build();
            }

            if(chromosomeFactory == null) {
                chromosomeFactory = new ChromosomeFactory();
            }

            if(selectionOperator == null) {
                selectionOperator = new GeneticSelectionOperator.Builder().build();
            }

            return new GeneticSearchCandidateGenerator(this);
        }
    }

    public final PopulationModel populationModel;
    public final ChromosomeFactory chromosomeFactory;
    public final SelectionOperator selectionOperator;

    private GeneticSearchCandidateGenerator(Builder builder) {
        super(builder.parameterSpace, builder.dataParameters, builder.initDone);

        initialize();

        chromosomeFactory = builder.chromosomeFactory;
        populationModel = builder.populationModel;
        selectionOperator = builder.selectionOperator;

        chromosomeFactory.initializeInstance(builder.parameterSpace.numParameters());
        populationModel.initializeInstance(builder.minimizeScore);
        selectionOperator.initializeInstance(populationModel, chromosomeFactory);

    }

    @Override
    public boolean hasMoreCandidates() {
        return true;
    }

    @Override
    public Candidate getCandidate() {

        double[] values = selectionOperator.buildNextGenes();
        Object value = null;
        Exception e = null;
        try {
            value = parameterSpace.getValue(values);
        } catch (Exception e2) {
            e = e2;
        }

        return new Candidate(value, candidateCounter.getAndIncrement(), values, dataParameters, e);
    }

    @Override
    public Class<?> getCandidateType() {
        return null;
    }

    @Override
    public String toString() {
        return "GeneticSearchCandidateGenerator";
    }

    @Override
    public void reportResults(OptimizationResult result) {
        if(result.getScore() == null) {
            return;
        }

        Chromosome newChromosome = chromosomeFactory.createChromosome(result.getCandidate().getFlatParameters(), result.getScore());
        populationModel.add(newChromosome);
    }
}
