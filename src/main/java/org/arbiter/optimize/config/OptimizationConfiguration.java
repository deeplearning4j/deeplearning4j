package org.arbiter.optimize.config;

import lombok.Data;
import org.arbiter.optimize.api.CandidateGenerator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.saving.ResultSaver;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.arbiter.optimize.api.termination.TerminationCondition;

import java.util.Arrays;
import java.util.List;

/**
 *
 * @param <T>    Type of candidates
 * @param <M>    Type of model returned
 * @param <D>    Type of data
 */
@Data
public class OptimizationConfiguration<T,M,D> {

    private DataProvider<D> dataProvider;
    private CandidateGenerator<T> candidateGenerator;
    private ResultSaver<T,M> resultSaver;
    private ScoreFunction<M,D> scoreFunction;
    private List<TerminationCondition> terminationConditions;

    private OptimizationConfiguration(Builder<T,M,D> builder ){
        this.dataProvider = builder.dataProvider;
        this.candidateGenerator = builder.candidateGenerator;
        this.resultSaver = builder.resultSaver;
        this.scoreFunction = builder.scoreFunction;
        this.terminationConditions = builder.terminationConditions;
    }

    public static class Builder<T,M,D> {

        private DataProvider<D> dataProvider;
        private CandidateGenerator<T> candidateGenerator;
        private ResultSaver<T,M> resultSaver;
        private ScoreFunction<M,D> scoreFunction;
        private List<TerminationCondition> terminationConditions;

        public Builder dataProvider(DataProvider<D> dataProvider){
            this.dataProvider = dataProvider;
            return this;
        }

        public Builder candidateGenerator(CandidateGenerator<T> candidateGenerator){
            this.candidateGenerator = candidateGenerator;
            return this;
        }

        public Builder modelSaver(ResultSaver<T,M> resultSaver){
            this.resultSaver = resultSaver;
            return this;
        }

        public Builder scoreFunction(ScoreFunction<M,D> scoreFunction){
            this.scoreFunction = scoreFunction;
            return this;
        }

        public Builder terminationConditions(TerminationCondition... conditions){
            terminationConditions = Arrays.asList(conditions);
            return this;
        }

        public Builder terminationConditions(List<TerminationCondition> terminationConditions ){
            this.terminationConditions = terminationConditions;
            return this;
        }

        public OptimizationConfiguration<T,M,D> build(){
            return new OptimizationConfiguration<T,M,D>(this);
        }


    }

}
