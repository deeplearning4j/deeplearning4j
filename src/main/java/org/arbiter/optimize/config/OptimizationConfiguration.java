package org.arbiter.optimize.config;

import lombok.Builder;
import lombok.Data;
import org.arbiter.optimize.api.CandidateGenerator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.saving.ModelSaver;
import org.arbiter.optimize.api.score.ScoreFunction;

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
    private ModelSaver<M> modelSaver;
    private ScoreFunction<M> scoreFunction;

    private OptimizationConfiguration(Builder<T,M,D> builder ){
        this.dataProvider = builder.dataProvider;
        this.candidateGenerator = builder.candidateGenerator;
        this.modelSaver = builder.modelSaver;
        this.scoreFunction = builder.scoreFunction;
    }

    public static class Builder<T,M,D> {

        private DataProvider<D> dataProvider;
        private CandidateGenerator<T> candidateGenerator;
        private ModelSaver<M> modelSaver;
        private ScoreFunction<M> scoreFunction;

        public Builder dataProvider(DataProvider<D> dataProvider){
            this.dataProvider = dataProvider;
            return this;
        }

        public Builder candidateGenerator(CandidateGenerator<T> candidateGenerator){
            this.candidateGenerator = candidateGenerator;
            return this;
        }

        public Builder modelSaver(ModelSaver<M> modelSaver){
            this.modelSaver = modelSaver;
            return this;
        }

        public Builder scoreFunction(ScoreFunction<M> scoreFunction){
            this.scoreFunction = scoreFunction;
            return this;
        }

        public OptimizationConfiguration<T,M,D> build(){
            return new OptimizationConfiguration<>(this);
        }


    }

}
