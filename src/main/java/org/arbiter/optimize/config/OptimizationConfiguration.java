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
@Builder @Data
public class OptimizationConfiguration<T,M,D> {

    private DataProvider<D> dataProvider;
    private CandidateGenerator<T> candidateGenerator;
    private ModelSaver<M> modelSaver;
    private ScoreFunction<M> scoreFunction;



}
