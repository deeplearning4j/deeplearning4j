/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.arbiter.optimize.config;

import lombok.Data;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.arbiter.optimize.serde.jackson.YamlMapper;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.util.Arrays;
import java.util.List;

/**
 * OptimizationConfiguration ties together all of the various components (such as data, score functions, result saving etc)
 * required to execute hyperparameter optimization.
 *
 * @param <T> Type of candidates
 * @param <M> Type of model returned
 * @param <D> Type of data
 * @author Alex Black
 */
@Data
public class OptimizationConfiguration<T, M, D, A> {

    private DataProvider<D> dataProvider;
    private CandidateGenerator<T> candidateGenerator;
    private ResultSaver<T, M, A> resultSaver;
    private ScoreFunction<M, D> scoreFunction;
    private List<TerminationCondition> terminationConditions;
    private Long rngSeed;

    private OptimizationConfiguration(Builder<T, M, D, A> builder) {
        this.dataProvider = builder.dataProvider;
        this.candidateGenerator = builder.candidateGenerator;
        this.resultSaver = builder.resultSaver;
        this.scoreFunction = builder.scoreFunction;
        this.terminationConditions = builder.terminationConditions;
        this.rngSeed = builder.rngSeed;

        if (rngSeed != null) candidateGenerator.setRngSeed(rngSeed);
    }

    public static class Builder<T, M, D, A> {

        private DataProvider<D> dataProvider;
        private CandidateGenerator<T> candidateGenerator;
        private ResultSaver<T, M, A> resultSaver;
        private ScoreFunction<M, D> scoreFunction;
        private List<TerminationCondition> terminationConditions;
        private Long rngSeed;

        public Builder<T, M, D, A> dataProvider(DataProvider<D> dataProvider) {
            this.dataProvider = dataProvider;
            return this;
        }

        public Builder<T, M, D, A> candidateGenerator(CandidateGenerator<T> candidateGenerator) {
            this.candidateGenerator = candidateGenerator;
            return this;
        }

        public Builder<T, M, D, A> modelSaver(ResultSaver<T, M, A> resultSaver) {
            this.resultSaver = resultSaver;
            return this;
        }

        public Builder<T, M, D, A> scoreFunction(ScoreFunction<M, D> scoreFunction) {
            this.scoreFunction = scoreFunction;
            return this;
        }

        public Builder<T, M, D, A> terminationConditions(TerminationCondition... conditions) {
            terminationConditions = Arrays.asList(conditions);
            return this;
        }

        public Builder<T, M, D, A> terminationConditions(List<TerminationCondition> terminationConditions) {
            this.terminationConditions = terminationConditions;
            return this;
        }

        public Builder<T, M, D, A> rngSeed(long rngSeed) {
            this.rngSeed = rngSeed;
            return this;
        }

        public OptimizationConfiguration<T, M, D, A> build() {
            return new OptimizationConfiguration<T, M, D, A>(this);
        }
    }


    /**
     * Return a json configuration of this optimization configuration
     *
     * @return
     */
    public String toJson() {
        try {
            return JsonMapper.getMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Return a yaml configuration of this optimization configuration
     *
     * @return
     */
    public String toYaml() {
        try {
            return YamlMapper.getMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
}
