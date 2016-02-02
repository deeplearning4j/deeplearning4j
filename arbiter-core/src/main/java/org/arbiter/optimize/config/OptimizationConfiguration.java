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
public class OptimizationConfiguration<T,M,D,A> {

    private DataProvider<D> dataProvider;
    private CandidateGenerator<T> candidateGenerator;
    private ResultSaver<T,M,A> resultSaver;
    private ScoreFunction<M,D> scoreFunction;
    private List<TerminationCondition> terminationConditions;

    private OptimizationConfiguration(Builder<T,M,D,A> builder ){
        this.dataProvider = builder.dataProvider;
        this.candidateGenerator = builder.candidateGenerator;
        this.resultSaver = builder.resultSaver;
        this.scoreFunction = builder.scoreFunction;
        this.terminationConditions = builder.terminationConditions;
    }

    public static class Builder<T,M,D,A> {

        private DataProvider<D> dataProvider;
        private CandidateGenerator<T> candidateGenerator;
        private ResultSaver<T,M,A> resultSaver;
        private ScoreFunction<M,D> scoreFunction;
        private List<TerminationCondition> terminationConditions;

        public Builder<T,M,D,A> dataProvider(DataProvider<D> dataProvider){
            this.dataProvider = dataProvider;
            return this;
        }

        public Builder<T,M,D,A> candidateGenerator(CandidateGenerator<T> candidateGenerator){
            this.candidateGenerator = candidateGenerator;
            return this;
        }

        public Builder<T,M,D,A> modelSaver(ResultSaver<T,M,A> resultSaver){
            this.resultSaver = resultSaver;
            return this;
        }

        public Builder<T,M,D,A> scoreFunction(ScoreFunction<M,D> scoreFunction){
            this.scoreFunction = scoreFunction;
            return this;
        }

        public Builder<T,M,D,A> terminationConditions(TerminationCondition... conditions){
            terminationConditions = Arrays.asList(conditions);
            return this;
        }

        public Builder<T,M,D,A> terminationConditions(List<TerminationCondition> terminationConditions ){
            this.terminationConditions = terminationConditions;
            return this;
        }

        public OptimizationConfiguration<T,M,D,A> build(){
            return new OptimizationConfiguration<T,M,D,A>(this);
        }


    }

}
