/*-
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
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.arbiter.optimize.serde.jackson.YamlMapper;
import org.nd4j.reflectionloader.JacksonReflectionLoader;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * OptimizationConfiguration ties together all of the various
 * components (such as data, score functions, result saving etc)
 * required to execute hyperparameter optimization.
 *
 * @param <T> Type of candidates usually a {@link DL4JConfiguration }
 * @param <M> Type of model returned usually a MultiLayerNetwork or ComputationGraph
 * @param <D> Type of data usually a DataSetIterator
 * @param <A> The Result class usually something like {@link org.deeplearning4j.eval.Evaluation}
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(exclude = {"dataProvider","terminationConditions","candidateGenerator","resultSaver"})
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "type")
public class OptimizationConfiguration<T, M, D, A> {
    @JsonSerialize
    private DataProvider<D> dataProvider;
    @JsonSerialize
    private CandidateGenerator<T> candidateGenerator;
    @JsonSerialize
    private ResultSaver<T, M, A> resultSaver;
    @JsonSerialize
    private ScoreFunction<M, D> scoreFunction;
    @JsonSerialize
    private List<TerminationCondition> terminationConditions;
    @JsonSerialize
    private Long rngSeed;
    private static ObjectMapper jsonMapper;
    private static ObjectMapper yamlMapper;

    static {
        // List<Class<?>> classes = Arrays.asList(DataProvider.class,CandidateGenerator.class,ResultSaver.class,ScoreFunction.class,TerminationCondition.class);
        jsonMapper = JacksonReflectionLoader.findTypesFor(new ArrayList<Class<?>>());
        yamlMapper = JacksonReflectionLoader.findTypesFor(new ArrayList<Class<?>>(),false);

    }


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

        /**
         * Termination conditions to use
         * @param conditions
         * @return
         */
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
            return new OptimizationConfiguration<>(this);
        }
    }


    /**
     * Create an optimization configuration from the json
     * @param json the json to create the config from
     * @param tCLazz the type of candidates class
     * @param mClazz the model return type class
     * @param dCLazz the type of data class
     * @param aClazz the result type
     *  For type definitions
     *  @see OptimizationConfiguration
     * @return
     */
    public static <T,M,D,A> OptimizationConfiguration<T,M,D,A> fromYaml(String json,Class<T> tCLazz,Class<M> mClazz,Class<D> dCLazz,Class<A> aClazz) {
        try {
            return jsonMapper.readValue(json,
                    jsonMapper.getTypeFactory()
                            .constructParametrizedType(OptimizationConfiguration.class,
                                    OptimizationConfiguration.class,tCLazz,mClazz,dCLazz,aClazz));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create an optimization configuration from the json
     * @param json the json to create the config from
     * @param tCLazz the type of candidates class
     * @param mClazz the model return type class
     * @param dCLazz the type of data class
     * @param aClazz the result type
     *  For type definitions
     *  @see OptimizationConfiguration
     * @return
     */
    public static <T,M,D,A> OptimizationConfiguration<T,M,D,A> fromJson(String json,Class<T> tCLazz,Class<M> mClazz,Class<D> dCLazz,Class<A> aClazz) {
        try {
            return jsonMapper.readValue(json,
                    jsonMapper.getTypeFactory()
                            .constructParametrizedType(OptimizationConfiguration.class,
                                    OptimizationConfiguration.class,tCLazz,mClazz,dCLazz,aClazz));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Return a json configuration of this optimization configuration
     *
     * @return
     */
    public String toJson() {
        try {
            return jsonMapper.writeValueAsString(this);
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
            return yamlMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
}
