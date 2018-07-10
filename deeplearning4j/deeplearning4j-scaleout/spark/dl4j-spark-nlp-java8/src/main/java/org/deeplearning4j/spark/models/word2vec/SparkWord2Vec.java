/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.models.word2vec;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.sequencevectors.SparkSequenceVectors;
import org.deeplearning4j.spark.models.sequencevectors.export.SparkModelExporter;
import org.deeplearning4j.spark.models.sequencevectors.functions.TokenizerFunction;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkElementsLearningAlgorithm;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkSequenceLearningAlgorithm;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SparkWord2Vec extends SparkSequenceVectors<VocabWord> {

    protected SparkWord2Vec() {
        // FIXME: this is development-time constructor, please remove before release
        configuration = new VectorsConfiguration();
        configuration.setTokenizerFactory(DefaultTokenizerFactory.class.getCanonicalName());
    }

    public SparkWord2Vec(@NonNull VoidConfiguration psConfiguration, @NonNull VectorsConfiguration configuration) {
        this.configuration = configuration;
        this.paramServerConfiguration = psConfiguration;
    }

    @Override
    protected VocabCache<ShallowSequenceElement> getShallowVocabCache() {
        return super.getShallowVocabCache();
    }


    @Override
    protected void validateConfiguration() {
        super.validateConfiguration();

        if (configuration.getTokenizerFactory() == null)
            throw new DL4JInvalidConfigException("TokenizerFactory is undefined. Can't train Word2Vec without it.");
    }

    /**
     * PLEASE NOTE: This method isn't supported for Spark implementation. Consider using fitLists() or fitSequences() instead.
     */
    @Override
    @Deprecated
    public void fit() {
        throw new UnsupportedOperationException("To use fit() method, please consider using standalone implementation");
    }

    public void fitSentences(JavaRDD<String> sentences) {
        /**
         * Basically all we want here is tokenization, to get JavaRDD<Sequence<VocabWord>> out of Strings, and then we just go  for SeqVec
         */

        validateConfiguration();

        final JavaSparkContext context = new JavaSparkContext(sentences.context());

        broadcastEnvironment(context);

        JavaRDD<Sequence<VocabWord>> seqRdd = sentences.map(new TokenizerFunction(configurationBroadcast));

        // now since we have new rdd - just pass it to SeqVec
        super.fitSequences(seqRdd);
    }


    public static class Builder extends SparkSequenceVectors.Builder<VocabWord> {

        /**
         * This method should NOT be used in real world applications
         */
        @Deprecated
        public Builder() {
            super();
        }

        public Builder(@NonNull VoidConfiguration psConfiguration) {
            super(psConfiguration);
        }

        public Builder(@NonNull VoidConfiguration psConfiguration, @NonNull VectorsConfiguration configuration) {
            super(psConfiguration, configuration);
        }

        /**
         * This method defines tokenizer htat will be used for corpus tokenization
         *
         * @param tokenizerFactory
         * @return
         */
        public Builder setTokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            configuration.setTokenizerFactory(tokenizerFactory.getClass().getCanonicalName());
            if (tokenizerFactory.getTokenPreProcessor() != null)
                configuration.setTokenPreProcessor(
                                tokenizerFactory.getTokenPreProcessor().getClass().getCanonicalName());

            return this;
        }


        /**
         * This method defines the learning algorithm that will be used during training
         *
         * @param ela
         * @return
         */
        public Builder setLearningAlgorithm(@NonNull SparkElementsLearningAlgorithm ela) {
            this.configuration.setElementsLearningAlgorithm(ela.getClass().getCanonicalName());
            return this;
        }

        /**
         * This method defines the way model will be exported after training is finished
         *
         * @param exporter
         * @return
         */
        public Builder setModelExporter(@NonNull SparkModelExporter<VocabWord> exporter) {
            this.modelExporter = exporter;
            return this;
        }

        /**
         *
         *
         * @param numWorkers
         * @return
         */
        public Builder workers(int numWorkers) {
            super.workers(numWorkers);
            return this;
        }


        public Builder epochs(int numEpochs) {
            super.epochs(numEpochs);
            return this;
        }

        @Override
        public Builder setStorageLevel(StorageLevel level) {
            super.setStorageLevel(level);
            return this;
        }

        @Override
        public Builder minWordFrequency(int num) {
            super.minWordFrequency(num);
            return this;
        }

        @Override
        public Builder setLearningRate(double lr) {
            super.setLearningRate(lr);;
            return this;
        }

        @Override
        public Builder setParameterServerConfiguration(@NonNull VoidConfiguration configuration) {
            super.setParameterServerConfiguration(configuration);
            return this;
        }

        @Override
        public Builder iterations(int num) {
            super.iterations(num);
            return this;
        }

        @Override
        public Builder subsampling(double rate) {
            super.subsampling(rate);
            return this;
        }

        @Override
        public Builder negativeSampling(long samples) {
            super.negativeSampling(samples);
            return this;
        }

        @Override
        public Builder setElementsLearningAlgorithm(@NonNull SparkElementsLearningAlgorithm ela) {
            super.setElementsLearningAlgorithm(ela);
            return this;
        }

        @Override
        public Builder setSequenceLearningAlgorithm(@NonNull SparkSequenceLearningAlgorithm sla) {
            throw new UnsupportedOperationException("This method isn't supported by Word2Vec");
        }

        @Override
        public Builder useHierarchicSoftmax(boolean reallyUse) {
            super.useHierarchicSoftmax(reallyUse);
            return this;
        }

        @Override
        public Builder layerSize(int layerSize) {
            super.layerSize(layerSize);
            return this;
        }

        /**
         * This method returns you SparkWord2Vec instance ready for training
         *
         * @return
         */
        public SparkWord2Vec build() {
            SparkWord2Vec sw2v = new SparkWord2Vec(peersConfiguration, configuration);
            sw2v.exporter = this.modelExporter;
            sw2v.storageLevel = this.storageLevel;
            sw2v.workers = this.workers;

            return sw2v;
        }
    }
}
