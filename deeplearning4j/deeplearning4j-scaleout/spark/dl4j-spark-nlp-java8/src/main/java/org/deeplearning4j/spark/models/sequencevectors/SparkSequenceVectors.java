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

package org.deeplearning4j.spark.models.sequencevectors;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.spark.models.sequencevectors.export.ExportContainer;
import org.deeplearning4j.spark.models.sequencevectors.export.SparkModelExporter;
import org.deeplearning4j.spark.models.sequencevectors.functions.*;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkElementsLearningAlgorithm;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkSequenceLearningAlgorithm;
import org.deeplearning4j.spark.models.sequencevectors.primitives.ExtraCounter;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.FaultToleranceStrategy;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;
import org.nd4j.parameterserver.distributed.util.NetworkInformation;
import org.nd4j.parameterserver.distributed.util.NetworkOrganizer;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Generic SequenceVectors implementation for dl4j-spark-nlp
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SparkSequenceVectors<T extends SequenceElement> extends SequenceVectors<T> {
    protected Accumulator<Counter<Long>> elementsFreqAccum;
    protected Accumulator<ExtraCounter<Long>> elementsFreqAccumExtra;
    protected StorageLevel storageLevel = StorageLevel.MEMORY_ONLY();


    // FIXME: we probably do not need this at all
    protected Broadcast<VocabCache<T>> vocabCacheBroadcast;

    protected Broadcast<VocabCache<ShallowSequenceElement>> shallowVocabCacheBroadcast;
    protected Broadcast<VectorsConfiguration> configurationBroadcast;

    protected transient boolean isEnvironmentReady = false;
    protected transient VocabCache<ShallowSequenceElement> shallowVocabCache;
    protected boolean isAutoDiscoveryMode = true;

    protected SparkModelExporter<T> exporter;

    protected SparkElementsLearningAlgorithm ela;
    protected SparkSequenceLearningAlgorithm sla;

    protected VoidConfiguration paramServerConfiguration;

    protected SparkSequenceVectors() {
        this(new VectorsConfiguration());
    }

    protected SparkSequenceVectors(@NonNull VectorsConfiguration configuration) {
        this.configuration = configuration;
    }

    protected VocabCache<ShallowSequenceElement> getShallowVocabCache() {
        return shallowVocabCache;
    }


    /**
     * PLEASE NOTE: This method isn't supported for Spark implementation. Consider using fitLists() or fitSequences() instead.
     */
    @Override
    @Deprecated
    public void fit() {
        throw new UnsupportedOperationException("To use fit() method, please consider using standalone implementation");
    }

    protected void validateConfiguration() {
        if (!configuration.isUseHierarchicSoftmax() && configuration.getNegative() == 0)
            throw new DL4JInvalidConfigException(
                            "Both HierarchicSoftmax and NegativeSampling are disabled. Nothing to learn here.");

        if (configuration.getElementsLearningAlgorithm() == null
                        && configuration.getSequenceLearningAlgorithm() == null)
            throw new DL4JInvalidConfigException("No LearningAlgorithm was set. Nothing to learn here.");

        if (exporter == null)
            throw new DL4JInvalidConfigException(
                            "SparkModelExporter is undefined. No sense for training, if model won't be exported.");
    }

    protected void broadcastEnvironment(JavaSparkContext context) {
        if (!isEnvironmentReady) {
            configurationBroadcast = context.broadcast(configuration);

            isEnvironmentReady = true;
        }
    }

    /**
     * Utility method. fitSequences() used within.
     *
     * PLEASE NOTE: This method can't be used to train for labels, since List<T> can't hold labels. If you need labels - consider manual Sequence creation instead.
     *
     * @param corpus
     */
    public void fitLists(JavaRDD<List<T>> corpus) {
        // we just convert List to sequences
        JavaRDD<Sequence<T>> rdd = corpus.map(new ListSequenceConvertFunction<T>());

        // and use fitSequences()
        fitSequences(rdd);
    }

    /**
     * Base training entry point
     *
     * @param corpus
     */
    public void fitSequences(JavaRDD<Sequence<T>> corpus) {
        /**
         * Basically all we want for base implementation here is 3 things:
         * a) build vocabulary
         * b) build huffman tree
         * c) do training
         *
         * in this case all classes extending SeqVec, like deepwalk or word2vec will be just building their RDD<Sequence<T>>,
         * and calling this method for training, instead implementing own routines
         */

        validateConfiguration();

        if (ela == null) {
            try {
                ela = (SparkElementsLearningAlgorithm) Class.forName(configuration.getElementsLearningAlgorithm())
                                .newInstance();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }


        if (workers > 1) {
            log.info("Repartitioning corpus to {} parts...", workers);
            corpus.repartition(workers);
        }

        if (storageLevel != null)
            corpus.persist(storageLevel);

        final JavaSparkContext sc = new JavaSparkContext(corpus.context());

        // this will have any effect only if wasn't called before, in extension classes
        broadcastEnvironment(sc);

        Counter<Long> finalCounter;
        long numberOfSequences = 0;

        /**
         * Here we s
         */
        if (paramServerConfiguration == null)
            paramServerConfiguration = VoidConfiguration.builder().faultToleranceStrategy(FaultToleranceStrategy.NONE)
                            .numberOfShards(2).unicastPort(40123).multicastPort(40124).build();

        isAutoDiscoveryMode = paramServerConfiguration.getShardAddresses() != null
                        && !paramServerConfiguration.getShardAddresses().isEmpty() ? false : true;

        Broadcast<VoidConfiguration> paramServerConfigurationBroadcast = null;

        if (isAutoDiscoveryMode) {
            log.info("Trying auto discovery mode...");

            elementsFreqAccumExtra = corpus.context().accumulator(new ExtraCounter<Long>(),
                            new ExtraElementsFrequenciesAccumulator());

            ExtraCountFunction<T> elementsCounter =
                            new ExtraCountFunction<>(elementsFreqAccumExtra, configuration.isTrainSequenceVectors());

            JavaRDD<Pair<Sequence<T>, Long>> countedCorpus = corpus.map(elementsCounter);

            // just to trigger map function, since we need huffman tree before proceeding
            numberOfSequences = countedCorpus.count();

            finalCounter = elementsFreqAccumExtra.value();

            ExtraCounter<Long> spareReference = (ExtraCounter<Long>) finalCounter;

            // getting list of available hosts
            Set<NetworkInformation> availableHosts = spareReference.getNetworkInformation();

            log.info("availableHosts: {}", availableHosts);
            if (availableHosts.size() > 1) {
                // now we have to pick N shards and optionally N backup nodes, and pass them within configuration bean
                NetworkOrganizer organizer =
                                new NetworkOrganizer(availableHosts, paramServerConfiguration.getNetworkMask());

                paramServerConfiguration
                                .setShardAddresses(organizer.getSubset(paramServerConfiguration.getNumberOfShards()));

                // backup shards are optional
                if (paramServerConfiguration.getFaultToleranceStrategy() != FaultToleranceStrategy.NONE) {
                    paramServerConfiguration.setBackupAddresses(
                                    organizer.getSubset(paramServerConfiguration.getNumberOfShards(),
                                                    paramServerConfiguration.getShardAddresses()));
                }
            } else {
                // for single host (aka driver-only, aka spark-local) just run on loopback interface
                paramServerConfiguration.setShardAddresses(
                                Arrays.asList("127.0.0.1:" + paramServerConfiguration.getUnicastPort()));
                paramServerConfiguration.setFaultToleranceStrategy(FaultToleranceStrategy.NONE);
            }



            log.info("Got Shards so far: {}", paramServerConfiguration.getShardAddresses());

            // update ps configuration with real values where required
            paramServerConfiguration.setNumberOfShards(paramServerConfiguration.getShardAddresses().size());
            paramServerConfiguration.setUseHS(configuration.isUseHierarchicSoftmax());
            paramServerConfiguration.setUseNS(configuration.getNegative() > 0);

            paramServerConfigurationBroadcast = sc.broadcast(paramServerConfiguration);

        } else {

            // update ps configuration with real values where required
            paramServerConfiguration.setNumberOfShards(paramServerConfiguration.getShardAddresses().size());
            paramServerConfiguration.setUseHS(configuration.isUseHierarchicSoftmax());
            paramServerConfiguration.setUseNS(configuration.getNegative() > 0);

            paramServerConfigurationBroadcast = sc.broadcast(paramServerConfiguration);


            // set up freqs accumulator
            elementsFreqAccum = corpus.context().accumulator(new Counter<Long>(), new ElementsFrequenciesAccumulator());
            CountFunction<T> elementsCounter =
                            new CountFunction<>(configurationBroadcast, paramServerConfigurationBroadcast,
                                            elementsFreqAccum, configuration.isTrainSequenceVectors());

            // count all sequence elements and their sum
            JavaRDD<Pair<Sequence<T>, Long>> countedCorpus = corpus.map(elementsCounter);

            // just to trigger map function, since we need huffman tree before proceeding
            numberOfSequences = countedCorpus.count();

            // now we grab counter, which contains frequencies for all SequenceElements in corpus
            finalCounter = elementsFreqAccum.value();
        }

        long numberOfElements = (long) finalCounter.totalCount();

        long numberOfUniqueElements = finalCounter.size();

        log.info("Total number of sequences: {}; Total number of elements entries: {}; Total number of unique elements: {}",
                        numberOfSequences, numberOfElements, numberOfUniqueElements);

        /*
         build RDD of reduced SequenceElements, just get rid of labels temporary, stick to some numerical values,
         like index or hashcode. So we could reduce driver memory footprint
         */


        // build huffman tree, and update original RDD with huffman encoding info
        shallowVocabCache = buildShallowVocabCache(finalCounter);
        shallowVocabCacheBroadcast = sc.broadcast(shallowVocabCache);

        // FIXME: probably we need to reconsider this approach
        JavaRDD<T> vocabRDD = corpus
                        .flatMap(new VocabRddFunctionFlat<T>(configurationBroadcast, paramServerConfigurationBroadcast))
                        .distinct();
        vocabRDD.count();

        /**
         * now we initialize Shards with values. That call should be started from driver which is either Client or Shard in standalone mode.
         */
        VoidParameterServer.getInstance().init(paramServerConfiguration, new RoutedTransport(),
                        ela.getTrainingDriver());
        VoidParameterServer.getInstance().initializeSeqVec(configuration.getLayersSize(), (int) numberOfUniqueElements,
                        119, configuration.getLayersSize() / paramServerConfiguration.getNumberOfShards(),
                        paramServerConfiguration.isUseHS(), paramServerConfiguration.isUseNS());

        // proceed to training
        // also, training function is the place where we invoke ParameterServer
        TrainingFunction<T> trainer = new TrainingFunction<>(shallowVocabCacheBroadcast, configurationBroadcast,
                        paramServerConfigurationBroadcast);
        PartitionTrainingFunction<T> partitionTrainer = new PartitionTrainingFunction<>(shallowVocabCacheBroadcast,
                        configurationBroadcast, paramServerConfigurationBroadcast);

        if (configuration != null)
            for (int e = 0; e < configuration.getEpochs(); e++)
                corpus.foreachPartition(partitionTrainer);
        //corpus.foreach(trainer);


        // we're transferring vectors to ExportContainer
        JavaRDD<ExportContainer<T>> exportRdd =
                        vocabRDD.map(new DistributedFunction<T>(paramServerConfigurationBroadcast,
                                        configurationBroadcast, shallowVocabCacheBroadcast));

        // at this particular moment training should be pretty much done, and we're good to go for export
        if (exporter != null)
            exporter.export(exportRdd);

        // unpersist, if we've persisten corpus after all
        if (storageLevel != null)
            corpus.unpersist();

        log.info("Training finish, starting cleanup...");
        VoidParameterServer.getInstance().shutdown();
    }

    /**
     * This method builds shadow vocabulary and huffman tree
     *
     * @param counter
     * @return
     */
    protected VocabCache<ShallowSequenceElement> buildShallowVocabCache(Counter<Long> counter) {

        // TODO: need simplified cache here, that will operate on Long instead of string labels
        VocabCache<ShallowSequenceElement> vocabCache = new AbstractCache<>();
        for (Long id : counter.keySet()) {
            ShallowSequenceElement shallowElement = new ShallowSequenceElement(counter.getCount(id), id);
            vocabCache.addToken(shallowElement);
        }

        // building huffman tree
        Huffman huffman = new Huffman(vocabCache.vocabWords());
        huffman.build();
        huffman.applyIndexes(vocabCache);

        return vocabCache;
    }

    protected Counter<Long> getCounter() {
        if (isAutoDiscoveryMode)
            return elementsFreqAccumExtra.value();
        else
            return elementsFreqAccum.value();
    }


    public static class Builder<T extends SequenceElement> {
        protected VectorsConfiguration configuration;
        protected SparkModelExporter<T> modelExporter;
        protected VoidConfiguration peersConfiguration;
        protected int workers;
        protected StorageLevel storageLevel;

        /**
         * This method should NOT be used in real world environment
         */
        @Deprecated
        public Builder() {
            this(new VoidConfiguration(), new VectorsConfiguration());
        }

        public Builder(@NonNull VoidConfiguration psConfiguration) {
            this(psConfiguration, new VectorsConfiguration());
        }

        public Builder(@NonNull VoidConfiguration psConfiguration, @NonNull VectorsConfiguration w2vConfiguration) {
            this.configuration = w2vConfiguration;
            this.peersConfiguration = psConfiguration;
        }

        /**
         *
         * @param level
         * @return
         */
        public Builder<T> setStorageLevel(StorageLevel level) {
            storageLevel = level;
            return this;
        }

        /**
         *
         * @param num
         * @return
         */
        public Builder<T> minWordFrequency(int num) {
            configuration.setMinWordFrequency(num);
            return this;
        }

        /**
         *
         * @param num
         * @return
         */
        public Builder<T> workers(int num) {
            this.workers = num;
            return this;
        }

        /**
         *
         * @param lr
         * @return
         */
        public Builder<T> setLearningRate(double lr) {
            configuration.setLearningRate(lr);
            return this;
        }

        /**
         *
         * @param configuration
         * @return
         */
        public Builder<T> setParameterServerConfiguration(@NonNull VoidConfiguration configuration) {
            peersConfiguration = configuration;
            return this;
        }

        /**
         *
         * @param modelExporter
         * @return
         */
        public Builder<T> setModelExporter(@NonNull SparkModelExporter<T> modelExporter) {
            this.modelExporter = modelExporter;
            return this;
        }

        /**
         *
         * @param num
         * @return
         */
        public Builder<T> epochs(int num) {
            configuration.setEpochs(num);
            return this;
        }

        /**
         *
         * @param num
         * @return
         */
        public Builder<T> iterations(int num) {
            configuration.setIterations(num);
            return this;
        }

        /**
         *
         * @param rate
         * @return
         */
        public Builder<T> subsampling(double rate) {
            configuration.setSampling(rate);
            return this;
        }

        /**
         *
         * @param reallyUse
         * @return
         */
        public Builder<T> useHierarchicSoftmax(boolean reallyUse) {
            configuration.setUseHierarchicSoftmax(reallyUse);
            return this;
        }

        /**
         *
         * @param samples
         * @return
         */
        public Builder<T> negativeSampling(long samples) {
            configuration.setNegative((double) samples);
            return this;
        }

        /**
         *
         * @param ela
         * @return
         */
        public Builder<T> setElementsLearningAlgorithm(@NonNull SparkElementsLearningAlgorithm ela) {
            configuration.setElementsLearningAlgorithm(ela.getClass().getCanonicalName());
            return this;
        }

        /**
         *
         * @param sla
         * @return
         */
        public Builder<T> setSequenceLearningAlgorithm(@NonNull SparkSequenceLearningAlgorithm sla) {
            configuration.setSequenceLearningAlgorithm(sla.getClass().getCanonicalName());
            return this;
        }

        public Builder<T> layerSize(int layerSize) {
            if (layerSize < 1)
                throw new DL4JInvalidConfigException("LayerSize should be positive value");

            configuration.setLayersSize(layerSize);
            return this;
        }


        public SparkSequenceVectors<T> build() {
            if (modelExporter == null)
                throw new IllegalStateException("ModelExporter is undefined!");

            SparkSequenceVectors seqVec = new SparkSequenceVectors(configuration);
            seqVec.exporter = modelExporter;
            seqVec.paramServerConfiguration = peersConfiguration;
            seqVec.storageLevel = storageLevel;
            seqVec.workers = workers;

            return seqVec;
        }
    }
}
