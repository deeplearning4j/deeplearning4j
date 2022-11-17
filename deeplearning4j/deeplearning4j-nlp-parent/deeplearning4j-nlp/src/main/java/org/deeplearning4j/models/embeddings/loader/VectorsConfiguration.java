/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.models.embeddings.loader;

import org.apache.commons.codec.binary.Base64;
import org.nd4j.shade.jackson.annotation.*;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Objects;

public class VectorsConfiguration implements Serializable {

    // word2vec params
    private Integer minWordFrequency = 5;
    private Double learningRate = 0.025;
    private Double minLearningRate = 0.0001;
    private Integer layersSize = 200;
    private Boolean useAdaGrad = false;
    private Integer batchSize = 512;
    private Integer iterations = 1;
    private Integer epochs = 1;
    private Integer window = 5;
    private Long seed = 1234L;
    private Double negative = 0.0d;
    private Boolean useHierarchicSoftmax = true;
    private Double sampling = 0.0d;
    private Integer learningRateDecayWords = 3;
    private int[] variableWindows;

    private Boolean hugeModelExpected = false;
    private Boolean useUnknown = false;

    private Integer scavengerActivationThreshold = 2000000;
    private Integer scavengerRetentionDelay = 3;

    private String elementsLearningAlgorithm;
    private String sequenceLearningAlgorithm;
    private String modelUtils;

    private int workers;

    private String tokenizerFactory;
    private String tokenPreProcessor;

    // this is one-off configuration value specially for NGramTokenizerFactory
    private Integer nGram;

    private String UNK = "UNK";
    private String STOP = "STOP";

    private Collection<String> stopList = new ArrayList<>();

    // overall model info
    private Integer vocabSize;

    // paravec-specific option
    private Boolean trainElementsVectors;
    private Boolean trainSequenceVectors;
    private Boolean allowParallelTokenization;
    private Boolean preciseWeightInit;
    private Boolean preciseMode ;

    private static ObjectMapper mapper;
    private static final Object lock = new Object();

    public VectorsConfiguration() {
        this.minWordFrequency = 5;
        this.learningRate = 0.025;
        this.minLearningRate = 0.0001;
        this.layersSize = 200;
        this.useAdaGrad = false;
        this.batchSize = 512;
        this.iterations = 1;
        this.epochs = 1;
        this.window = 5;
        this.negative = 0.0d;
        this.useHierarchicSoftmax = true;
        this.sampling = 0.0d;
        this.hugeModelExpected = false;
        this.useUnknown = false;
        this.scavengerActivationThreshold = 2000000;
        this.scavengerRetentionDelay = 3;
        this.UNK = "UNK";
        this.STOP = "STOP";
        this.stopList = new ArrayList<>();
        this.trainElementsVectors = true;
        this.trainSequenceVectors = true;
        this.allowParallelTokenization = false;
        this.preciseWeightInit = false;
        this.preciseMode = false;
        this.workers = Runtime.getRuntime().availableProcessors();

    }

    public Boolean getUseHierarchicSoftmax() {
        return useHierarchicSoftmax;
    }

    public Boolean getHugeModelExpected() {
        return hugeModelExpected;
    }

    public Boolean getUseUnknown() {
        return useUnknown;
    }

    public int getWorkers() {
        return workers;
    }

    public void setWorkers(int workers) {
        this.workers = workers;
    }

    public Boolean getTrainElementsVectors() {
        return trainElementsVectors;
    }

    public Boolean getTrainSequenceVectors() {
        return trainSequenceVectors;
    }

    public Boolean getAllowParallelTokenization() {
        return allowParallelTokenization;
    }

    public Boolean getPreciseWeightInit() {
        return preciseWeightInit;
    }

    public Boolean getPreciseMode() {
        return preciseMode;
    }

    @JsonCreator
    public VectorsConfiguration(@JsonProperty("minWordFrequency")  Integer minWordFrequency,
                                @JsonProperty("learningRate") Double learningRate,
                                @JsonProperty("minLearningRate") Double minLearningRate,
                                @JsonProperty("layersSize") Integer layersSize,
                                @JsonProperty("useAdaGrad") Boolean useAdaGrad,
                                @JsonProperty("batchSize") Integer batchSize,
                                @JsonProperty("iterations") Integer iterations,
                                @JsonProperty("epochs") Integer epochs,
                                @JsonProperty("window") Integer window,
                                @JsonProperty("seed") Long seed,
                                @JsonProperty("negative") Double negative,
                                @JsonProperty("useHierarchicSoftmax") Boolean useHierarchicSoftmax,
                                @JsonProperty("sampling") Double sampling,
                                @JsonProperty("learningRateDecayWords") Integer learningRateDecayWords,
                                @JsonProperty("variableWindows") int[] variableWindows,
                                @JsonProperty("hugeModelExpected") Boolean hugeModelExpected,
                                @JsonProperty("useUnknown")  Boolean useUnknown,
                                @JsonProperty("scavengerActivationThreshold") Integer scavengerActivationThreshold,
                                @JsonProperty("scavengerRetentionDelay") Integer scavengerRetentionDelay,
                                @JsonProperty("elementsLearningAlgorithm") String elementsLearningAlgorithm,
                                @JsonProperty("sequenceLearningAlgorithm") String sequenceLearningAlgorithm,
                                @JsonProperty("modelUtils") String modelUtils,
                                @JsonProperty("tokenizerFactory") String tokenizerFactory,
                                @JsonProperty("tokenPreProcessor") String tokenPreProcessor,
                                @JsonProperty("nGram") Integer nGram,
                                @JsonProperty("UNK") String UNK,
                                @JsonProperty("STOP") String STOP,
                                @JsonProperty("stopList") Collection<String> stopList,
                                @JsonProperty("vocabSize")   Integer vocabSize,
                                @JsonProperty("trainElementsVectors") Boolean trainElementsVectors,
                                @JsonProperty("trainSequenceVectors") Boolean trainSequenceVectors,
                                @JsonProperty("allowParallelTokenization") Boolean allowParallelTokenization,
                                @JsonProperty("preciseWeightInit") Boolean preciseWeightInit,
                                @JsonProperty("preciseMode") Boolean preciseMode,
                                @JsonProperty("workers") Integer workers) {
        if(minWordFrequency != null)
            this.minWordFrequency = minWordFrequency;
        else
            this.minWordFrequency = 5;
        if(learningRate != null)
            this.learningRate = learningRate;
        else
            this.learningRate = 0.025;
        if(minLearningRate != null)
            this.minLearningRate = minLearningRate;
        else
            this.minLearningRate = 0.0001;
        if(layersSize != null)
            this.layersSize = layersSize;
        else
            this.layersSize = 200;
        if(useAdaGrad != null)
            this.useAdaGrad = useAdaGrad;
        else
            this.useAdaGrad = false;
        if(batchSize != null)
            this.batchSize = batchSize;
        else
            this.batchSize = 512;
        if(iterations != null)
            this.iterations = iterations;
        else
            this.iterations = 1;
        if(epochs != null)
            this.epochs = epochs;
        else
            this.epochs = 1;
        if(window != null)
            this.window = window;
        else
            this.window = 5;

        if(seed != null)
            this.seed = seed;
        else
            this.seed = 0L;
        if(negative != null)
            this.negative = negative;
        else
            this.negative = 0.0d;
        if(useHierarchicSoftmax != null)
            this.useHierarchicSoftmax = useHierarchicSoftmax;
        else
            this.useHierarchicSoftmax = true;
        if(this.sampling != null)
            this.sampling = sampling;
        else
            this.sampling = 0.0d;
        if(learningRateDecayWords != null)
            this.learningRateDecayWords = learningRateDecayWords;
        else
            this.learningRateDecayWords = 0;
        this.variableWindows = variableWindows;
        if(hugeModelExpected != null)
            this.hugeModelExpected = hugeModelExpected;
        else
            this.hugeModelExpected = false;
        if(this.useUnknown != null)
            this.useUnknown = useUnknown;
        else
            this.useUnknown = false;
        if(scavengerActivationThreshold != null)
            this.scavengerActivationThreshold = scavengerActivationThreshold;
        else
            this.scavengerActivationThreshold = 2000000;
        if(scavengerRetentionDelay != null)
            this.scavengerRetentionDelay = scavengerRetentionDelay;
        else {
            this.scavengerRetentionDelay = 3;
        }

        this.elementsLearningAlgorithm = elementsLearningAlgorithm;
        this.sequenceLearningAlgorithm = sequenceLearningAlgorithm;

        this.modelUtils = modelUtils;
        this.tokenizerFactory = tokenizerFactory;
        this.tokenPreProcessor = tokenPreProcessor;
        this.nGram = nGram;

        if(workers != null)
            this.workers = workers;
        else this.workers = Runtime.getRuntime().availableProcessors();

        if(UNK != null)
            this.UNK = UNK;
        else
            this.UNK = "UNK";
        if(STOP != null)
            this.STOP = STOP;
        else
            this.STOP = "STOP";
        if(stopList != null)
            this.stopList = stopList;
        else
            this.stopList = new ArrayList<>();
        this.vocabSize = vocabSize;
        this.trainElementsVectors = trainElementsVectors;
        this.trainSequenceVectors = trainSequenceVectors;
        if(allowParallelTokenization != null)
            this.allowParallelTokenization = allowParallelTokenization;
        else
            this.allowParallelTokenization = false;
        if(preciseWeightInit != null)
            this.preciseWeightInit = preciseWeightInit;
        else
            this.preciseWeightInit = false;
        if(preciseMode != null)
            this.preciseMode = preciseMode;
        else
            this.preciseMode = false;
    }

    private static ObjectMapper mapper() {
        if (mapper == null) {
            synchronized (lock) {
                if (mapper == null) {
                    mapper = new ObjectMapper();
                    mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
                    mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
                    mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, false);         //Use order in which fields are defined in classes
                    mapper.enable(SerializationFeature.INDENT_OUTPUT);
                    mapper.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
                    mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
                    mapper.setVisibility(PropertyAccessor.CREATOR, JsonAutoDetect.Visibility.ANY);
                    mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);


                    mapper.configure(SerializationFeature.FAIL_ON_UNWRAPPED_TYPE_IDENTIFIERS, false);
                    return mapper;
                }
            }
        }
        return mapper;
    }

    public String toJson() {
        ObjectMapper mapper = mapper();
        try {
            /*
                we need JSON as single line to save it at first line of the CSV model file
                That's ugly method, but its way more memory-friendly then loading whole 10GB json file just to create another 10GB memory array.
            */
            return mapper.writeValueAsString(this);
        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public String toEncodedJson() {
        Base64 base64 = new Base64(Integer.MAX_VALUE);
        try {
            return base64.encodeAsString(this.toJson().getBytes("UTF-8"));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }



    public static VectorsConfiguration fromEncodedJson(String json) {
        Base64 base64 = new Base64(Integer.MAX_VALUE);
        try {
            String decoded = new String(base64.decode(json.getBytes("UTF-8")));
            return VectorsConfiguration.fromJson(decoded);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static VectorsConfiguration fromJson(String json) {
        ObjectMapper mapper = mapper();
        try {
            VectorsConfiguration ret = mapper.readValue(json, VectorsConfiguration.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public Integer getMinWordFrequency() {
        return minWordFrequency;
    }

    public void setMinWordFrequency(Integer minWordFrequency) {
        this.minWordFrequency = minWordFrequency;
    }

    public Double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(Double learningRate) {
        this.learningRate = learningRate;
    }

    public Double getMinLearningRate() {
        return minLearningRate;
    }

    public void setMinLearningRate(Double minLearningRate) {
        this.minLearningRate = minLearningRate;
    }

    public Integer getLayersSize() {
        return layersSize;
    }

    public void setLayersSize(Integer layersSize) {
        this.layersSize = layersSize;
    }

    public Boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public void setUseAdaGrad(Boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }

    public Integer getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(Integer batchSize) {
        this.batchSize = batchSize;
    }

    public Integer getIterations() {
        return iterations;
    }

    public void setIterations(Integer iterations) {
        this.iterations = iterations;
    }

    public Integer getEpochs() {
        return epochs;
    }

    public void setEpochs(Integer epochs) {
        this.epochs = epochs;
    }

    public Integer getWindow() {
        return window;
    }

    public void setWindow(Integer window) {
        this.window = window;
    }

    public Long getSeed() {
        return seed;
    }

    public void setSeed(Long seed) {
        this.seed = seed;
    }

    public Double getNegative() {
        return negative;
    }

    public void setNegative(Double negative) {
        this.negative = negative;
    }

    public Boolean isUseHierarchicSoftmax() {
        return useHierarchicSoftmax;
    }

    public void setUseHierarchicSoftmax(Boolean useHierarchicSoftmax) {
        this.useHierarchicSoftmax = useHierarchicSoftmax;
    }

    public Double getSampling() {
        return sampling;
    }

    public void setSampling(Double sampling) {
        this.sampling = sampling;
    }

    public Integer getLearningRateDecayWords() {
        return learningRateDecayWords;
    }

    public void setLearningRateDecayWords(Integer learningRateDecayWords) {
        this.learningRateDecayWords = learningRateDecayWords;
    }

    public int[] getVariableWindows() {
        return variableWindows;
    }

    public void setVariableWindows(int[] variableWindows) {
        this.variableWindows = variableWindows;
    }

    public Boolean isHugeModelExpected() {
        return hugeModelExpected;
    }

    public void setHugeModelExpected(Boolean hugeModelExpected) {
        this.hugeModelExpected = hugeModelExpected;
    }

    public Boolean isUseUnknown() {
        return useUnknown;
    }

    public void setUseUnknown(Boolean useUnknown) {
        this.useUnknown = useUnknown;
    }

    public Integer getScavengerActivationThreshold() {
        return scavengerActivationThreshold;
    }

    public void setScavengerActivationThreshold(Integer scavengerActivationThreshold) {
        this.scavengerActivationThreshold = scavengerActivationThreshold;
    }

    public Integer getScavengerRetentionDelay() {
        return scavengerRetentionDelay;
    }

    public void setScavengerRetentionDelay(Integer scavengerRetentionDelay) {
        this.scavengerRetentionDelay = scavengerRetentionDelay;
    }

    public String getElementsLearningAlgorithm() {
        return elementsLearningAlgorithm;
    }

    public void setElementsLearningAlgorithm(String elementsLearningAlgorithm) {
        this.elementsLearningAlgorithm = elementsLearningAlgorithm;
    }

    public String getSequenceLearningAlgorithm() {
        return sequenceLearningAlgorithm;
    }

    public void setSequenceLearningAlgorithm(String sequenceLearningAlgorithm) {
        this.sequenceLearningAlgorithm = sequenceLearningAlgorithm;
    }

    public String getModelUtils() {
        return modelUtils;
    }

    public void setModelUtils(String modelUtils) {
        this.modelUtils = modelUtils;
    }

    public String getTokenizerFactory() {
        return tokenizerFactory;
    }

    public void setTokenizerFactory(String tokenizerFactory) {
        this.tokenizerFactory = tokenizerFactory;
    }

    public String getTokenPreProcessor() {
        return tokenPreProcessor;
    }

    public void setTokenPreProcessor(String tokenPreProcessor) {
        this.tokenPreProcessor = tokenPreProcessor;
    }

    public Integer getnGram() {
        return nGram;
    }

    public void setnGram(Integer nGram) {
        this.nGram = nGram;
    }

    public String getUNK() {
        return UNK;
    }

    public void setUNK(String UNK) {
        this.UNK = UNK;
    }

    public String getSTOP() {
        return STOP;
    }

    public void setSTOP(String STOP) {
        this.STOP = STOP;
    }

    public Collection<String> getStopList() {
        return stopList;
    }

    public void setStopList(Collection<String> stopList) {
        this.stopList = stopList;
    }

    public Integer getVocabSize() {
        return vocabSize;
    }

    public void setVocabSize(Integer vocabSize) {
        this.vocabSize = vocabSize;
    }

    public Boolean isTrainElementsVectors() {
        return trainElementsVectors;
    }

    public void setTrainElementsVectors(Boolean trainElementsVectors) {
        this.trainElementsVectors = trainElementsVectors;
    }

    public Boolean isTrainSequenceVectors() {
        return trainSequenceVectors;
    }

    public void setTrainSequenceVectors(Boolean trainSequenceVectors) {
        this.trainSequenceVectors = trainSequenceVectors;
    }

    public Boolean isAllowParallelTokenization() {
        return allowParallelTokenization;
    }

    public void setAllowParallelTokenization(Boolean allowParallelTokenization) {
        this.allowParallelTokenization = allowParallelTokenization;
    }

    public Boolean isPreciseWeightInit() {
        return preciseWeightInit;
    }

    public void setPreciseWeightInit(Boolean preciseWeightInit) {
        this.preciseWeightInit = preciseWeightInit;
    }

    public Boolean isPreciseMode() {
        return preciseMode;
    }

    public void setPreciseMode(Boolean preciseMode) {
        this.preciseMode = preciseMode;
    }

    public static ObjectMapper getMapper() {
        return mapper;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof VectorsConfiguration)) return false;
        VectorsConfiguration that = (VectorsConfiguration) o;
        return Objects.equals(getMinWordFrequency(), that.getMinWordFrequency()) && Objects.equals(getLearningRate(), that.getLearningRate()) && Objects.equals(getMinLearningRate(), that.getMinLearningRate()) && Objects.equals(getLayersSize(), that.getLayersSize()) && Objects.equals(useAdaGrad, that.useAdaGrad) && Objects.equals(getBatchSize(), that.getBatchSize()) && Objects.equals(getIterations(), that.getIterations()) && Objects.equals(getEpochs(), that.getEpochs()) && Objects.equals(getWindow(), that.getWindow()) && Objects.equals(getSeed(), that.getSeed()) && Objects.equals(getNegative(), that.getNegative()) && Objects.equals(useHierarchicSoftmax, that.useHierarchicSoftmax) && Objects.equals(getSampling(), that.getSampling()) && Objects.equals(getLearningRateDecayWords(), that.getLearningRateDecayWords()) && Arrays.equals(getVariableWindows(), that.getVariableWindows()) && Objects.equals(hugeModelExpected, that.hugeModelExpected) && Objects.equals(useUnknown, that.useUnknown) && Objects.equals(getScavengerActivationThreshold(), that.getScavengerActivationThreshold()) && Objects.equals(getScavengerRetentionDelay(), that.getScavengerRetentionDelay()) && Objects.equals(getElementsLearningAlgorithm(), that.getElementsLearningAlgorithm()) && Objects.equals(getSequenceLearningAlgorithm(), that.getSequenceLearningAlgorithm()) && Objects.equals(getModelUtils(), that.getModelUtils()) && Objects.equals(getTokenizerFactory(), that.getTokenizerFactory()) && Objects.equals(getTokenPreProcessor(), that.getTokenPreProcessor()) && Objects.equals(getnGram(), that.getnGram()) && Objects.equals(getUNK(), that.getUNK()) && Objects.equals(getSTOP(), that.getSTOP()) && Objects.equals(getStopList(), that.getStopList()) && Objects.equals(getVocabSize(), that.getVocabSize()) && Objects.equals(trainElementsVectors, that.trainElementsVectors) && Objects.equals(trainSequenceVectors, that.trainSequenceVectors) && Objects.equals(allowParallelTokenization, that.allowParallelTokenization) && Objects.equals(preciseWeightInit, that.preciseWeightInit) && Objects.equals(preciseMode, that.preciseMode);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(getMinWordFrequency(), getLearningRate(), getMinLearningRate(), getLayersSize(), useAdaGrad, getBatchSize(), getIterations(), getEpochs(), getWindow(), getSeed(), getNegative(), useHierarchicSoftmax, getSampling(), getLearningRateDecayWords(), hugeModelExpected, useUnknown, getScavengerActivationThreshold(), getScavengerRetentionDelay(), getElementsLearningAlgorithm(), getSequenceLearningAlgorithm(), getModelUtils(), getTokenizerFactory(), getTokenPreProcessor(), getnGram(), getUNK(), getSTOP(), getStopList(), getVocabSize(), trainElementsVectors, trainSequenceVectors, allowParallelTokenization, preciseWeightInit, preciseMode);
        result = 31 * result + Arrays.hashCode(getVariableWindows());
        return result;
    }

    @Override
    public String toString() {
        return "VectorsConfiguration{" +
                "minWordFrequency=" + minWordFrequency +
                ", learningRate=" + learningRate +
                ", minLearningRate=" + minLearningRate +
                ", layersSize=" + layersSize +
                ", useAdaGrad=" + useAdaGrad +
                ", batchSize=" + batchSize +
                ", iterations=" + iterations +
                ", epochs=" + epochs +
                ", window=" + window +
                ", seed=" + seed +
                ", negative=" + negative +
                ", useHierarchicSoftmax=" + useHierarchicSoftmax +
                ", sampling=" + sampling +
                ", learningRateDecayWords=" + learningRateDecayWords +
                ", variableWindows=" + Arrays.toString(variableWindows) +
                ", hugeModelExpected=" + hugeModelExpected +
                ", useUnknown=" + useUnknown +
                ", scavengerActivationThreshold=" + scavengerActivationThreshold +
                ", scavengerRetentionDelay=" + scavengerRetentionDelay +
                ", elementsLearningAlgorithm='" + elementsLearningAlgorithm + '\'' +
                ", sequenceLearningAlgorithm='" + sequenceLearningAlgorithm + '\'' +
                ", modelUtils='" + modelUtils + '\'' +
                ", tokenizerFactory='" + tokenizerFactory + '\'' +
                ", tokenPreProcessor='" + tokenPreProcessor + '\'' +
                ", nGram=" + nGram +
                ", UNK='" + UNK + '\'' +
                ", STOP='" + STOP + '\'' +
                ", stopList=" + stopList +
                ", vocabSize=" + vocabSize +
                ", trainElementsVectors=" + trainElementsVectors +
                ", trainSequenceVectors=" + trainSequenceVectors +
                ", allowParallelTokenization=" + allowParallelTokenization +
                ", preciseWeightInit=" + preciseWeightInit +
                ", preciseMode=" + preciseMode +
                '}';
    }

    public static void setMapper(ObjectMapper mapper) {
        VectorsConfiguration.mapper = mapper;
    }
}
