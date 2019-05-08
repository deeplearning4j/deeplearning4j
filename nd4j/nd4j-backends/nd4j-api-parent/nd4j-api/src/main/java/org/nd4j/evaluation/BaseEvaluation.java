/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.evaluation;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.classification.*;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.AtomicDouble;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.primitives.serde.JsonDeserializerAtomicBoolean;
import org.nd4j.linalg.primitives.serde.JsonDeserializerAtomicDouble;
import org.nd4j.linalg.primitives.serde.JsonSerializerAtomicBoolean;
import org.nd4j.linalg.primitives.serde.JsonSerializerAtomicDouble;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.module.SimpleModule;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;

/**
 * BaseEvaluation implement common evaluation functionality (for time series, etc) for {@link Evaluation},
 * {@link RegressionEvaluation}, {@link ROC}, {@link ROCMultiClass} etc.
 *
 * @author Alex Black
 */
@EqualsAndHashCode
public abstract class BaseEvaluation<T extends BaseEvaluation> implements IEvaluation<T> {

    @Getter
    private static ObjectMapper objectMapper = configureMapper(new ObjectMapper());
    @Getter
    private static ObjectMapper yamlMapper = configureMapper(new ObjectMapper(new YAMLFactory()));

    private static ObjectMapper configureMapper(ObjectMapper ret) {
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, false);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        SimpleModule atomicModule = new SimpleModule();
        atomicModule.addSerializer(AtomicDouble.class, new JsonSerializerAtomicDouble());
        atomicModule.addSerializer(AtomicBoolean.class, new JsonSerializerAtomicBoolean());
        atomicModule.addDeserializer(AtomicDouble.class, new JsonDeserializerAtomicDouble());
        atomicModule.addDeserializer(AtomicBoolean.class, new JsonDeserializerAtomicBoolean());
        ret.registerModule(atomicModule);
        //Serialize fields only, not using getters
        ret.setVisibilityChecker(ret.getSerializationConfig().getDefaultVisibilityChecker()
                .withFieldVisibility(JsonAutoDetect.Visibility.ANY)
                .withGetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withSetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withCreatorVisibility(JsonAutoDetect.Visibility.NONE));
        return ret;
    }

    /**
     * @param yaml  YAML representation
     * @param clazz Class
     * @param <T>   Type to return
     * @return Evaluation instance
     */
    public static <T extends IEvaluation> T fromYaml(String yaml, Class<T> clazz) {
        try {
            return yamlMapper.readValue(yaml, clazz);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @param json  Jason representation of the evaluation instance
     * @param clazz Class
     * @param <T>   Type to return
     * @return Evaluation instance
     */
    public static <T extends IEvaluation> T fromJson(String json, Class<T> clazz) {
        try {
            return objectMapper.readValue(json, clazz);
        } catch (IllegalArgumentException e) {
            if (e.getMessage().contains("Invalid type id")) {
                try {
                    return (T) attempFromLegacyFromJson(json, e);
                } catch (Throwable t) {
                    throw new RuntimeException("Cannot deserialize from JSON - JSON is invalid?", t);
                }
            }
            throw e;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Attempt to load DL4J IEvaluation JSON from 1.0.0-beta2 or earlier.
     * Given IEvaluation classes were moved to ND4J with no major changes, a simple "find and replace" for the class
     * names is used.
     *
     * @param json              JSON to attempt to deserialize
     * @param originalException Original exception to be re-thrown if it isn't legacy JSON
     */
    protected static <T extends IEvaluation> T attempFromLegacyFromJson(String json, IllegalArgumentException originalException) {
        if (json.contains("org.deeplearning4j.eval.Evaluation")) {
            String newJson = json.replaceAll("org.deeplearning4j.eval.Evaluation", "org.nd4j.evaluation.classification.Evaluation");
            return (T) fromJson(newJson, Evaluation.class);
        }

        if (json.contains("org.deeplearning4j.eval.EvaluationBinary")) {
            String newJson = json.replaceAll("org.deeplearning4j.eval.EvaluationBinary", "org.nd4j.evaluation.classification.EvaluationBinary")
                    .replaceAll("org.deeplearning4j.eval.ROC", "org.nd4j.evaluation.classification.ROC")
                    .replaceAll("org.deeplearning4j.eval.curves.", "org.nd4j.evaluation.curves.");
            return (T) fromJson(newJson, EvaluationBinary.class);
        }

        if (json.contains("org.deeplearning4j.eval.EvaluationCalibration")) {
            String newJson = json.replaceAll("org.deeplearning4j.eval.EvaluationCalibration", "org.nd4j.evaluation.classification.EvaluationCalibration")
                    .replaceAll("org.deeplearning4j.eval.curves.", "org.nd4j.evaluation.curves.");
            return (T) fromJson(newJson, EvaluationCalibration.class);
        }

        if (json.contains("org.deeplearning4j.eval.ROCBinary")) {
            String newJson = json.replaceAll("org.deeplearning4j.eval.ROCBinary", "org.nd4j.evaluation.classification.ROCBinary")
                    .replaceAll("org.deeplearning4j.eval.ROC", "org.nd4j.evaluation.classification.ROC")   //Nested ROC instances internally
                    .replaceAll("org.deeplearning4j.eval.curves.", "org.nd4j.evaluation.curves.");

            return (T) fromJson(newJson, ROCBinary.class);
        }

        if (json.contains("org.deeplearning4j.eval.ROCMultiClass")) {
            String newJson = json.replaceAll("org.deeplearning4j.eval.ROCMultiClass", "org.nd4j.evaluation.classification.ROCMultiClass")
                    .replaceAll("org.deeplearning4j.eval.ROC", "org.nd4j.evaluation.classification.ROC")   //Nested ROC instances internally
                    .replaceAll("org.deeplearning4j.eval.curves.", "org.nd4j.evaluation.curves.");
            return (T) fromJson(newJson, ROCMultiClass.class);
        }

        if (json.contains("org.deeplearning4j.eval.ROC")) {       //Has to be checked after ROCBinary/ROCMultiClass due to it being a prefix
            String newJson = json.replaceAll("org.deeplearning4j.eval.ROC", "org.nd4j.evaluation.classification.ROC")
                    .replaceAll("org.deeplearning4j.eval.curves.", "org.nd4j.evaluation.curves.");
            return (T) fromJson(newJson, ROC.class);
        }

        if (json.contains("org.deeplearning4j.eval.RegressionEvaluation")) {
            String newJson = json.replaceAll("org.deeplearning4j.eval.RegressionEvaluation", "org.nd4j.evaluation.regression.RegressionEvaluation");
            return (T) fromJson(newJson, RegressionEvaluation.class);
        }

        throw originalException;
    }

    public static Triple<INDArray, INDArray, INDArray> reshapeAndExtractNotMasked(INDArray labels, INDArray predictions, INDArray mask, int axis) {

        if (labels.rank() == 2) {
            Preconditions.checkState(axis == 1, "Only axis=1 is supported 2d data - got axis=%s for labels array shape %ndShape", axis, labels);
            if (mask == null) {
                //no-op
                return new Triple<>(labels, predictions, null);
            } else {
                //2 possible cases: per-output masking, and per example masking
                if (mask.rank() == 1 || mask.isColumnVector()) {
                    int notMaskedCount = mask.neq(0.0).castTo(DataType.INT).sumNumber().intValue();
                    if (notMaskedCount == 0) {
                        //All steps masked - nothing left to evaluate
                        return null;
                    }
                    if (notMaskedCount == mask.length()) {
                        //No masked steps - returned as-is
                        return new Triple<>(labels, predictions, null);
                    }
                    int[] arr = mask.toIntVector();
                    int[] idxs = new int[notMaskedCount];
                    int pos = 0;
                    for (int i = 0; i < arr.length; i++) {
                        if (arr[i] != 0) {
                            idxs[pos++] = i;
                        }
                    }
                    INDArray retLabel = Nd4j.pullRows(labels, 1, idxs, 'c');
                    INDArray retPredictions = Nd4j.pullRows(predictions, 1, idxs, 'c');
                    return new Triple<>(retLabel, retPredictions, null);
                } else {
                    Preconditions.checkState(labels.equalShapes(mask), "If a mask array is present for 2d data, it must either be a vector (column vector)" +
                            " or have shape equal to the labels (for per-output masking, when supported). Got labels shape %ndShape, mask shape %ndShape",
                            labels, mask);
                    //Assume evaluation instances with per-output masking will handle that themselves (or throw exception if not supported)
                    return new Triple<>(labels, predictions, mask);
                }
            }
        } else if (labels.rank() == 3 || labels.rank() == 4 || labels.rank() == 5) {
            if(mask == null){
                return reshapeSameShapeTo2d(axis, labels, predictions, mask);
            } else {
                if(labels.rank() == 3){
                    if(mask.rank() == 2){
                        //Per time step masking
                        Pair<INDArray,INDArray> p = EvaluationUtils.extractNonMaskedTimeSteps(labels, predictions, mask);
                        if(p == null){
                            return null;
                        }
                        return new Triple<>(p.getFirst(), p.getSecond(), null);
                    } else {
                        //Per output mask
                        Preconditions.checkState(labels.equalShapes(mask), "If a mask array is present for 3d data, it must either be 2d (shape [minibatch, sequenceLength])" +
                                        " or have shape equal to the labels (for per-output masking, when supported). Got labels shape %ndShape, mask shape %ndShape",
                                labels, mask);
                        //Assume evaluation instances with per-output masking will handle that themselves (or throw exception if not supported)
                        //Just reshape to 2d

                        return reshapeSameShapeTo2d(axis, labels, predictions, mask);
                    }
                } else {
                    if(labels.equalShapes(mask)){
                        //Per output masking case
                        return reshapeSameShapeTo2d(axis, labels, predictions, mask);
                    }
                    throw new UnsupportedOperationException("Evaluation case not yet implemented: rank 4/5 labels with non-per-output mask arrays");
                }
            }
        } else {
            throw new IllegalStateException("Unknown array type passed to evaluation: labels array rank " + labels.rank()
                    + " with shape " + labels.shapeInfoToString() + ". Labels and predictions must always be rank 2 or higher, with leading dimension being minibatch dimension");
        }
    }

    private static Triple<INDArray,INDArray,INDArray> reshapeSameShapeTo2d(int axis, INDArray labels, INDArray predictions, INDArray mask){
        int[] permuteDims = new int[labels.rank()];
        int j=0;
        for( int i=0; i<labels.rank(); i++ ){
            if(i == axis){
                continue;
            }
            permuteDims[j++] = i;
        }
        permuteDims[j] = axis;
        long size0 = 1;
        for( int i=0; i<permuteDims.length-1; i++ ){
            size0 *= labels.size(permuteDims[i]);
        }

        INDArray lOut = labels.permute(permuteDims).dup('c').reshape('c',size0, labels.size(axis));
        INDArray pOut = predictions.permute(permuteDims).dup('c').reshape('c',size0, labels.size(axis));
        INDArray mOut = mask == null ? null : mask.permute(permuteDims).dup('c').reshape('c',size0, labels.size(axis));

        return new Triple<>(lOut, pOut, mOut);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, null, null);
    }

    @Override
    public void eval(@NonNull INDArray labels, @NonNull final INDArray predictions, final List<? extends Serializable> recordMetaData) {
        eval(labels, predictions, null, recordMetaData);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray) {
        eval(labels, networkPredictions, maskArray, null);
    }

    @Override
    public void evalTimeSeries(INDArray labels, INDArray predicted) {
        evalTimeSeries(labels, predicted, null);
    }

    @Override
    public void evalTimeSeries(INDArray labels, INDArray predictions, INDArray labelsMask) {
        Pair<INDArray, INDArray> pair = EvaluationUtils.extractNonMaskedTimeSteps(labels, predictions, labelsMask);
        if (pair == null) {
            //No non-masked steps
            return;
        }
        INDArray labels2d = pair.getFirst();
        INDArray predicted2d = pair.getSecond();

        eval(labels2d, predicted2d);
    }

    /**
     * @return JSON representation of the evaluation instance
     */
    @Override
    public String toJson() {
        try {
            return objectMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return stats();
    }

    /**
     * @return YAML  representation of the evaluation instance
     */
    @Override
    public String toYaml() {
        try {
            return yamlMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
}
