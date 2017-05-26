package org.deeplearning4j.eval;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;
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

    private static ObjectMapper objectMapper = new ObjectMapper();
    private static ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());

    @Override
    public void evalTimeSeries(INDArray labels, INDArray predicted) {
        evalTimeSeries(labels, predicted, null);
    }

    @Override
    public void evalTimeSeries(INDArray labels, INDArray predictions, INDArray labelsMask) {
        Pair<INDArray, INDArray> pair = EvaluationUtils.extractNonMaskedTimeSteps(labels, predictions, labelsMask);
        INDArray labels2d = pair.getFirst();
        INDArray predicted2d = pair.getSecond();

        eval(labels2d, predicted2d);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, List<? extends Serializable> recordMetaData) {
        eval(labels, networkPredictions);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray) {
        if (maskArray == null) {
            if (labels.rank() == 3) {
                evalTimeSeries(labels, networkPredictions, maskArray);
            } else {
                eval(labels, networkPredictions);
            }
            return;
        }
        if (labels.rank() == 3 && maskArray.rank() == 2) {
            //Per-output masking
            evalTimeSeries(labels, networkPredictions, maskArray);
            return;
        }

        throw new UnsupportedOperationException(
                this.getClass().getSimpleName() + " does not support per-output masking");
    }

    /**
     * @return
     */
    @Override
    public String toJson() {
        try {
            return objectMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @return
     */
    @Override
    public String toYaml() {
        try {
            return yamlMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * @param json
     * @param clazz
     * @param <T>
     * @return
     */
    public static <T extends BaseEvaluation> T fromYaml(String json, Class<T> clazz) {
        try {
            return yamlMapper.readValue(json, clazz);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @param json
     * @param clazz
     * @param <T>
     * @return
     */
    public static <T extends BaseEvaluation> T fromJson(String json, Class<T> clazz) {
        try {
            return objectMapper.readValue(json, clazz);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return stats();
    }
}
