package org.deeplearning4j.eval;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.AtomicDouble;
import org.nd4j.linalg.primitives.Pair;
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
        atomicModule.addSerializer(AtomicDouble.class,new JsonSerializerAtomicDouble());
        atomicModule.addSerializer(AtomicBoolean.class,new JsonSerializerAtomicBoolean());
        atomicModule.addDeserializer(AtomicDouble.class,new JsonDeserializerAtomicDouble());
        atomicModule.addDeserializer(AtomicBoolean.class,new JsonDeserializerAtomicBoolean());
        ret.registerModule(atomicModule);
        //Serialize fields only, not using getters
        ret.setVisibilityChecker(ret.getSerializationConfig().getDefaultVisibilityChecker()
                .withFieldVisibility(JsonAutoDetect.Visibility.ANY)
                .withGetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withSetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withCreatorVisibility(JsonAutoDetect.Visibility.NONE));
        return ret;
    }

    @Override
    public void evalTimeSeries(INDArray labels, INDArray predicted) {
        evalTimeSeries(labels, predicted, null);
    }

    @Override
    public void evalTimeSeries(INDArray labels, INDArray predictions, INDArray labelsMask) {
        Pair<INDArray, INDArray> pair = EvaluationUtils.extractNonMaskedTimeSteps(labels, predictions, labelsMask);
        if(pair == null){
            //No non-masked steps
            return;
        }
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
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return stats();
    }
}
