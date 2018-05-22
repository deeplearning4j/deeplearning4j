package org.deeplearning4j.eval;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.List;

/**
 * A general purpose interface for evaluating neural networks - methods are shared by implemetations such as
 * {@link Evaluation}, {@link RegressionEvaluation}, {@link ROC}, {@link ROCMultiClass}
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY)
public interface IEvaluation<T extends IEvaluation> extends Serializable {


    /**
     *
     * @param labels
     * @param networkPredictions
     */
    void eval(INDArray labels, INDArray networkPredictions);

    /**
     *
     * @param labels
     * @param networkPredictions
     * @param recordMetaData
     */
    void eval(INDArray labels, INDArray networkPredictions, List<? extends Serializable> recordMetaData);

    /**
     *
     * @param labels
     * @param networkPredictions
     * @param maskArray
     */
    void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray);


    /**
     *
     * @param labels
     * @param predicted
     */
    void evalTimeSeries(INDArray labels, INDArray predicted);

    /**
     *
     * @param labels
     * @param predicted
     * @param labelsMaskArray
     */
    void evalTimeSeries(INDArray labels, INDArray predicted, INDArray labelsMaskArray);

    /**
     *
     * @param other
     */
    void merge(T other);

    /**
     *
     */
    void reset();

    /**
     *
     * @return
     */
    String stats();

    /**
     *
     * @return
     */
    String toJson();

    /**
     *
     * @return
     */
    String toYaml();
}
