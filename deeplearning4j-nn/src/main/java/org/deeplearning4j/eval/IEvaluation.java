package org.deeplearning4j.eval;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.List;

/**
 * A general purpose interface for evaluating neural networks - methods are shared by implemetations such as
 * {@link Evaluation}, {@link RegressionEvaluation}, {@link ROC}, {@link ROCMultiClass}
 *
 * @author Alex Black
 */
public interface IEvaluation<T extends IEvaluation> extends Serializable {


    void eval(INDArray labels, INDArray networkPredictions);

    void eval(INDArray labels, INDArray networkPredictions, List<? extends Serializable> recordMetaData );

    void evalTimeSeries(INDArray labels, INDArray predicted);

    void evalTimeSeries(INDArray labels, INDArray predicted, INDArray labelsMaskArray);

    void merge(T other);

}
