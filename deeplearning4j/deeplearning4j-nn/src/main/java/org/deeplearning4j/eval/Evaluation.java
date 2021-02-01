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

package org.deeplearning4j.eval;

import lombok.EqualsAndHashCode;
import lombok.NonNull;
import org.nd4j.evaluation.EvaluationAveraging;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
 */
@EqualsAndHashCode(callSuper = true)
@Deprecated
public class Evaluation extends org.nd4j.evaluation.classification.Evaluation implements org.deeplearning4j.eval.IEvaluation<org.nd4j.evaluation.classification.Evaluation> {

    /**
     * Use {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public enum Metric {ACCURACY, F1, PRECISION, RECALL, GMEASURE, MCC;
        public org.nd4j.evaluation.classification.Evaluation.Metric toNd4j(){
            switch (this){
                case ACCURACY:
                    return org.nd4j.evaluation.classification.Evaluation.Metric.ACCURACY;
                case F1:
                    return org.nd4j.evaluation.classification.Evaluation.Metric.F1;
                case PRECISION:
                    return org.nd4j.evaluation.classification.Evaluation.Metric.PRECISION;
                case RECALL:
                    return org.nd4j.evaluation.classification.Evaluation.Metric.RECALL;
                case GMEASURE:
                    return org.nd4j.evaluation.classification.Evaluation.Metric.GMEASURE;
                case MCC:
                    return org.nd4j.evaluation.classification.Evaluation.Metric.MCC;
                default:
                    throw new IllegalStateException("Unknown enum state: " + this);
            }
        }
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation() {
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation(int numClasses) {
        super(numClasses);
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation(int numClasses, Integer binaryPositiveClass){
        super(numClasses, binaryPositiveClass);
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation(List<String> labels) {
        super(labels);
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation(Map<Integer, String> labels) {
        super(labels);
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation(List<String> labels, int topN) {
        super(labels, topN);
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation(double binaryDecisionThreshold) {
        super(binaryDecisionThreshold);
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation(double binaryDecisionThreshold, @NonNull Integer binaryPositiveClass) {
        super(binaryDecisionThreshold, binaryPositiveClass);
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation(INDArray costArray) {
        super(costArray);
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public Evaluation(List<String> labels, INDArray costArray) {
        super(labels, costArray);
    }

    @Deprecated
    public double precision(org.deeplearning4j.eval.EvaluationAveraging averaging) {
        return precision(averaging.toNd4j());
    }

    @Deprecated
    public double recall(org.deeplearning4j.eval.EvaluationAveraging averaging) {
        return recall(averaging.toNd4j());
    }

    public double falsePositiveRate(org.deeplearning4j.eval.EvaluationAveraging averaging) {
        return falsePositiveRate(averaging.toNd4j());
    }

    public double falseNegativeRate(org.deeplearning4j.eval.EvaluationAveraging averaging) {
        return falseNegativeRate(averaging.toNd4j());
    }

    public double f1(org.deeplearning4j.eval.EvaluationAveraging averaging) {
        return f1(averaging.toNd4j());
    }

    public double fBeta(double beta, org.deeplearning4j.eval.EvaluationAveraging averaging) {
        return fBeta(beta, averaging.toNd4j());
    }

    public double gMeasure(org.deeplearning4j.eval.EvaluationAveraging averaging) {
        return gMeasure(averaging.toNd4j());
    }

    public double matthewsCorrelation(org.deeplearning4j.eval.EvaluationAveraging averaging) {
        return matthewsCorrelation(averaging.toNd4j());
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    public double scoreForMetric(Metric metric){
        return scoreForMetric(metric.toNd4j());
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public static Evaluation fromJson(String json) {
        return fromJson(json, Evaluation.class);
    }

    /**
     * @deprecated Use ND4J Evaluation class, which has the same interface: {@link org.nd4j.evaluation.classification.Evaluation.Metric}
     */
    @Deprecated
    public static Evaluation fromYaml(String yaml) {
        return fromYaml(yaml, Evaluation.class);
    }
}
