/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.evaluation.custom;

import org.nd4j.shade.guava.collect.Lists;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.nd4j.evaluation.BaseEvaluation;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.IMetric;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A evaluation using lambdas to calculate the score.
 *
 * Uses 3 lambdas:<br>
 *     EvaluationLambda: takes in the labels, predictions, mask, and metadata and returns a value of type T<br>
 *     MergeLambda: takes in two lists of Ts, returns one.  Used in merging for distributed training<br>
*      ResultLambda (in Metric): takes a list of Ts, returns a double value<br>
 *     <br>
 * The EvaluationLambda will be called on each batch, and the results will be stored in a list.
 * MergeLambda merges two of those lists for distributed training (think Spark or Map-Reduce).
 * ResultLambda gets a score from that list.
 *
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class CustomEvaluation<T> extends BaseEvaluation<CustomEvaluation> {

    /**
     * The metric used to get a score for the CustomEvaluation.  Uses a ResultLambda
     */
    @AllArgsConstructor
    @RequiredArgsConstructor
    public static class Metric<T> implements IMetric{

        @Getter
        @NonNull private ResultLambda<T> getResult;

        private boolean minimize = false;

        @Override
        public Class<? extends IEvaluation> getEvaluationClass() {
            return CustomEvaluation.class;
        }

        @Override
        public boolean minimize() {
            return minimize;
        }

        /**
         * A metric that takes the average of a list of doubles
         */
        public static Metric<Double> doubleAverage(boolean minimize){
            return new Metric<>(new ResultLambda<Double>() {
                @Override
                public double toResult(List<Double> data) {
                    int count = 0;
                    double sum = 0;
                    for (Double d : data) {
                        count++;
                        sum += d;
                    }
                    return sum / count;
                }
            }, minimize);
        }


        /**
         * A metric that takes the max of a list of doubles
         */
        public static Metric<Double> doubleMax(boolean minimize){
            return new Metric<>(new ResultLambda<Double>() {
                @Override
                public double toResult(List<Double> data) {
                    double max = 0;
                    for (Double d : data) {
                        if(d > max)
                            max = d;
                    }
                    return max;
                }
            }, minimize);
        }


        /**
         * A metric that takes the min of a list of doubles
         */
        public static Metric<Double> doubleMin(boolean minimize){
            return new Metric<>(new ResultLambda<Double>() {
                @Override
                public double toResult(List<Double> data) {
                    double max = 0;
                    for (Double d : data) {
                        if(d < max)
                            max = d;
                    }
                    return max;
                }
            }, minimize);
        }
    }

    /**
     * A MergeLambda that merges by concatenating the two lists
     */
    public static <R> MergeLambda<R> mergeConcatenate(){
        return new MergeLambda<R>() {
            @Override
            public List<R> merge(List<R> a, List<R> b) {
                List<R> res = Lists.newArrayList(a);
                res.addAll(b);
                return res;
            }
        };
    }

    @NonNull private EvaluationLambda<T> evaluationLambda;
    @NonNull private MergeLambda<T> mergeLambda;

    private List<T> evaluations = new ArrayList<>();

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray,
            List<? extends Serializable> recordMetaData) {
        evaluations.add(evaluationLambda.eval(labels, networkPredictions, maskArray, recordMetaData));
    }

    @Override
    public void merge(CustomEvaluation other) {
        evaluations = mergeLambda.merge(evaluations, other.evaluations);
    }

    @Override
    public void reset() {
        evaluations = new ArrayList<>();
    }

    @Override
    public String stats() {
        return "";
    }

    @Override
    public double getValue(IMetric metric) {
        if(metric instanceof Metric){
            return ((Metric<T>) metric).getGetResult().toResult(evaluations);
        } else
            throw new IllegalStateException("Can't get value for non-regression Metric " + metric);
    }

    @Override
    public CustomEvaluation<T> newInstance() {
        return new CustomEvaluation<T>(evaluationLambda, mergeLambda);
    }
}
