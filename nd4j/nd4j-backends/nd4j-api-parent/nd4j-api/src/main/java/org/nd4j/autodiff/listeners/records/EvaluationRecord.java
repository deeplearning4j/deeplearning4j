/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.autodiff.listeners.records;

import org.nd4j.shade.guava.base.Predicates;
import org.nd4j.shade.guava.collect.Collections2;
import org.nd4j.shade.guava.collect.ImmutableMap;
import org.nd4j.shade.guava.collect.Lists;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import lombok.Getter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.IMetric;

/**
 * A helper class to hold evaluations and provide methods to easily query them
 */
@Getter
public class EvaluationRecord {

    private Map<String, List<IEvaluation>> evaluations;
    private Map<Class<? extends IEvaluation>, IEvaluation> classEvaluations = new HashMap<>();
    private boolean isEmpty = true;

    public EvaluationRecord(Map<String, List<IEvaluation>> evaluations) {
        this.evaluations = Collections.unmodifiableMap(evaluations);

        for (List<IEvaluation> le : evaluations.values()) {
            for (IEvaluation e : le) {
                isEmpty = false;
                if (classEvaluations.containsKey(e.getClass()))
                    classEvaluations.remove(e.getClass());
                else
                    classEvaluations.put(e.getClass(), e);
            }
        }
    }

    private EvaluationRecord() {

    }

    public boolean isEmpty() {
        return isEmpty;
    }

    /**
     * Get all evaluations
     */
    public Map<String, List<IEvaluation>> evaluations() {
        return evaluations;
    }

    /**
     * Get evaluations for a given param/variable
     *
     * @param param The target param/variable
     */
    public List<IEvaluation> evaluations(String param) {
        Preconditions.checkArgument(evaluations.containsKey(param),
                "No evaluations for %s.", param);

        return evaluations.get(param);
    }

    /**
     * Get evaluations for a given param/variable
     *
     * @param param The target param/variable
     */
    public List<IEvaluation> evaluations(SDVariable param) {
        return evaluations(param.name());
    }

    /**
     * Get the evaluation for param at the specified index
     */
    public IEvaluation evaluation(String param, int index) {
        return evaluations(param).get(index);
    }

    /**
     * Get the evaluation for param at the specified index
     */
    public IEvaluation evaluation(SDVariable param, int index) {
        return evaluation(param.name(), index);
    }

    /**
     * Get the evaluation for a given param/variable
     * <p>
     * Will throw an exception if there are more than one or no evaluations for the param
     *
     * @param param The target param/variable
     */
    public <T extends IEvaluation> T evaluation(String param) {
        Preconditions.checkArgument(evaluations.containsKey(param),
                "No evaluations for %s.", param);
        Preconditions.checkArgument(evaluations.get(param).size() == 1,
                "Multiple evaluations for %s.  Use evaluations().", param);

        return (T) evaluations.get(param).get(0);
    }

    /**
     * Get the evaluation for a given param/variable
     * <p>
     * Will throw an exception if there are more than one or no evaluations for the param
     *
     * @param param The target param/variable
     */
    public <T extends IEvaluation> T evaluation(SDVariable param) {
        return evaluation(param.name());
    }

    /**
     * Get the evaluation of a given type
     * <p>
     * Will throw an exception if there are more than one or no evaluations of that type
     *
     * @param evalClass The type of evaluation to look for
     */
    public <T extends IEvaluation<T>> T evaluation(Class<T> evalClass) {
        Preconditions.checkArgument(classEvaluations.containsKey(evalClass),
                "Can't get evaluation for %s.  Either no evaluations with that class are present, or more than one are.", evalClass);

        return (T) classEvaluations.get(evalClass);
    }

    /**
     * Get the evaluation of a given type, for a given param/variable
     * <p>
     * Will throw an exception if there are more than one or no evaluations of that type for the given param
     *
     * @param param     The target param/variable
     * @param evalClass The type of evaluation to look for
     */
    public <T extends IEvaluation<T>> T evaluation(String param, Class<T> evalClass) {
        Collection<IEvaluation> evals = Collections2.filter(evaluations(param), Predicates.instanceOf(evalClass));

        Preconditions.checkArgument(evals.size() == 1, "Multiple or no evaluations of type %s for param %s.", evalClass, param);

        return (T) evals.iterator().next();
    }

    /**
     * Get the evaluation of a given type, for a given param/variable
     * <p>
     * Will throw an exception if there are more than one or no evaluations of that type for the given param
     *
     * @param param     The target param/variable
     * @param evalClass The type of evaluation to look for
     */
    public <T extends IEvaluation<T>> T evaluation(SDVariable param, Class<T> evalClass) {
        return evaluation(param.name(), evalClass);
    }

    /**
     * Get the metric's value for the evaluation of the metric's type
     * <p>
     * Will throw an exception if there are more than one or no evaluations of that type
     *
     * @param metric The metric to calculate
     */
    public double getValue(IMetric metric) {
        return evaluation(metric.getEvaluationClass()).getValue(metric);
    }

    /**
     * Get the metric's value for the evaluation of the metric's type, for a given param/variable
     * <p>
     * Will throw an exception if there are more than one or no evaluations of that type for the given param
     *
     * @param param  The target param/variable
     * @param metric The metric to calculate
     */
    public double getValue(String param, IMetric metric) {
        return evaluation(param, metric.getEvaluationClass()).getValue(metric);
    }

    /**
     * Get the metric's value for the evaluation of the metric's type, for a given param/variable
     * <p>
     * Will throw an exception if there are more than one or no evaluations of that type for the given param
     *
     * @param param  The target param/variable
     * @param metric The metric to calculate
     */
    public double getValue(SDVariable param, IMetric metric) {
        return getValue(param.name(), metric);
    }

    /**
     * Get the metric's value for the evaluation for a given param/variable at the given index
     * <p>
     * Will throw an exception if the target evaluation doesn't support the given metric
     *
     * @param param  The target param/variable
     * @param index  The index of the target evaluation on the param
     * @param metric The metric to calculate
     */
    public double getValue(String param, int index, IMetric metric) {
        return evaluation(param, index).getValue(metric);
    }

    /**
     * Get the metric's value for the evaluation for a given param/variable at the given index
     * <p>
     * Will throw an exception if the target evaluation doesn't support the given metric
     *
     * @param param  The target param/variable
     * @param index  The index of the target evaluation on the param
     * @param metric The metric to calculate
     */
    public double getValue(SDVariable param, int index, IMetric metric) {
        return getValue(param.name(), index, metric);
    }

}
