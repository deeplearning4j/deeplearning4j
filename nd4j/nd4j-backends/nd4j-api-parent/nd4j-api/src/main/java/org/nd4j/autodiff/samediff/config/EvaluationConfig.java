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

package org.nd4j.autodiff.samediff.config;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.listeners.records.EvaluationRecord;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.dataset.adapter.MultiDataSetIteratorAdapter;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

@Getter
@Setter
public class EvaluationConfig {

    @NonNull
    private Map<String, List<IEvaluation>> evaluations = new HashMap<>();

    @NonNull
    private Map<String, Integer> labelIndices = new HashMap<>();

    private MultiDataSetIterator data;

    @NonNull
    private List<Listener> listeners = new ArrayList<>();

    private boolean singleInput = false;

    @Setter(AccessLevel.NONE)
    private SameDiff sd;

    public EvaluationConfig(@NonNull SameDiff sd){
        this.sd = sd;
    }

    /**
     * Add evaluations to be preformed on a specified variable, and set that variable's label index.
     *
     * Setting a label index is required if using a MultiDataSetIterator.
     *
     * @param param     The param to evaluate
     * @param labelIndex The label index of that parameter
     * @param evaluations The evaluations to preform
     */
    public EvaluationConfig evaluate(@NonNull String param, int labelIndex, @NonNull IEvaluation... evaluations){
        return evaluate(param, evaluations).labelIndex(param, labelIndex);
    }

    /**
     * See {@link #evaluate(String, int, IEvaluation[])}
     */
    public EvaluationConfig evaluate(@NonNull SDVariable variable, int labelIndex, @NonNull IEvaluation... evaluations){
        return evaluate(variable.name(), labelIndex, evaluations);
    }

    /**
     * Add evaluations to be preformed on a specified variable, without setting a label index.
     *
     * Setting a label index (which is not done here) is required if using a MultiDataSetIterator.
     *
     * @param param     The param to evaluate
     * @param evaluations The evaluations to preform
     */
    public EvaluationConfig evaluate(@NonNull String param, @NonNull IEvaluation... evaluations){
        if(this.evaluations.get(param) == null){
            this.evaluations.put(param, new ArrayList<IEvaluation>());
        }

        this.evaluations.get(param).addAll(Arrays.asList(evaluations));
        return this;
    }


    /**
     * See {@link #evaluate(String, IEvaluation[])}
     */
    public EvaluationConfig evaluate(@NonNull SDVariable variable, @NonNull IEvaluation... evaluations){
        return evaluate(variable.name(), evaluations);
    }

    /**
     * Set the label index for a parameter
     */
    public EvaluationConfig labelIndex(@NonNull String param, int labelIndex){
        if(this.labelIndices.get(param) != null){
            int existingIndex = this.labelIndices.get(param);
            Preconditions.checkArgument(existingIndex == labelIndex,
                    "Different label index already specified for param %s.  Already specified: %s, given: %s",
                    param, existingIndex, labelIndex);
        }

        labelIndices.put(param, labelIndex);

        return this;
    }

    /**
     * See {@link #labelIndex(String, int)}
     */
    public EvaluationConfig labelIndex(@NonNull SDVariable variable, int labelIndex){
        return labelIndex(variable.name(), labelIndex);
    }

    /**
     * Add listeners for this operation
     */
    public EvaluationConfig listeners(@NonNull Listener... listeners){
        this.listeners.addAll(Arrays.asList(listeners));
        return this;
    }

    /**
     * Set the data to evaluate on.
     *
     * Setting a label index for each variable to evaluate is required
     */
    public EvaluationConfig data(@NonNull MultiDataSetIterator data){
        this.data = data;
        singleInput = false;
        return this;
    }

    /**
     * Set the data to evaluate on.
     *
     * Setting a label index for each variable to evaluate is NOT required (since there is only one input)
     */
    public EvaluationConfig data(@NonNull DataSetIterator data){
        this.data = new MultiDataSetIteratorAdapter(data);
        singleInput = true;
        return this;
    }

    private void validateConfig(){
        Preconditions.checkNotNull(data, "Must specify data.  It may not be null.");

        if(!singleInput){
            for(String param : this.evaluations.keySet()){
                Preconditions.checkState(labelIndices.containsKey(param),
                        "Using multiple input dataset iterator without specifying a label index for %s", param);
            }
        }

        for(String param : this.evaluations.keySet()){
            Preconditions.checkState(sd.variableMap().containsKey(param),
                    "Parameter %s not present in this SameDiff graph", param);
        }
    }

    /**
     * Run the evaluation.
     *
     * Note that the evaluations in the returned {@link EvaluationRecord} are the evaluations set using {@link #evaluate(String, int, IEvaluation[])},
     * it does not matter which you use to access results.
     *
     * @return The specified listeners, in an {@link EvaluationRecord} for easy access.
     */
    public EvaluationRecord exec(){
        validateConfig();

        if(singleInput){
            for(String param : this.evaluations.keySet()){
                labelIndices.put(param, 0);
            }
        }

        sd.evaluate(data, evaluations, labelIndices, listeners.toArray(new Listener[0]));
        return new EvaluationRecord(evaluations);
    }

}
