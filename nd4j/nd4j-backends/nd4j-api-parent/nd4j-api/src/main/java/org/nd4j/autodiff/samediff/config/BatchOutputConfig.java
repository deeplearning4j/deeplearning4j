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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

@Getter
@Setter
public class BatchOutputConfig {

    @Setter(AccessLevel.NONE)
    private SameDiff sd;

    @NonNull
    private List<String> outputs = new ArrayList<>();

    private Map<String, INDArray> placeholders = new HashMap<>();

    @NonNull
    private List<Listener> listeners = new ArrayList<>();

    public BatchOutputConfig(@NonNull SameDiff sd){
        this.sd = sd;
    }

    /**
     * Add required outputs
     */
    public BatchOutputConfig output(@NonNull String... outputs) {
        this.outputs.addAll(Arrays.asList(outputs));
        return this;
    }

    /**
     * Add required outputs
     */
    public BatchOutputConfig output(@NonNull SDVariable... outputs){
        String[] outNames = new String[outputs.length];
        for(int i = 0 ; i < outputs.length ; i++){
            outNames[i] = outputs[i].name();
        }

        return output(outNames);
    }

    /**
     * Add all variables as required outputs
     */
    public BatchOutputConfig outputAll(){
        return output(sd.variables().toArray(new SDVariable[0]));
    }

    /**
     * Add a placeholder value for a specified variable
     */
    public BatchOutputConfig input(@NonNull String variable, @NonNull INDArray placeholder){
        Preconditions.checkState(!placeholders.containsKey(variable),
                "Placeholder for variable %s already specified", variable);

        Preconditions.checkNotNull(sd.getVariable(variable),
                "Variable %s does not exist in this SameDiff graph", variable);

        placeholders.put(variable, placeholder);
        return this;
    }

    /**
     * See {@link #input(String, INDArray)}
     */
    public BatchOutputConfig input(@NonNull SDVariable variable, @NonNull INDArray placeholder){
        return input(variable.name(), placeholder);
    }

    /**
     * Calls {@link #input(String, INDArray)} on each entry in the map.
     */
    public BatchOutputConfig inputs(Map<String, INDArray> placeholders){

        if(placeholders == null) {
            this.placeholders = null;
            return this;
        }

        for(Map.Entry<String, INDArray> e : placeholders.entrySet()){
            input(e.getKey(), e.getValue());
        }

        return this;
    }

    /**
     * Add listeners for this operation
     */
    public BatchOutputConfig listeners(@NonNull Listener... listeners){
        this.listeners.addAll(Arrays.asList(listeners));
        return this;
    }

    /**
     * @deprecated Use {@link #output()}
     */
    @Deprecated
    public Map<String, INDArray> exec() {
        return output();
    }

    /**
     * Do inference and return the results
     */
    public Map<String,INDArray> output() {
        return sd.output(placeholders, listeners, outputs.toArray(new String[0]));
    }

    /**
     * @deprecated Use {@link #outputSingle()}
     */
    @Deprecated
    public INDArray execSingle() {
        return outputSingle();
    }

    /**
     * Do inference and return the results for the single output
     *
     * Only works if exactly one output is specified
     */
    public INDArray outputSingle(){
        Preconditions.checkState(outputs.size() == 1,
                "Can only use execSingle() when exactly one output is specified, there were %s", outputs.size());
        return exec().get(outputs.get(0));
    }
}
