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
import java.util.List;
import java.util.Map;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.adapter.MultiDataSetIteratorAdapter;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

@Getter
@Setter
public class OutputConfig {

    @Setter(AccessLevel.NONE)
    private SameDiff sd;

    @NonNull
    private List<String> outputs = new ArrayList<>();

    @NonNull
    private List<Listener> listeners = new ArrayList<>();

    private MultiDataSetIterator data;

    public OutputConfig(@NonNull SameDiff sd) {
        this.sd = sd;
    }

    /**
     * Add required outputs
     */
    public OutputConfig output(@NonNull String... outputs) {
        this.outputs.addAll(Arrays.asList(outputs));
        return this;
    }

    /**
     * Add required outputs
     */
    public OutputConfig output(@NonNull SDVariable... outputs) {
        String[] outNames = new String[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            outNames[i] = outputs[i].name();
        }

        return output(outNames);
    }

    /**
     * Set the data to use as input.
     */
    public OutputConfig data(@NonNull MultiDataSetIterator data) {
        this.data = data;
        return this;
    }

    /**
     * Set the data to use as input.
     */
    public OutputConfig data(@NonNull DataSetIterator data) {
        this.data = new MultiDataSetIteratorAdapter(data);
        return this;
    }

    /**
     * Set the data to use as input.
     */
    public OutputConfig data(@NonNull DataSet data){
        return data(new SingletonMultiDataSetIterator(data.toMultiDataSet()));
    }

    /**
     * Set the data to use as input.
     */
    public OutputConfig data(@NonNull MultiDataSet data){
        return data(new SingletonMultiDataSetIterator(data));
    }

    /**
     * Add listeners for this operation
     */
    public OutputConfig listeners(@NonNull Listener... listeners) {
        this.listeners.addAll(Arrays.asList(listeners));
        return this;
    }

    private void validateConfig() {
        Preconditions.checkNotNull(data, "Must specify data.  It may not be null.");
    }

    /**
     * Do inference and return the results.
     *
     * Uses concatenation on the outputs of {@link #execBatches()} which may cause issues with some inputs. RNNs with
     * variable time series length and CNNs with variable image sizes will most likely have issues.
     */
    public Map<String, INDArray> exec() {
        return sd.output(data, listeners, outputs.toArray(new String[0]));
    }

    /**
     * Do inference and return the results in batches.
     */
    public List<Map<String, INDArray>> execBatches() {
        return sd.outputBatches(data, listeners, outputs.toArray(new String[0]));
    }

    /**
     * Do inference and return the results for the single output variable specified.
     *
     * Only works if exactly one output is specified.
     *
     * Uses concatenation on the outputs of {@link #execBatches()} which may cause issues with some inputs. RNNs with
     * variable time series length and CNNs with variable image sizes will most likely have issues.
     */
    public INDArray execSingle() {
        Preconditions.checkState(outputs.size() == 1,
                "Can only use execSingle() when exactly one output is specified, there were %s", outputs.size());

        return sd.output(data, listeners, outputs.toArray(new String[0])).get(outputs.get(0));
    }


    /**
     * Do inference and return the results (in batches) for the single output variable specified.
     *
     * Only works if exactly one output is specified.
     */
    public List<INDArray> execSingleBatches() {
        Preconditions.checkState(outputs.size() == 1,
                "Can only use execSingleBatches() when exactly one output is specified, there were %s", outputs.size());

        return SameDiffUtils
                .getSingleOutput(sd.outputBatches(data, listeners, outputs.toArray(new String[0])), outputs.get(0));
    }
}
