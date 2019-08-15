/*
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package org.nd4j.autodiff.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.exception.ND4JException;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Utilities for SameDiff training and inference
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class TrainingUtils {

    /**
     * Stack batch outputs, like an output from {@link org.nd4j.autodiff.samediff.SameDiff#output(MultiDataSetIterator, String...)}
     */
    public static Map<String, INDArray> stackOutputs(List<Map<String, INDArray>> outputs){
        Map<String, List<INDArray>> outs = new HashMap<>();
        for(Map<String, INDArray> batch : outputs){
            for(String k : batch.keySet()){
                if(!outs.containsKey(k))
                    outs.put(k, new ArrayList<INDArray>());
                outs.get(k).add(batch.get(k));
            }
        }

        Map<String, INDArray> ret = new HashMap<>();
        for(String k : outs.keySet()){
            try {
                ret.put(k, Nd4j.concat(0, outs.get(k).toArray(new INDArray[0])));
            } catch(Exception e){
                throw new ND4JException("Error concatenating batch outputs", e);
            }
        }
        return ret;
    }

    /**
     * Get a list of batch outputs for a single variable from a list of batch outputs for all variables
     */
    public static List<INDArray> getSingleOutput(List<Map<String, INDArray>> outputs, String output){
        List<INDArray> batches = new ArrayList<>();
        for(Map<String, INDArray> batch : outputs)
            batches.add(batch.get(output));

        return batches;
    }
}
