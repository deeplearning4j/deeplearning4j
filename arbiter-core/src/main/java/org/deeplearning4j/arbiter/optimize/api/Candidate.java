/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.optimize.api;

import lombok.Data;

import java.io.Serializable;
import java.util.Map;

/**
 * Candidate: a proposed hyperparameter configuration.
 * Also includes a map for data parameters, to configure things like data preprocessing, etc.
 */
@Data
public class Candidate<C> implements Serializable {

    private C value;
    private int index;
    private double[] flatParameters;
    private Map<String, Object> dataParameters;

    public Candidate(C value, int index, double[] flatParameters) {
        this(value, index, flatParameters, null);
    }

    public Candidate(C value, int index, double[] flatParameters, Map<String, Object> dataParameters) {
        this.value = value;
        this.index = index;
        this.flatParameters = flatParameters;
        this.dataParameters = dataParameters;
    }


}
