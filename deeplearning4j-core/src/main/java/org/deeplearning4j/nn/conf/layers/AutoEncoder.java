/*
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 *  Autoencoder.
 * Add Gaussian noise to input and learn
 * a reconstruction function.
 *
 */
@Data
@NoArgsConstructor
public class AutoEncoder extends BasePretrainNetwork {

    private static final long serialVersionUID = -7624965662728637504L;
    private double corruptionLevel;
    private double dropOut;
    private String activationFunction;
    private double sparsity;

    public AutoEncoder(int nIn, int nOut) {
        this.nIn = nIn;
        this.nOut = nOut;
    }

    public AutoEncoder(int nIn, int nOut, double corruptionLevel) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.corruptionLevel = corruptionLevel;
    }

    public AutoEncoder(int nIn, int nOut, double corruptionLevel, double dropOut) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.corruptionLevel = corruptionLevel;
        this.dropOut = dropOut;
    }

    public AutoEncoder(int nIn, int nOut, double corruptionLevel, double dropOut, String activationFunction) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.corruptionLevel = corruptionLevel;
        this.dropOut = dropOut;
        this.activationFunction = activationFunction;
    }

    public AutoEncoder(int nIn, int nOut, double corruptionLevel, double dropOut, String activationFunction, double sparsity) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.corruptionLevel = corruptionLevel;
        this.dropOut = dropOut;
        this.sparsity = sparsity;
        this.activationFunction = activationFunction;
    }

}
