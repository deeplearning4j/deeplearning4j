/*
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
package org.arbiter.deeplearning4j;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

import java.io.Serializable;

/**DL4JConfiguration: simple configuration method that contains the following:<br>
 * - MultiLayerConfiguration<br>
 * - Early stopping settings, OR number of epochs<br>
 * Note: if early stopping configuration is absent, a fixed number of epochs (default: 1) will be used.
 * If both early stopping and number of epochs is present: early stopping will be used.
 */
@AllArgsConstructor @Data
public class DL4JConfiguration implements Serializable {

    private MultiLayerConfiguration multiLayerConfiguration;
    private EarlyStoppingConfiguration earlyStoppingConfiguration;
    private Integer numEpochs;

}
