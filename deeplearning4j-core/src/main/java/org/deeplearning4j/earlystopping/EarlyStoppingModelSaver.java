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

package org.deeplearning4j.earlystopping;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;
import java.io.Serializable;

/** Interface for saving MultiLayerNetworks learned during early stopping, and retrieving them again later */
public interface EarlyStoppingModelSaver<T extends Model> extends Serializable {

    /** Save the best model (so far) learned during early stopping training */
    void saveBestModel( T net, double score ) throws IOException;

    /** Save the latest (most recent) model learned during early stopping */
    void saveLatestModel( T net, double score ) throws IOException;

    /** Retrieve the best model that was previously saved */
    T getBestModel() throws IOException;

    /** Retrieve the most recent model that was previously saved */
    T getLatestModel() throws IOException;

}
