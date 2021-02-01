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

package org.deeplearning4j.earlystopping.listener;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.nn.api.Model;

public interface EarlyStoppingListener<T extends Model> {

    /**Method to be called when early stopping training is first started
     */
    void onStart(EarlyStoppingConfiguration<T> esConfig, T net);

    /**Method that is called at the end of each epoch completed during early stopping training
     * @param epochNum The number of the epoch just completed (starting at 0)
     * @param score The score calculated
     * @param esConfig Configuration
     * @param net Network (current)
     */
    void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<T> esConfig, T net);

    /**Method that is called at the end of early stopping training
     * @param esResult The early stopping result. Provides details of why early stopping training was terminated, etc
     */
    void onCompletion(EarlyStoppingResult<T> esResult);

}
