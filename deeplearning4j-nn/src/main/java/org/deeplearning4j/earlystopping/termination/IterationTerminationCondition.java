/*-
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

package org.deeplearning4j.earlystopping.termination;

import java.io.Serializable;

/**Interface for termination conditions to be evaluated once per iteration (i.e., once per minibatch).
 * Used for example to more quickly terminate training, instead of waiting for an epoch to complete before
 * checking termination conditions.
 * */
public interface IterationTerminationCondition extends Serializable {

    /** Initialize the iteration termination condition (sometimes a no-op)*/
    void initialize();

    /** Should early stopping training terminate at this iteration, based on the score for the last iteration?
     * return true if training should be terminated immediately, or false otherwise
     * @param lastMiniBatchScore Score of the last minibatch
     * @return whether to terminate or not
     */
    boolean terminate(double lastMiniBatchScore);

}
