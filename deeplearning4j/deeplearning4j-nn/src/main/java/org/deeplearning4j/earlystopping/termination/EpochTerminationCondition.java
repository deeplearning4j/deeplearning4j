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


import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/** Interface for termination conditions to be evaluated once per epoch (i.e., once per pass of the full data set),
 *  based on a score and epoch number
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonSubTypes(value = {
                @JsonSubTypes.Type(value = BestScoreEpochTerminationCondition.class,
                                name = "BestScoreEpochTerminationCondition"),
                @JsonSubTypes.Type(value = MaxEpochsTerminationCondition.class, name = "MaxEpochsTerminationCondition"),
                @JsonSubTypes.Type(value = MaxScoreIterationTerminationCondition.class,
                                name = "MaxScoreIterationTerminationCondition"),

})
public interface EpochTerminationCondition extends Serializable {

    /** Initialize the epoch termination condition (often a no-op)*/
    void initialize();

    /**Should the early stopping training terminate at this epoch, based on the calculated score and the epoch number?
     * Returns true if training should terminated, or false otherwise
     * @param epochNum Number of the last completed epoch (starting at 0)
     * @param score Score calculate for this epoch
     * @return Whether training should be terminated at this epoch
     */
    boolean terminate(int epochNum, double score);

}
