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

package org.deeplearning4j.clustering.iteration;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.clustering.info.ClusterSetInfo;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class IterationHistory implements Serializable {
    @Getter
    @Setter
    private Map<Integer, IterationInfo> iterationsInfos = new HashMap<>();

    /**
     *
     * @return
     */
    public ClusterSetInfo getMostRecentClusterSetInfo() {
        IterationInfo iterationInfo = getMostRecentIterationInfo();
        return iterationInfo == null ? null : iterationInfo.getClusterSetInfo();
    }

    /**
     *
     * @return
     */
    public IterationInfo getMostRecentIterationInfo() {
        return getIterationInfo(getIterationCount() - 1);
    }

    /**
     *
     * @return
     */
    public int getIterationCount() {
        return getIterationsInfos().size();
    }

    /**
     *
     * @param iterationIdx
     * @return
     */
    public IterationInfo getIterationInfo(int iterationIdx) {
        return getIterationsInfos().get(iterationIdx);
    }



}
