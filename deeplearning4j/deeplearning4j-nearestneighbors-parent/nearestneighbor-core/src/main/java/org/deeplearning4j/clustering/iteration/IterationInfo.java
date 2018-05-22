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

import lombok.AccessLevel;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.clustering.info.ClusterSetInfo;

import java.io.Serializable;

@Data
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class IterationInfo implements Serializable {

    private int index;
    private ClusterSetInfo clusterSetInfo;
    private boolean strategyApplied;

    public IterationInfo(int index) {
        super();
        this.index = index;
    }

    public IterationInfo(int index, ClusterSetInfo clusterSetInfo) {
        super();
        this.index = index;
        this.clusterSetInfo = clusterSetInfo;
    }

}
