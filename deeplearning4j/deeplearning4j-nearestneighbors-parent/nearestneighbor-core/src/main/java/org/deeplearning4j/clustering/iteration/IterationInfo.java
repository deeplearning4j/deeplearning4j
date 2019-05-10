/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
 ******************************************************************************/

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
