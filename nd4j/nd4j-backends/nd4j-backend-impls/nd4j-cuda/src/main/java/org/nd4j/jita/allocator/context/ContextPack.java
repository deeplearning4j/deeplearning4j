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

package org.nd4j.jita.allocator.context;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.util.HashMap;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
public class ContextPack {
    @Getter
    @Setter
    private Integer deviceId;
    @Getter
    private int availableLanes;
    private Map<Integer, CudaContext> lanes = new HashMap<>();

    public ContextPack(int totalLanes) {
        availableLanes = totalLanes;
    }

    public ContextPack(CudaContext context) {
        this.availableLanes = 1;
        lanes.put(0, context);
    }

    public void addLane(@NonNull Integer laneId, @NonNull CudaContext context) {
        lanes.put(laneId, context);
        context.setLaneId(laneId);
    }

    public CudaContext getContextForLane(Integer laneId) {
        return lanes.get(laneId);
    }

    public int nextRandomLane() {
        if (availableLanes == 1)
            return 0;
        return RandomUtils.nextInt(0, availableLanes);
    }
}
