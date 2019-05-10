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

package org.nd4j.parameterserver.status.play;


import org.nd4j.parameterserver.model.SubscriberState;

import java.util.HashMap;
import java.util.Map;

/**
 * In memory status storage
 * for parameter server subscribers
 * @author Adam Gibson
 */
public class InMemoryStatusStorage extends BaseStatusStorage {

    /**
     * Create the storage map
     *
     * @return
     */
    @Override
    public Map<Integer, Long> createUpdatedMap() {
        return new HashMap<>();
    }

    @Override
    public Map<Integer, SubscriberState> createMap() {
        return new HashMap<>();
    }
}
