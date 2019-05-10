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

package org.deeplearning4j.api.storage;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * StatsStorageEvent: use with {@link StatsStorageListener} to specify when the state of the {@link StatsStorage}
 * implementation changes.<br>
 * Note that depending on the {@link StatsStorageListener.EventType}, some of the
 * field may be null.
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class StatsStorageEvent {
    private final StatsStorage statsStorage;
    private final StatsStorageListener.EventType eventType;
    private final String sessionID;
    private final String typeID;
    private final String workerID;
    private final long timestamp;
}
