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

package org.deeplearning4j.ui.storage.impl;

import lombok.AllArgsConstructor;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;

import java.util.Queue;

/**
 * A very simple {@link StatsStorageListener}, that adds the {@link StatsStorageEvent} instances to a provided queue
 * for later processing.
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class QueueStatsStorageListener implements StatsStorageListener {

    private final Queue<StatsStorageEvent> queue;

    @Override
    public void notify(StatsStorageEvent event) {
        queue.add(event);
    }
}
