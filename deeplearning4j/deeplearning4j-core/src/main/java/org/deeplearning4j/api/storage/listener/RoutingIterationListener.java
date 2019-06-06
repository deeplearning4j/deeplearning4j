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

package org.deeplearning4j.api.storage.listener;

import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.optimize.api.TrainingListener;

import java.io.Serializable;

/**
 * An extension of the {@link TrainingListener} interface for those listeners that pass data off to a
 * {@link org.deeplearning4j.api.storage.StatsStorageRouter} instance.
 * The most common use case here is in distributed training scenarios: each worker has a set of listeners, that have
 * to be serialized and transferred across the network, to some storage mechanism.<br>
 * The StatsStorageRouter implementations themselves may not be serializable, or should be shared between multiple workers,
 * so instead, we use a {@link org.deeplearning4j.api.storage.StatsStorageRouterProvider}
 *
 * @author Alex Black
 */
public interface RoutingIterationListener extends TrainingListener, Cloneable, Serializable {

    void setStorageRouter(StatsStorageRouter router);

    StatsStorageRouter getStorageRouter();

    void setWorkerID(String workerID);

    String getWorkerID();

    void setSessionID(String sessionID);

    String getSessionID();

    RoutingIterationListener clone();

}
