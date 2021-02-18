/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.core.storage.listener;

import org.deeplearning4j.core.storage.StatsStorageRouter;
import org.deeplearning4j.core.storage.StatsStorageRouterProvider;
import org.deeplearning4j.optimize.api.TrainingListener;

import java.io.Serializable;

public interface RoutingIterationListener extends TrainingListener, Cloneable, Serializable {

    void setStorageRouter(StatsStorageRouter router);

    StatsStorageRouter getStorageRouter();

    void setWorkerID(String workerID);

    String getWorkerID();

    void setSessionID(String sessionID);

    String getSessionID();

    RoutingIterationListener clone();

}
