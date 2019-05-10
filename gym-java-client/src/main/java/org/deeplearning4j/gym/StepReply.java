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

package org.deeplearning4j.gym;

import lombok.Value;
import org.json.JSONObject;

/**
 * @param <T> type of observation
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/6/16.
 *
 *  StepReply is the container for the data returned after each step(action).
 */
@Value
public class StepReply<T> {

    T observation;
    double reward;
    boolean done;
    JSONObject info;

}
