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

package org.deeplearning4j.malmo;

import org.deeplearning4j.rl4j.space.ObservationSpace;

import com.microsoft.msr.malmo.WorldState;

/**
 * Abstract base class for all Malmo-specific observation spaces
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public abstract class MalmoObservationSpace implements ObservationSpace<MalmoBox> {
    public abstract MalmoBox getObservation(WorldState world_state);
}
