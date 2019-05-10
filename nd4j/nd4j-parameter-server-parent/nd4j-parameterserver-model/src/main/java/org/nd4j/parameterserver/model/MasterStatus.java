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

package org.nd4j.parameterserver.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Status of a master node, covered
 * both by the master node itself and its responder.
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MasterStatus implements Serializable {
    private String master, responder;
    private int responderN;


    /**
     * Returns true if bth
     * the master and responder are started.
     * @return
     */
    public boolean started() {
        return master.equals(ServerState.STARTED.name().toLowerCase())
                        && responder.equals(ServerState.STARTED.name().toLowerCase());
    }

}
