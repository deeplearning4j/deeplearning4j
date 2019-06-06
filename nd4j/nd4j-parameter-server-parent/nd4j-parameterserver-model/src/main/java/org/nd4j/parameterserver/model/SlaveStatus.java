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
 * The slave status of whether the
 * slave node is started or not.
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SlaveStatus implements Serializable {
    private String slave;

    /**
     * Whether the slavenode is started or not.
     * @return
     */
    public boolean started() {
        return slave.equals(ServerState.STARTED.name().toLowerCase());
    }

}
