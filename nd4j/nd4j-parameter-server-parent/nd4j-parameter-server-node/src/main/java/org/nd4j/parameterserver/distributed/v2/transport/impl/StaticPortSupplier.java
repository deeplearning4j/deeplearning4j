/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.parameterserver.distributed.v2.transport.impl;

import lombok.*;
import org.nd4j.common.base.Preconditions;
import org.nd4j.parameterserver.distributed.v2.transport.PortSupplier;

/**
 * This class provides static pre-defined port - a fixed value for all machines in the cluster.
 * @author raver119@gmail.com
 */
@Data
public class StaticPortSupplier implements PortSupplier {
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private int port = 49876;

    protected StaticPortSupplier() {
        //
    }

    /**
     * This constructor builds StaticPortSupplier instance with pre-defined port
     * @param port
     */
    public StaticPortSupplier(int port) {
        Preconditions.checkArgument(port > 0 && port <= 65535, "Invalid port: must be in range 1 to 65535 inclusive. Got: %s", port);
    }

    @Override
    public int getPort() {
        return port;
    }
}
