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

package org.nd4j.parameterserver.distributed.v2.transport.impl;

import lombok.*;
import org.nd4j.base.Preconditions;
import org.nd4j.parameterserver.distributed.v2.transport.PortSupplier;

/**
 * This class provides static pre-defined port
 * @author raver119@gmail.com
 */
@Data
public class StaticPortSupplier implements PortSupplier {
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private int port = 40123;

    protected StaticPortSupplier() {
        //
    }

    /**
     * This constructor builds StaticPortSupplier instance with pre-defined port
     * @param port
     */
    public StaticPortSupplier(int port) {
        Preconditions.checkArgument(port > 0, "Port must have positive value");
    }

    @Override
    public int getPreferredPort() {
        return port;
    }
}
