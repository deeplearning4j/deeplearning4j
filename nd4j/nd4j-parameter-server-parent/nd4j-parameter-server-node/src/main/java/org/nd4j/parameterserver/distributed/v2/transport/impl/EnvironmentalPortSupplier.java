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
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.v2.transport.PortSupplier;

/**
 * This class provides port information for Transport, based on environment variables
 * @author raver119@gmail.com
 */
@Data
public class EnvironmentalPortSupplier implements PortSupplier {
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private int port = -1;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private String variableName;

    protected EnvironmentalPortSupplier() {
        //
    }

    /**
     * This constructor builds StaticPortSupplier instance with pre-defined port
     * @param port
     */
    public EnvironmentalPortSupplier(@NonNull String varName) {
        variableName = varName;
    }

    @Override
    public int getPreferredPort() {
        if (port < 1) {
            val variable = System.getenv(variableName);
            if (variable == null)
                throw new ND4JIllegalStateException("Environment variable ["+ variableName+"] isn't defined");

            try {
                port = Integer.valueOf(variable);
            } catch (NumberFormatException e) {
                throw new ND4JIllegalStateException("Environment variable ["+ variableName+"] contains bad value: [" + variable + "]");
            }
        }

        return port;
    }
}
