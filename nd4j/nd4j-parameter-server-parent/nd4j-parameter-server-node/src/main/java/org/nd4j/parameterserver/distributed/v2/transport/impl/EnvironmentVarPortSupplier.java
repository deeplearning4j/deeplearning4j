/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
 * This class is an implementation of {@link PortSupplier} that provides port information for Transport, based on
 * an environment variable.<br>
 * Note: The environment variable must be available on all machines in the cluster, and contain a valid port in range 1
 * to 65535 (inclusive) as an integer value. The environment variable may have different values on different machines,
 * which can be used to set the port to a different value on each worker node.<br>
 * Note that this implementation does not check if a port is actually available - it merely reads and parses
 * the environment variable on the worker, to decide the port to use.
 *
 * @author raver119@gmail.com
 */
@Data
public class EnvironmentVarPortSupplier implements PortSupplier {
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private int port = -1;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private String variableName;

    protected EnvironmentVarPortSupplier() {
        //
    }

    /**
     * This constructor builds StaticPortSupplier instance with pre-defined port
     * @param port
     */
    public EnvironmentVarPortSupplier(@NonNull String varName) {
        variableName = varName;
    }

    @Override
    public int getPort() {
            val variable = System.getenv(variableName);
            if (variable == null)
                throw new ND4JIllegalStateException("Unable to get networking port from environment variable:" +
                        " environment variable ["+ variableName+"] isn't defined");

            try {
                port = Integer.valueOf(variable);
            } catch (NumberFormatException e) {
                throw new ND4JIllegalStateException("Unable to get network port from environment variable:" +
                        " environment variable ["+ variableName+"] contains bad value: [" + variable + "]");
            }

            Preconditions.checkState(port > 0 && port <= 65535, "Invalid port for environment variable: ports must be" +
                    "between 0 (exclusive) and 65535 (inclusive). Got port: %s", port);

        return port;
    }
}
