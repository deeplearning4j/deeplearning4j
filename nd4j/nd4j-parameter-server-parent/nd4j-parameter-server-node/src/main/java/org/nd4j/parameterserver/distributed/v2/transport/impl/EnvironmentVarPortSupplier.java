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

package org.nd4j.parameterserver.distributed.v2.transport.impl;

import lombok.*;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.v2.transport.PortSupplier;

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
