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

package org.nd4j.parameterserver.distributed.util;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@Data
public class NetworkInformation implements Serializable {
    protected long totalMemory = 0;
    protected long availableMemory = 0;
    protected List<String> ipAddresses = new ArrayList<>();


    public void addIpAddress(@NonNull String ip) {
        ipAddresses.add(ip);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        NetworkInformation that = (NetworkInformation) o;

        return ipAddresses != null ? ipAddresses.equals(that.ipAddresses) : that.ipAddresses == null;
    }

    @Override
    public int hashCode() {
        return ipAddresses != null ? ipAddresses.hashCode() : 0;
    }
}
