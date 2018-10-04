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

package org.nd4j.parameterserver.distributed.enums;

/**
 * TransportType enum describes different Transport usable for ParameterServers implementation
 *
 * @author raver119@gmail.com
 */
public enum TransportType {
    /**
     * This is default Transport implementation, suitable for network environments without UDP Broadcast support
     */
    ROUTED_UDP,

    /**
     * This option means you'll provide own Transport interface implementation via VoidParameterServer.init() method
     */
    CUSTOM,

    /**
     * This is hardware-specific Transport implementat.
     * PLEASE NOTE: Public cloud providers, like AWS, Azure, GCP?
     */
    MULTICAST,
}
