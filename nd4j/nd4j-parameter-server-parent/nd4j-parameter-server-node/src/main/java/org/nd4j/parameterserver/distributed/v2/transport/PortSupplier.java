/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.parameterserver.distributed.v2.transport;

import java.io.Serializable;

/**
 * PortSupplier is an interface that is used to control the network ports used by Aeron (used in DL4J GradientSharing
 * implementation and ND4J's parameter server functionality).<br>
 *<br>
 * Build-in implementations include:<br>
 * {@link org.nd4j.parameterserver.distributed.v2.transport.impl.StaticPortSupplier}: used to provide a single fixed port for
 * use across all machines.<br>
 * {@link org.nd4j.parameterserver.distributed.v2.transport.impl.EnvironmentVarPortSupplier}: used to specify an environment
 * variable that should be read to determine the port to be used. The environment variable (and hence port) may be different
 * on different machines in the cluster.<br>
 * <br>
 * Note that users may provide their own PortSupplier implementation to provide different behaviour for determining the
 * networking port to use.
 *
 * @author raver119@gmail.com
 */
public interface PortSupplier extends Serializable {

    /**
     * This method determines which port should be used for networking - Aeron Transport layer
     * @return Port to use for networking. This must be available for use on the machine on which it is used.
     */
    int getPort();
}
