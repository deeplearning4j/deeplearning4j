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

package org.datavec.api.io.serializers;

import java.io.IOException;
import java.io.OutputStream;

public interface Serializer<T> {
    /**
     * <p>Prepare the serializer for writing.</p>
     */
    void open(OutputStream out) throws IOException;

    /**
     * <p>Serialize <code>t</code> to the underlying output stream.</p>
     */
    void serialize(T t) throws IOException;

    /**
     * <p>Close the underlying output stream and clear up any resources.</p>
     */
    void close() throws IOException;
}
