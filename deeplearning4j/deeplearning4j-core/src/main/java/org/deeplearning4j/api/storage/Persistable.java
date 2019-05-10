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

package org.deeplearning4j.api.storage;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;

/**
 * Created by Alex on 07/10/2016.
 */
public interface Persistable extends Serializable {

    /**
     * Get the session id
     * @return
     */
    String getSessionID();

    /**
     * Get the type id
     * @return
     */
    String getTypeID();

    /**
     * Get the worker id
     * @return
     */
    String getWorkerID();

    /**
     * Get when this was created.
     * @return
     */
    long getTimeStamp();


    //SerDe methods:

    /**
     * Length of the encoding, in bytes, when using {@link #encode()}
     * Length may be different using {@link #encode(OutputStream)}, due to things like stream headers
     * @return
     */
    int encodingLengthBytes();

    byte[] encode();

    /**
     * Encode this persistable in to a {@link ByteBuffer}
     * @param buffer
     */
    void encode(ByteBuffer buffer);

    /**
     * Encode this persistable in to an output stream
     * @param outputStream
     * @throws IOException
     */
    void encode(OutputStream outputStream) throws IOException;

    /**
     * Decode the content of the given
     * byte array in to this persistable
     * @param decode
     */
    void decode(byte[] decode);

    /**
     * Decode from the given {@link ByteBuffer}
     * @param buffer
     */
    void decode(ByteBuffer buffer);

    /**
     * Decode from the given input stream
     * @param inputStream
     * @throws IOException
     */
    void decode(InputStream inputStream) throws IOException;

}
