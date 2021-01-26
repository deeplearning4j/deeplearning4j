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
package com.atilika.kuromoji.dict;

import com.atilika.kuromoji.io.ByteBufferIO;
import com.atilika.kuromoji.util.KuromojiBinFilesFetcher;
import com.atilika.kuromoji.util.ResourceResolver;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ShortBuffer;

public class ConnectionCosts {

    //    public static final String CONNECTION_COSTS_FILENAME = "connectionCosts.bin";
    public static final String CONNECTION_COSTS_FILENAME = new File(KuromojiBinFilesFetcher.getKuromojiRoot(),
            "connectionCosts.bin").getAbsolutePath();

    private int size;

    private ShortBuffer costs;

    public ConnectionCosts(int size, ShortBuffer costs) {
        this.size = size;
        this.costs = costs;
    }

    public int get(int forwardId, int backwardId) {
        return costs.get(backwardId + forwardId * size);
    }

    public static ConnectionCosts newInstance(ResourceResolver resolver) throws IOException {
        return read(resolver.resolve(CONNECTION_COSTS_FILENAME));
    }

    private static ConnectionCosts read(InputStream input) throws IOException {
        DataInputStream dataInput = new DataInputStream(new BufferedInputStream(input));

        int size = dataInput.readInt();

        ByteBuffer byteBuffer = ByteBufferIO.read(dataInput);
        ShortBuffer costs = byteBuffer.asShortBuffer();

        return new ConnectionCosts(size, costs);
    }
}
