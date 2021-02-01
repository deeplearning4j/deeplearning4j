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
package com.atilika.kuromoji.compile;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;

public class ConnectionCostsCompiler implements Compiler {

    private static final int SHORT_BYTES = Short.SIZE / Byte.SIZE;

    private OutputStream output;

    private int cardinality;

    private int bufferSize;

    private ShortBuffer costs;

    public ConnectionCostsCompiler(OutputStream output) {
        this.output = output;
    }

    public void readCosts(InputStream input) throws IOException {
        BufferedReader lineReader = new BufferedReader(new InputStreamReader(input));

        String line = lineReader.readLine();
        String[] cardinalities = line.split("\\s+");

        assert cardinalities.length == 2;

        int forwardSize = Integer.parseInt(cardinalities[0]);
        int backwardSize = Integer.parseInt(cardinalities[1]);

        assert forwardSize == backwardSize;
        assert forwardSize > 0;
        assert backwardSize > 0;

        cardinality = backwardSize;
        bufferSize = forwardSize * backwardSize;
        costs = ShortBuffer.allocate(bufferSize);

        while ((line = lineReader.readLine()) != null) {
            String[] fields = line.split("\\s+");

            assert fields.length == 3;

            short forwardId = Short.parseShort(fields[0]);
            short backwardId = Short.parseShort(fields[1]);
            short cost = Short.parseShort(fields[2]);

            putCost(forwardId, backwardId, cost);
        }
    }

    public void putCost(short forwardId, short backwardId, short cost) {
        this.costs.put(backwardId + forwardId * cardinality, cost);
    }

    @Override
    public void compile() throws IOException {
        DataOutputStream dataOutput = new DataOutputStream(new BufferedOutputStream(output));

        dataOutput.writeInt(cardinality);
        dataOutput.writeInt(bufferSize * SHORT_BYTES);

        ByteBuffer byteBuffer = ByteBuffer.allocate(costs.array().length * SHORT_BYTES);

        for (short cost : this.costs.array()) {
            byteBuffer.putShort(cost);
        }

        WritableByteChannel channel = Channels.newChannel(dataOutput);

        byteBuffer.flip();
        channel.write(byteBuffer);
        dataOutput.close();
    }

    public int getCardinality() {
        return cardinality;
    }

    public ShortBuffer getCosts() {
        return costs;
    }
}
