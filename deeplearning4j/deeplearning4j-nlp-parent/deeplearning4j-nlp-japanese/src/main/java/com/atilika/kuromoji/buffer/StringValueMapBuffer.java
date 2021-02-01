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
package com.atilika.kuromoji.buffer;

import com.atilika.kuromoji.io.ByteBufferIO;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.TreeMap;

public class StringValueMapBuffer {

    private static final int INTEGER_BYTES = Integer.SIZE / Byte.SIZE;

    private static final int SHORT_BYTES = Short.SIZE / Byte.SIZE;

    private ByteBuffer buffer;

    public StringValueMapBuffer(TreeMap<Integer, String> features) {
        putMap(features);
    }

    public StringValueMapBuffer(InputStream is) throws IOException {
        buffer = ByteBufferIO.read(is);
    }

    private static int getMetaDataSize() {
        return INTEGER_BYTES;
    }

    public void putMap(TreeMap<Integer, String> input) {
        buffer = ByteBuffer.wrap(new byte[calculateSize(input) + getMetaDataSize()]);

        buffer.putInt(input.size());
        int position = getMetaDataSize();
        int address = position + input.size() * INTEGER_BYTES;


        for (Integer index : input.keySet()) {
            buffer.putInt(position, address);
            address = putString(address, input.get(index));
            position += INTEGER_BYTES;
        }
    }

    private int calculateSize(TreeMap<Integer, String> input) {
        int size = 0;
        for (String value : input.values()) {
            size += INTEGER_BYTES + value.getBytes(StandardCharsets.UTF_8).length + 2 * INTEGER_BYTES;
        }
        return size;
    }

    private int putString(int address, String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);

        buffer.position(address);
        // TODO: The length field in the entry (bytes.length) field can be optimized (shrunk) for most dictionary types.
        buffer.putShort((short) bytes.length);
        buffer.put(bytes);

        return address + SHORT_BYTES + bytes.length;
    }

    public String get(int i) {
        int address = buffer.getInt(i * INTEGER_BYTES + getMetaDataSize());
        return getString(address);
    }

    private String getString(int address) {
        int length = buffer.getShort(address);
        return new String(buffer.array(), address + SHORT_BYTES, length, StandardCharsets.UTF_8);
    }

    public void write(OutputStream os) throws IOException {
        ByteBufferIO.write(os, buffer);
    }

}
