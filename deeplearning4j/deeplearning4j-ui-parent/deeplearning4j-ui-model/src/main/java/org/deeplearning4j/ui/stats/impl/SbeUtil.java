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

package org.deeplearning4j.ui.stats.impl;

import java.io.*;
import java.nio.charset.Charset;
import java.util.Map;

/**
 * Utilities for use in {@link SbeStatsInitializationReport} and {@link SbeStatsReport}
 *
 * @author Alex Black
 */
public class SbeUtil {

    public static final Charset UTF8 = Charset.forName("UTF-8");
    public static final byte[] EMPTY_BYTES = new byte[0]; //Also equivalent to "".getBytes(UTF8);

    private SbeUtil() {}

    public static int length(byte[] bytes) {
        if (bytes == null)
            return 0;
        return bytes.length;
    }

    public static int length(byte[][] bytes) {
        if (bytes == null)
            return 0;
        int count = 0;
        for (int i = 0; i < bytes.length; i++) {
            if (bytes[i] != null)
                count += bytes[i].length;
        }
        return count;
    }

    public static int length(byte[][][] bytes) {
        if (bytes == null)
            return 0;
        int count = 0;
        for (byte[][] arr : bytes) {
            count += length(arr);
        }
        return count;
    }

    public static int length(String str) {
        if (str == null)
            return 0;
        return str.length();
    }

    public static int length(String[] arr) {
        if (arr == null || arr.length == 0)
            return 0;
        int sum = 0;
        for (String s : arr)
            sum += length(s);
        return sum;
    }

    public static byte[] toBytes(boolean present, String str) {
        if (!present || str == null)
            return EMPTY_BYTES;
        return str.getBytes(UTF8);
    }

    public static byte[][] toBytes(boolean present, String[] str) {
        if (str == null)
            return null;
        byte[][] b = new byte[str.length][0];
        for (int i = 0; i < str.length; i++) {
            if (str[i] == null)
                continue;
            b[i] = toBytes(present, str[i]);
        }
        return b;
    }

    public static byte[][][] toBytes(Map<String, String> map) {
        if (map == null)
            return null;
        byte[][][] b = new byte[map.size()][2][0];
        int i = 0;
        for (Map.Entry<String, String> entry : map.entrySet()) {
            b[i][0] = toBytes(true, entry.getKey());
            b[i][1] = toBytes(true, entry.getValue());
            i++;
        }
        return b;
    }

    public static byte[] toBytesSerializable(Serializable serializable) {
        if (serializable == null)
            return EMPTY_BYTES;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(serializable);
        } catch (IOException e) {
            throw new RuntimeException("Unexpected IOException during serialization", e);
        }
        return baos.toByteArray();
    }

    public static Serializable fromBytesSerializable(byte[] bytes) {
        if (bytes == null || bytes.length == 0)
            return null;
        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        try (ObjectInputStream ois = new ObjectInputStream(bais)) {
            return (Serializable) ois.readObject();
        } catch (IOException e) {
            throw new RuntimeException("Unexpected IOException during deserialization", e);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
