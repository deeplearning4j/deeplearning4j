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

package org.nd4j.linalg.util;


import lombok.NonNull;

/**
 * Stronger 64-bit hash function helper, as described here: http://www.javamex.com/tutorials/collections/strong_hash_code_implementation.shtml
 *
 * @author raver119@gmail.com
 */
public class HashUtil {
    private static final long[] byteTable = createLookupTable();
    private static final long HSTART = 0xBB40E64DA205B064L;
    private static final long HMULT = 7664345821815920749L;

    private static long[] createLookupTable() {
        long[] byteTable = new long[256];
        long h = 0x544B2FBACAAF1684L;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 31; j++) {
                h = (h >>> 7) ^ h;
                h = (h << 11) ^ h;
                h = (h >>> 10) ^ h;
            }
            byteTable[i] = h;
        }
        return byteTable;
    }

    /**
     * This method returns long hash for a given bytes array
     *
     * @param data
     * @return
     */
    public static long getLongHash(@NonNull byte[] data) {
        long h = HSTART;
        final long hmult = HMULT;
        final long[] ht = byteTable;
        for (int len = data.length, i = 0; i < len; i++) {
            h = (h * hmult) ^ ht[data[i] & 0xff];
        }
        return h;
    }

    /**
     * This method returns long hash for a given string
     *
     * @param string
     * @return
     */
    public static long getLongHash(@NonNull String string) {
        long h = HSTART;
        final long hmult = HMULT;
        final long[] ht = byteTable;
        final int len = string.length();
        for (int i = 0; i < len; i++) {
            char ch = string.charAt(i);
            h = (h * hmult) ^ ht[ch & 0xff];
            h = (h * hmult) ^ ht[(ch >>> 8) & 0xff];
        }
        return h;
    }
}
