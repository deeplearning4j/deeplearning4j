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

package org.ansj.util;

import java.io.IOException;
import java.io.Reader;

/**
 * 我又剽窃了下jdk...职业嫖客 为了效率这个流的操作是不支持多线程的,要么就是长时间不写这种东西了。发现好费劲啊 这个reader的特点。。只会输入
 * 句子不会输出\r\n .会有一个start来记录当前返回字符串。起始偏移量
 * 
 * @author ansj
 * 
 */
public class AnsjReader extends Reader {

    private Reader in;

    private char cb[];

    private static int defaultCharBufferSize = 8192;

    /**
     * Creates a buffering character-input stream that uses an input buffer of
     * the specified size.
     * 
     * @param in
     *            A Reader
     * @param sz
     *            Input-buffer size
     * 
     * @exception IllegalArgumentException
     *                If {@code sz <= 0}
     */
    public AnsjReader(Reader in, int sz) {
        super(in);
        if (sz <= 0)
            throw new IllegalArgumentException("Buffer size <= 0");
        this.in = in;
        cb = new char[sz];
    }

    /**
     * Creates a buffering character-input stream that uses a default-sized
     * input buffer.
     * 
     * @param in
     *            A Reader
     */
    public AnsjReader(Reader in) {
        this(in, defaultCharBufferSize);
    }

    /** Checks to make sure that the stream has not been closed */
    private void ensureOpen() throws IOException {
        if (in == null)
            throw new IOException("Stream closed");
    }

    /**
     * 为了功能的单一性我还是不实现了
     */
    @Override
    public int read(char cbuf[], int off, int len) throws IOException {
        throw new IOException("AnsjBufferedReader not support this interface! ");
    }

    private int start = 0;
    private int tempStart = 0;

    /**
     * 读取一行数据。ps 读取结果会忽略 \n \r
     */
    public String readLine() throws IOException {

        ensureOpen();

        StringBuilder sb = null;

        start = tempStart;

        firstRead = true;

        while (true) {

            tempLen = 0;
            ok = false;

            readString();
            // if (tempLen != 0)
            // System.out.println(new String(cb, tempOffe, tempLen));

            if (!isRead && (tempLen == 0 || len == 0)) {
                if (sb != null) {
                    return sb.toString();
                }
                return null;
            }

            if (!isRead) { // 如果不是需要读状态，那么返回
                tempStart += tempLen;
                if (sb == null) {
                    return new String(cb, tempOffe, tempLen);
                } else {
                    sb.append(cb, tempOffe, tempLen);
                    return sb.toString();
                }
            }

            if (tempLen == 0) {
                continue;
            }

            // 如果是需要读状态那么读取
            if (sb == null) {
                sb = new StringBuilder();
            }
            sb.append(cb, tempOffe, tempLen);
            tempStart += tempLen;
        }

    }

    int offe = 0;
    int len = 0;

    boolean isRead = false;
    boolean ok = false;
    boolean firstRead = true;

    int tempOffe;
    int tempLen;

    private void readString() throws IOException {

        if (offe <= 0) {
            if (offe == -1) {
                isRead = false;
                return;
            }

            len = in.read(cb);
            if (len <= 0) { // 说明到结尾了
                isRead = false;
                return;
            }
        }

        isRead = true;

        char c = 0;
        int i = offe;
        for (; i < len; i++) {
            c = cb[i];
            if (c != '\r' && c != '\n') {
                break;
            }
            if (!firstRead) {
                i++;
                tempStart++;
                offe = i;
                tempOffe = offe;
                isRead = false;
                return;
            }
            tempStart++;
            start++;
        }

        if (i == len) {
            isRead = true;
            offe = 0;
            return;
        }

        firstRead = false;

        offe = i;

        for (; i < len; i++) {
            c = cb[i];
            if (c == '\n' || c == '\r') {
                isRead = false;
                break;
            }
        }

        tempOffe = offe;
        tempLen = i - offe;

        if (i == len) {
            if (len < cb.length) { // 说明到结尾了
                isRead = false;
                offe = -1;
            } else {
                offe = 0;
            }
        } else {
            offe = i;
        }

    }

    @Override
    public void close() throws IOException {
        synchronized (lock) {
            if (in == null)
                return;
            try {
                in.close();
            } finally {
                in = null;
                cb = null;
            }
        }
    }

    public int getStart() {
        return this.start;
    }

}
