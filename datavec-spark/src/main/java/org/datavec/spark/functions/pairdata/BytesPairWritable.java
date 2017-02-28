/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.spark.functions.pairdata;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.nio.charset.Charset;

/**A Hadoop writable class for a pair of byte arrays, plus the original URIs (as Strings) of the files they came from
 * @author Alex Black
 */
public class BytesPairWritable implements Serializable, org.apache.hadoop.io.Writable {
    private byte[] first;
    private byte[] second;
    private String uriFirst;
    private String uriSecond;

    public BytesPairWritable() {}

    public BytesPairWritable(byte[] first, byte[] second, String uriFirst, String uriSecond) {
        this.first = first;
        this.second = second;
        this.uriFirst = uriFirst;
        this.uriSecond = uriSecond;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        int length1 = (first != null ? first.length : 0);
        int length2 = (second != null ? second.length : 0);
        byte[] s1Bytes = (uriFirst != null ? uriFirst.getBytes(Charset.forName("UTF-8")) : null);
        byte[] s2Bytes = (uriSecond != null ? uriSecond.getBytes(Charset.forName("UTF-8")) : null);
        int s1Len = (s1Bytes != null ? s1Bytes.length : 0);
        int s2Len = (s2Bytes != null ? s2Bytes.length : 0);
        dataOutput.writeInt(length1);
        dataOutput.writeInt(length2);
        dataOutput.writeInt(s1Len);
        dataOutput.writeInt(s2Len);
        if (first != null)
            dataOutput.write(first);
        if (second != null)
            dataOutput.write(second);
        if (s1Bytes != null)
            dataOutput.write(s1Bytes);
        if (s2Bytes != null)
            dataOutput.write(s2Bytes);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        int length1 = dataInput.readInt();
        int length2 = dataInput.readInt();
        int s1Len = dataInput.readInt();
        int s2Len = dataInput.readInt();
        if (length1 > 0) {
            first = new byte[length1];
            dataInput.readFully(first);
        }
        if (length2 > 0) {
            second = new byte[length2];
            dataInput.readFully(second);
        }
        if (s1Len > 0) {
            byte[] s1Bytes = new byte[s1Len];
            dataInput.readFully(s1Bytes);
            uriFirst = new String(s1Bytes, Charset.forName("UTF-8"));
        }
        if (s2Len > 0) {
            byte[] s2Bytes = new byte[s2Len];
            dataInput.readFully(s2Bytes);
            uriSecond = new String(s2Bytes, Charset.forName("UTF-8"));
        }
    }

    public byte[] getFirst() {
        return first;
    }

    public byte[] getSecond() {
        return second;
    }

    public String getUriFirst() {
        return uriFirst;
    }

    public String getUriSecond() {
        return uriSecond;
    }

    public void setFirst(byte[] first) {
        this.first = first;
    }

    public void setSecond(byte[] second) {
        this.second = second;
    }

    public void setUriFirst(String uriFirst) {
        this.uriFirst = uriFirst;
    }

    public void setUriSecond(String uriSecond) {
        this.uriSecond = uriSecond;
    }
}
