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

package org.deeplearning4j.ui.storage.impl;

import lombok.Data;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.ui.stats.impl.SbeUtil;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;

/**
 * Created by Alex on 14/12/2016.
 */
@Data
public class JavaStorageMetaData implements StorageMetaData {

    private long timeStamp;
    private String sessionID;
    private String typeID;
    private String workerID;
    private String initTypeClass;
    private String updateTypeClass;
    //Store serialized; saves class exceptions if we don't have the right class, and don't care about deserializing
    // on this machine, right now
    private byte[] extraMeta;

    public JavaStorageMetaData() {
        //No arg constructor for serialization/deserialization
    }

    public JavaStorageMetaData(long timeStamp, String sessionID, String typeID, String workerID, Class<?> initType,
                    Class<?> updateType) {
        this(timeStamp, sessionID, typeID, workerID, (initType != null ? initType.getName() : null),
                        (updateType != null ? updateType.getName() : null));
    }

    public JavaStorageMetaData(long timeStamp, String sessionID, String typeID, String workerID, String initTypeClass,
                    String updateTypeClass) {
        this(timeStamp, sessionID, typeID, workerID, initTypeClass, updateTypeClass, null);
    }

    public JavaStorageMetaData(long timeStamp, String sessionID, String typeID, String workerID, String initTypeClass,
                    String updateTypeClass, Serializable extraMetaData) {
        this.timeStamp = timeStamp;
        this.sessionID = sessionID;
        this.typeID = typeID;
        this.workerID = workerID;
        this.initTypeClass = initTypeClass;
        this.updateTypeClass = updateTypeClass;
        this.extraMeta = (extraMetaData == null ? null : SbeUtil.toBytesSerializable(extraMetaData));
    }

    @Override
    public int encodingLengthBytes() {
        //TODO - presumably a more efficient way to do this
        byte[] encoded = encode();
        return encoded.length;
    }

    @Override
    public byte[] encode() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(this);
        } catch (IOException e) {
            throw new RuntimeException(e); //Should never happen
        }
        return baos.toByteArray();
    }

    @Override
    public void encode(ByteBuffer buffer) {
        buffer.put(encode());
    }

    @Override
    public void encode(OutputStream outputStream) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(outputStream)) {
            oos.writeObject(this);
        }
    }

    @Override
    public void decode(byte[] decode) {
        JavaStorageMetaData r;
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(decode))) {
            r = (JavaStorageMetaData) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e); //Should never happen
        }

        Field[] fields = JavaStorageMetaData.class.getDeclaredFields();
        for (Field f : fields) {
            f.setAccessible(true);
            try {
                f.set(this, f.get(r));
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e); //Should never happen
            }
        }
    }

    @Override
    public void decode(ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        decode(bytes);
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        decode(IOUtils.toByteArray(inputStream));
    }

    @Override
    public Serializable getExtraMetaData() {
        return null;
    }
}
