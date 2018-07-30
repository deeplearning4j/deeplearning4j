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

import org.apache.commons.io.FileUtils;

import java.io.*;

/**
 * Serialization utils for saving and reading serializable objects
 *
 * @author Adam Gibson
 */
public class SerializationUtils {

    protected SerializationUtils() {}

    @SuppressWarnings("unchecked")
    public static <T> T readObject(File file) {
        try {
            ObjectInputStream ois = new ObjectInputStream(FileUtils.openInputStream(file));
            T ret = (T) ois.readObject();
            ois.close();
            return ret;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Reads an object from the given input stream
     * @param is the input stream to read from
     * @return the read object
     */
    @SuppressWarnings("unchecked")
    public static <T> T readObject(InputStream is) {
        try {
            ObjectInputStream ois = new ObjectInputStream(is);
            T ret = (T) ois.readObject();
            ois.close();
            return ret;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }



    /**
     * Converts the given object to a byte array
     * @param toSave the object to save
     */
    public static byte[] toByteArray(Serializable toSave) {
        try {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream os = new ObjectOutputStream(bos);
            os.writeObject(toSave);
            byte[] ret = bos.toByteArray();
            os.close();
            return ret;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }


    /**
     * Writes the object to the output stream
     * THIS DOES NOT FLUSH THE STREAM
     * @param toSave the object to save
     * @param writeTo the output stream to write to
     */
    public static void writeObject(Serializable toSave, OutputStream writeTo) {
        try {
            ObjectOutputStream os = new ObjectOutputStream(writeTo);
            os.writeObject(toSave);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    public static void saveObject(Object toSave, File saveTo) {
        try {
            OutputStream os1 = FileUtils.openOutputStream(saveTo);
            ObjectOutputStream os = new ObjectOutputStream(os1);
            os.writeObject(toSave);
            os.flush();
            os.close();
            os1.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }
}
