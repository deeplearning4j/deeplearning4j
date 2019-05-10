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

package org.deeplearning4j.arbiter.optimize.generator.util;

import org.nd4j.linalg.function.Supplier;

import java.io.*;

public class SerializedSupplier<T> implements Serializable, Supplier<T> {

    private byte[] asBytes;

    public SerializedSupplier(T obj){
        try(ByteArrayOutputStream baos = new ByteArrayOutputStream(); ObjectOutputStream oos = new ObjectOutputStream(baos)){
            oos.writeObject(obj);
            oos.flush();
            oos.close();
            asBytes = baos.toByteArray();
        } catch (Exception e){
            throw new RuntimeException("Error serializing object - must be serializable",e);
        }
    }

    @Override
    public T get() {
        try(ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(asBytes))){
            return (T)ois.readObject();
        } catch (Exception e){
            throw new RuntimeException("Error deserializing object",e);
        }
    }
}
