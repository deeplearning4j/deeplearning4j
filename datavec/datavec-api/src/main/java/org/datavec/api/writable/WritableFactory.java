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

package org.datavec.api.writable;

import lombok.NonNull;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Factory class for creating and saving writable objects to/from DataInput and DataOutput
 *
 * @author Alex Black
 */
public class WritableFactory {

    private static WritableFactory INSTANCE = new WritableFactory();

    private Map<Short, Class<? extends Writable>> map = new ConcurrentHashMap<>();
    private Map<Short, Constructor<? extends Writable>> constructorMap = new ConcurrentHashMap<>();

    private WritableFactory() {
        for (WritableType wt : WritableType.values()) {
            if (wt.isCoreWritable()) {
                registerWritableType((short) wt.ordinal(), wt.getWritableClass());
            }
        }
    }

    /**
     * @return Singleton WritableFactory instance
     */
    public static WritableFactory getInstance() {
        return INSTANCE;
    }

    /**
     * Register a writable class with a specific key (as a short). Note that key values must be unique for each type of
     * Writable, as they are used as type information in certain types of serialisation. Consequently, an exception will
     * be thrown If the key value is not unique or is already assigned.<br>
     * Note that in general, this method needs to only be used for custom Writable types; Care should be taken to ensure
     * that the given key does not change once assigned.
     *
     * @param writableTypeKey Key for the Writable
     * @param writableClass   Class for the given key. Must have a no-arg constructor
     */
    public void registerWritableType(short writableTypeKey, @NonNull Class<? extends Writable> writableClass) {
        if (map.containsKey(writableTypeKey)) {
            throw new UnsupportedOperationException("Key " + writableTypeKey + " is already registered to type "
                            + map.get(writableTypeKey) + " and cannot be registered to " + writableClass);
        }

        Constructor<? extends Writable> c;
        try {
            c = writableClass.getDeclaredConstructor();
        } catch (NoSuchMethodException e) {
            throw new RuntimeException("Cannot find no-arg constructor for class " + writableClass);
        }

        map.put(writableTypeKey, writableClass);
        constructorMap.put(writableTypeKey, c);
    }

    /**
     * Create a new writable instance (using reflection) given the specified key
     *
     * @param writableTypeKey Key to create a new writable instance for
     * @return A new (empty/default) Writable instance
     */
    public Writable newWritable(short writableTypeKey) {
        Constructor<? extends Writable> c = constructorMap.get(writableTypeKey);
        if (c == null) {
            throw new IllegalStateException("Unknown writable key: " + writableTypeKey);
        }
        try {
            return c.newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Could not create new Writable instance");
        }
    }

    /**
     * A convenience method for writing a given Writable  object to a DataOutput. The key is 1st written (a single short)
     * followed by the value from writable.
     *
     * @param w          Writable value
     * @param dataOutput DataOutput to write both key and value to
     * @throws IOException If an error occurs during writing to the DataOutput
     */
    public void writeWithType(Writable w, DataOutput dataOutput) throws IOException {
        w.writeType(dataOutput);
        w.write(dataOutput);
    }

    /**
     * Read a Writable From the DataInput, where the Writable was previously written using {@link #writeWithType(Writable, DataOutput)}
     *
     * @param dataInput DataInput to read the Writable from
     * @return Writable from the DataInput
     * @throws IOException In an error occurs during reading
     */
    public Writable readWithType(DataInput dataInput) throws IOException {
        Writable w = newWritable(dataInput.readShort());
        w.readFields(dataInput);
        return w;
    }

}
