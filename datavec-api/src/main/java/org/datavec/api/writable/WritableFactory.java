/*
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

package org.datavec.api.writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by Alex on 29/05/2017.
 */
public class WritableFactory {

    private static WritableFactory INSTANCE = new WritableFactory();

    private Map<Short, Class<? extends Writable>> map = new ConcurrentHashMap<>();
    private Map<Short, Constructor<? extends Writable>> constructorMap = new ConcurrentHashMap<>();

    private WritableFactory(){

        for(WritableType wt : WritableType.values()){
            if(wt.isCoreWritable()){
                registerWritableType((short)wt.ordinal(), wt.getWritableClass());
            }
        }
    }

    public static WritableFactory getInstance(){
        return INSTANCE;
    }

    public void registerWritableType(short writableTypeKey, Class<? extends Writable> writableClass){
        if (map.containsKey(writableTypeKey)) {
            throw new UnsupportedOperationException("Key " + writableTypeKey + " is already registered to type "
                    + map.get(writableTypeKey) + " and cannot be registered to " + writableClass);
        }

        Constructor<? extends Writable> c;
        try{
            c = writableClass.getDeclaredConstructor();
        } catch (NoSuchMethodException e){
            throw new RuntimeException("Cannot find no-arg constructor for class " + writableClass);
        }

        map.put(writableTypeKey, writableClass);
        constructorMap.put(writableTypeKey, c);
    }

    public Writable newWritable(short writableTypeKey){
        Constructor<? extends Writable> c = constructorMap.get(writableTypeKey);
        if(c == null){
            throw new IllegalStateException("Unknown writable key: " + writableTypeKey);
        }
        try{
            return c.newInstance();
        } catch (Exception e){
            throw new RuntimeException("Could not create new Writable instance");
        }
    }

    public void writeWithType(Writable w, DataOutput dataOutput) throws IOException {
        w.writeType(dataOutput);
        w.write(dataOutput);
    }

    public Writable readWithType(DataInput dataInput) throws IOException {
        Writable w = newWritable(dataInput.readShort());
        w.readFields(dataInput);
        return  w;
    }

}
