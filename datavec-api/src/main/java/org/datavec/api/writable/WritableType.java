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

/**
 * Created by Alex on 30/05/2017.
 */
public enum WritableType {
    Boolean,
    Byte,
    Double,
    Float,
    Int,
    Long,
    Null,
    Text,
    NDArray,
    Image;

    public boolean isCoreWritable(){
        switch (this){
            case NDArray:
            case Image:
                return false;
            default:
                return true;
        }
    }

    public short typeIdx(){
        return (short)this.ordinal();
    }

    public Class<? extends Writable> getWritableClass(){
        switch (this){
            case Boolean:
                return BooleanWritable.class;
            case Byte:
                return ByteWritable.class;
            case Double:
                return DoubleWritable.class;
            case Float:
                return FloatWritable.class;
            case Int:
                return IntWritable.class;
            case Long:
                return LongWritable.class;
            case Null:
                return NullWritable.class;
            case Text:
                return Text.class;
            case NDArray:
            case Image:
            default:
                return null;
        }
    }

}
