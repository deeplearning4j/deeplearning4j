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

package org.datavec.api.writable;

/**
 * Created by Alex on 30/05/2017.
 */
public enum WritableType {
    Boolean, Byte, Double, Float, Int, Long, Null, Text, NDArray, Image,Arrow,Bytes;

    //NOTE TO DEVELOPERS:
    //In the current implementation, the order (ordinal idx) for the WritableType values matters.
    //New writables can be added to the end of the list, but not between exiting types, as this will change the
    //ordinal value for all writable types that follow, which will mess up serialization in some cases (like Spark
    // sequence and map files)
    //Alternatively, modify WritableType.typeIdx() to ensure backward compatibility


    /**
     *
     * @return True if Writable is defined in datavec-api, false otherwise
     */
    public boolean isCoreWritable() {
        switch (this) {
            case Image:
            case Arrow:
                return false;
            default:
                return true;
        }
    }

    /**
     * Return a unique type index for the given writable
     *
     * @return Type index for the writable
     */
    public short typeIdx() {
        return (short) this.ordinal();
    }

    /**
     * Return the class of the implementation corresponding to each WritableType.
     * Note that if {@link #isCoreWritable()} returns false, null will be returned by this method.
     *
     * @return Class for the given WritableType
     */
    public Class<? extends Writable> getWritableClass() {
        switch (this) {
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
                return NDArrayWritable.class;
            case Bytes:
                return ByteWritable.class;
            case Image:
            case Arrow:
            default:
                return null;
        }
    }

}
