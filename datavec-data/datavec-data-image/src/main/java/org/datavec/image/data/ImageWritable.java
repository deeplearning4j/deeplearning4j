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
package org.datavec.image.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameConverter;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableFactory;
import org.datavec.api.writable.WritableType;

/**
 * Wraps a {@link Frame} to allow serialization within this framework.
 * Frame objects can be converted back and forth easily to and from classes
 * used by Android, Java 2D, OpenCV, FFmpeg, and others.
 *
 * @author saudet
 * @see Frame
 * @see FrameConverter
 */
public class ImageWritable implements Writable {
    static {
        WritableFactory.getInstance().registerWritableType(WritableType.Image.typeIdx(), ImageWritable.class);
    }

    protected Frame frame;

    public ImageWritable(){
        //No-arg cosntructor for reflection-based creation of ImageWritable objects
    }

    public ImageWritable(Frame frame) {
        this.frame = frame;
    }

    public Frame getFrame() {
        return frame;
    }

    public void setFrame(Frame frame) {
        this.frame = frame;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        out.writeShort(WritableType.Image.typeIdx());
    }

    @Override
    public double toDouble() {
        throw new UnsupportedOperationException();
    }

    @Override
    public float toFloat() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int toInt() {
        throw new UnsupportedOperationException();
    }

    @Override
    public long toLong() {
        throw new UnsupportedOperationException();
    }

}
