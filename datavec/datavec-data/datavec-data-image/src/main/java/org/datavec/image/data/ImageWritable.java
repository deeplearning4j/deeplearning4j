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

package org.datavec.image.data;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameConverter;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableFactory;
import org.datavec.api.writable.WritableType;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.Buffer;

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

    public ImageWritable() {
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
    
    public int getWidth() {
        return frame.imageWidth;
    }
    
    public int getHeight() {
        return frame.imageHeight;
    }
    
    public int getDepth() {
        return frame.imageDepth;
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

    @Override
    public WritableType getType() {
        return WritableType.Image;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof ImageWritable) {
            Frame f2 = ((ImageWritable) obj).getFrame();

            Buffer[] b1 = this.frame.image;
            Buffer[] b2 = f2.image;

            if (b1.length != b2.length)
                return false;

            for (int i = 0; i < b1.length; i++) {
                if (!b1[i].equals(b2[i]))
                    return false;
            }

            return true;
        } else {
            return false;
        }
    }
}
