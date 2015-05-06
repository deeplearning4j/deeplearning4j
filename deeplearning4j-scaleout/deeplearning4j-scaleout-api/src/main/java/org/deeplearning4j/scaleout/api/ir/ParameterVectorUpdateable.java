/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.scaleout.api.ir;

import org.apache.commons.compress.utils.IOUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;


/**
 * Parameter vecotr updateable
 * @author Adam Gibson
 */
public class ParameterVectorUpdateable implements Updateable<INDArray> {

    INDArray paramMessage = null;

    public ParameterVectorUpdateable() {
    }

    public ParameterVectorUpdateable(INDArray g) {
        this.paramMessage = g;
    }

    @Override
    public void fromBytes(ByteBuffer b) {
        b.rewind();
        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(b.array()));
        try {
            paramMessage = Nd4j.read(dis);
        } catch (IOException e) {
            e.printStackTrace();
        }

        IOUtils.closeQuietly(dis);

    }

    @Override
    public INDArray get() {
        return this.paramMessage;
    }

    @Override
    public void set(INDArray t) {
        this.paramMessage = t;
    }

    @Override
    public ByteBuffer toBytes() {
        byte[] bytes = paramMessage.data().asBytes();
        ByteBuffer buf = ByteBuffer.wrap(bytes);
        return buf;
    }

    @Override
    public void fromString(String s) {
        String[] split = s.split(" ");
        paramMessage = Nd4j.create(split.length);
        if(Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            for(int i = 0 ;i < split.length; i++) {
                paramMessage.putScalar(i,Double.valueOf(split[i]));
            }
        }
        else {
            for(int i = 0 ;i < split.length; i++) {
                paramMessage.putScalar(i,Float.valueOf(split[i]));
            }
        }
    }
}