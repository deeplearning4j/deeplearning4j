package org.deeplearning4j.iterativereduce.impl;

import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.iterativereduce.runtime.Updateable;
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
        if(Nd4j.dataType() == DataBuffer.DOUBLE) {
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