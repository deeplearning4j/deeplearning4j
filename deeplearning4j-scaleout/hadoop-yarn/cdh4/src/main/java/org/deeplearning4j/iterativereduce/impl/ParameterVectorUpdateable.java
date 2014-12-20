package org.deeplearning4j.iterativereduce.impl;

import org.deeplearning4j.iterativereduce.runtime.Updateable;
import org.deeplearning4j.scaleout.job.Job;

import java.io.IOException;
import java.nio.ByteBuffer;



public class ParameterVectorUpdateable implements Updateable<Job> {

    Job param_msg = null;

    public ParameterVectorUpdateable() {
    }

    public ParameterVectorUpdateable(Job g) {
        this.param_msg = g;
    }

    @Override
    public void fromBytes(ByteBuffer b) {

        b.rewind();


    }

    @Override
    public Job get() {
        return this.param_msg;
    }

    @Override
    public void set(Job t) {
        this.param_msg = t;
    }

    @Override
    public ByteBuffer toBytes() {
        byte[] bytes = null;

        ByteBuffer buf = ByteBuffer.wrap(bytes);

        return buf;
    }

    @Override
    public void fromString(String s) {

    }
}