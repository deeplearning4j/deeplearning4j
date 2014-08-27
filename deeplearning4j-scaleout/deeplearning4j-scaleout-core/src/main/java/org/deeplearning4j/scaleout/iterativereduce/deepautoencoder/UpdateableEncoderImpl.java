package org.deeplearning4j.scaleout.iterativereduce.deepautoencoder;

import org.deeplearning4j.models.featuredetectors.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
import org.deeplearning4j.util.SerializationUtils;

import java.io.DataOutputStream;
import java.nio.ByteBuffer;

/**
 * Holds updates for a deep autoencoder
 * @author Adam Gibson
 */
public class UpdateableEncoderImpl implements Updateable<DeepAutoEncoder> {

    protected DeepAutoEncoder masterResults;

    public UpdateableEncoderImpl(DeepAutoEncoder masterResults) {
        this.masterResults = masterResults;
    }


    @Override
    public ByteBuffer toBytes() {
        return null;
    }

    @Override
    public void fromBytes(ByteBuffer b) {

    }

    @Override
    public void fromString(String s) {

    }

    @Override
    public DeepAutoEncoder get() {
        return masterResults;
    }

    @Override
    public void set(DeepAutoEncoder deepAutoEncoder) {
        this.masterResults = deepAutoEncoder;

    }

    @Override
    public void write(DataOutputStream dos) {
        SerializationUtils.writeObject(masterResults,dos);
    }
}
