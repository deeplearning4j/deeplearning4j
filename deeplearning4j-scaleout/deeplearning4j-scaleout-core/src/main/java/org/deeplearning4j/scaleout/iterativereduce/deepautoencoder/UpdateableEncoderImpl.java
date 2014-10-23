package org.deeplearning4j.scaleout.iterativereduce.deepautoencoder;

import org.deeplearning4j.models.featuredetectors.autoencoder.SemanticHashing;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
import org.deeplearning4j.util.SerializationUtils;

import java.io.DataOutputStream;
import java.nio.ByteBuffer;

/**
 * Holds updates for a deep autoencoder
 * @author Adam Gibson
 */
public class UpdateableEncoderImpl implements Updateable<SemanticHashing> {

    protected SemanticHashing masterResults;

    public UpdateableEncoderImpl(SemanticHashing masterResults) {
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
    public SemanticHashing get() {
        return masterResults;
    }

    @Override
    public void set(SemanticHashing semanticHashing) {
        this.masterResults = semanticHashing;

    }

    @Override
    public void write(DataOutputStream dos) {
        SerializationUtils.writeObject(masterResults,dos);
    }
}
