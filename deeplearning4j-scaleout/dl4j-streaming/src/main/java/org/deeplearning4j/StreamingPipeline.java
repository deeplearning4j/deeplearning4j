package org.deeplearning4j;

/**
 * Created by agibsoncccc on 6/6/16.
 */
public class StreamingPipeline implements Pipeline {

    @Override
    public String[] inputUris() {
        return new String[0];
    }

    @Override
    public String[] outputUris() {
        return new String[0];
    }

    @Override
    public String[] datavecUris() {
        return new String[0];
    }
}
