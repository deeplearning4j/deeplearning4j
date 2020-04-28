package org.deeplearning4j.nn.conf;

/**
 * CNN2DFormat defines the format of the activations (including input images) in to and out of all 2D convolution layers in
 * Deeplearning4j. Default value is NCHW.<br>
 * <br>
 * NCHW = "channels first" - arrays of shape [minibatch, channels, height, width]<br>
 * NHWC = "channels last" - arrays of shape [minibatch, height, width, channels]<br>
 *
 * @author Alex Black
 */
public enum CNN2DFormat implements DataFormat {
    NCHW,
    NHWC;

    /**
     * Returns a string that explains the dimensions:<br>
     * NCHW -> returns "[minibatch, channels, height, width]"<br>
     * NHWC -> returns "[minibatch, height, width, channels]"
     */
    public String dimensionNames(){
        switch (this){
            case NCHW:
                return "[minibatch, channels, height, width]";
            case NHWC:
                return "[minibatch, height, width, channels]";
            default:
                throw new IllegalStateException("Unknown enum: " + this);   //Should never happen
        }
    }
}
