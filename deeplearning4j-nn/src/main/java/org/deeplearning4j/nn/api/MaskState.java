package org.deeplearning4j.nn.api;

/**
 * MaskState: specifies whether a mask should be applied or not.
 *
 * Masks should not be applied in all cases, depending on the network configuration - for example input Dense -> RNN
 * -> Dense -> RnnOutputLayer<br>
 * The first dense layer should be masked (using the input mask) whereas the second shouldn't be, as it has valid data
 * coming from the RNN layer below. For variable length situations like that, the masking can be implemented using the
 * label mask, which will backpropagate 0s for those time steps.<br>
 * In other cases, the *should* be applied - for example, input -> BidirectionalRnn -> Dense -> Output. In such a case,
 * the dense layer should be masked using the input mask.<br>
 * <p>
 * Essentially: Active = apply mask to activations and errors.<br>
 * Passthrough = feed forward the input mask (if/when necessary) but don't actually apply it.<br>
 *
 * @author Alex Black
 */
public enum MaskState {
    Active, Passthrough
}
