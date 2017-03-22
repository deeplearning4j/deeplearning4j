package org.deeplearning4j.nn.conf;

/** Defines the type of backpropagation. 'Standard' setting (default) is used
 * for training most networks (MLP, CNN, etc)
 * In the context of recurrent neural networks, Standard means 
 * @author Alex
 *
 */
public enum BackpropType {
    /** Default option. Used for training most networks, including MLP, DBNs, CNNs etc.*/
    Standard,
    /** Truncated BackPropagation Through Time. Only applicable in context of
     * training networks with recurrent neural network layers such as GravesLSTM
     */
    TruncatedBPTT
}
