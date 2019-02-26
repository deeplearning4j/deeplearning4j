package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.*;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.*;

import java.util.Arrays;
import java.util.List;

/**
 * SameDiff Recurrent Neural Network operations<br>
 * Accessible via {@link SameDiff#rnn()}<br>
 * See also {@link SDNN} (accessible via {@link SameDiff#nn()} for general neural network ops.<br>
 * See also {@link SDCNN} (accessible via {@link SameDiff#cnn()} for convolutional neural network ops.<br>
 *
 * @author Alex Black
 */
public class SDRNN extends SDOps {
    public SDRNN(SameDiff sameDiff) {
        super(sameDiff);
    }


    /**
     * The gru cell
     *
     * @param configuration the configuration to use
     * @return
     */
    public List<SDVariable> gru(GRUCellConfiguration configuration) {
        GRUCell c = new GRUCell(sd, configuration);
        return Arrays.asList(c.outputVariables());
    }

    /**
     * The gru cell
     *
     * @param baseName      the base name for the gru cell
     * @param configuration the configuration to use
     * @return
     */
    public List<SDVariable> gru(String baseName, GRUCellConfiguration configuration) {
        GRUCell c = new GRUCell(sd, configuration);
        return Arrays.asList(c.outputVariables(baseName));
    }


    /**
     * LSTM unit
     *
     * @param baseName      the base name for outputs
     * @param configuration the configuration to use
     * @return
     */
    public SDVariable lstmCell(String baseName, LSTMCellConfiguration configuration) {
        return new LSTMCell(sd, configuration).outputVariables(baseName)[0];
    }

    public List<SDVariable> lstmBlockCell(String name, LSTMBlockCellConfiguration configuration){
        SDVariable[] v = new LSTMBlockCell(sd, configuration).outputVariables(name);
        return Arrays.asList(v);
    }

    public List<SDVariable> lstmLayer(String name, LSTMConfiguration configuration){
        SDVariable[] v = new LSTMLayer(sd, configuration).outputVariables(name);
        return Arrays.asList(v);
    }

    /**
     * Simple recurrent unit
     *
     * @param configuration the configuration for the sru
     * @return
     */
    public SDVariable sru(SRUConfiguration configuration) {
        return new SRU(sd, configuration).outputVariables()[0];
    }

    /**
     * Simiple recurrent unit
     *
     * @param baseName      the base name to use for output variables
     * @param configuration the configuration for the sru
     * @return
     */
    public SDVariable sru(String baseName, SRUConfiguration configuration) {
        return new SRU(sd, configuration).outputVariables(baseName)[0];
    }

    /**
     * An sru cell
     *
     * @param configuration the configuration for the sru cell
     * @return
     */
    public SDVariable sruCell(SRUCellConfiguration configuration) {
        return new SRUCell(sd, configuration).outputVariables()[0];
    }

    /**
     * An sru cell
     *
     * @param baseName      the base name to  use for the output variables
     * @param configuration the configuration for the sru cell
     * @return
     */
    public SDVariable sruCell(String baseName, SRUCellConfiguration configuration) {
        return new SRUCell(sd, configuration).outputVariables(baseName)[0];
    }

}
