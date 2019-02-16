package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.GRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUConfiguration;

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
    public SDVariable gru(GRUCellConfiguration configuration) {
        return new GRUCell(sd, configuration).outputVariables()[0];
    }

    /**
     * The gru cell
     *
     * @param baseName      the base name for the gru cell
     * @param configuration the configuration to use
     * @return
     */
    public SDVariable gru(String baseName, GRUCellConfiguration configuration) {
        return new GRUCell(sd, configuration).outputVariables(baseName)[0];
    }


    /**
     * LSTM unit
     *
     * @param baseName      the base name for outputs
     * @param configuration the configuration to use
     * @return
     */
    public SDVariable lstm(String baseName, LSTMCellConfiguration configuration) {
        return new LSTMCell(sd, configuration).outputVariables(baseName)[0];
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
