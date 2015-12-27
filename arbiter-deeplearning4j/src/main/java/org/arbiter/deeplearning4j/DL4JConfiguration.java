package org.arbiter.deeplearning4j;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

import java.io.Serializable;

/**DL4JConfiguration: simple configuration method that contains the following:<br>
 * - MultiLayerConfiguration<br>
 * - Early stopping settings, OR number of epochs<br>
 * Note: if early stopping configuration is absent, a fixed number of epochs (default: 1) will be used.
 * If both early stopping and number of epochs is present: early stopping will be used.
 */
@AllArgsConstructor @Data
public class DL4JConfiguration implements Serializable {

    private MultiLayerConfiguration multiLayerConfiguration;
    private EarlyStoppingConfiguration earlyStoppingConfiguration;
    private Integer numEpochs;

}
