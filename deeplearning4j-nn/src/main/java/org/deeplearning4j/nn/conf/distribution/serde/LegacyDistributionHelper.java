package org.deeplearning4j.nn.conf.distribution.serde;

import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

/**
 * A dummy helper "distribution" for deserializing distributions in legacy/different JSON format.
 * Used in conjuction with {@link LegacyDistributionDeserializer} to provide backward compatability;
 * see that class for details.
 *
 * @author Alex Black
 */
@JsonDeserialize(using = LegacyDistributionDeserializer.class)
public class LegacyDistributionHelper extends Distribution {

    private LegacyDistributionHelper() {

    }

}
