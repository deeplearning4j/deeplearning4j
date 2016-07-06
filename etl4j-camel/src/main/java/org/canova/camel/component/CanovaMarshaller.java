package org.canova.camel.component;

import org.apache.camel.Exchange;
import org.canova.api.split.InputSplit;

/**
 * Marshals na exchange in to an input split
 * @author Adam Gibson
 */
public interface CanovaMarshaller {


    /**
     *
     * @param exchange
     * @return
     */
     InputSplit getSplit(Exchange exchange);

}
