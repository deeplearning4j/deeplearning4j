package org.canova.camel.component.csv.marshaller;

import org.apache.camel.Exchange;
import org.canova.api.split.InputSplit;
import org.canova.api.split.ListStringSplit;
import org.canova.camel.component.CanovaMarshaller;

import java.util.List;

/**
 * Marshals List<List<String>>
 *
 *     @author Adam Gibson
 */
public class ListStringInputMarshaller implements CanovaMarshaller {
    /**
     * @param exchange
     * @return
     */
    @Override
    public InputSplit getSplit(Exchange exchange) {
        List<List<String>> data = (List<List<String>>) exchange.getIn().getBody();
        InputSplit listSplit = new ListStringSplit(data);
        return listSplit;
    }
}
