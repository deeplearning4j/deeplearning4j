package org.datavec.camel.component.csv.marshaller;

import org.apache.camel.Exchange;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.ListStringSplit;
import org.datavec.camel.component.DataVecMarshaller;

import java.util.List;

/**
 * Marshals List<List<String>>
 *
 *     @author Adam Gibson
 */
public class ListStringInputMarshaller implements DataVecMarshaller {
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
