package org.datavec.api.transform.reduce;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

/**A reducer aggregates or combines a set of examples into a single List<Writable>
 */
public interface IReducer extends Serializable {

    void setInputSchema(Schema schema);

    Schema getInputSchema();

    Schema transform(Schema schema);

    List<Writable> reduce(List<List<Writable>> examplesList);

    List<String> getKeyColumns();

}
