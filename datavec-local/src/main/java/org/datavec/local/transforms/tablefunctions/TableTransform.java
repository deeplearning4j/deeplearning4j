package org.datavec.local.transforms.tablefunctions;

import org.datavec.api.transform.Transform;
import tech.tablesaw.api.Table;

/**
 * A function for transforming a given input table
 * via the given transformNonTimeSeries.
 * This maps the given {@link Transform}
 * to a {@link Table} operation
 *
 * @author Adam Gibson
 */
public interface TableTransform {

    /**
     * Runs the given transformNonTimeSeries on
     * a table
     * @param transform the transformNonTimeSeries to run
     * @return the result table
     */
    Table transform(Table input, Transform transform);

}
