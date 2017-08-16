package org.datavec.local.transforms;

import org.datavec.api.transform.Transform;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import org.datavec.api.transform.transform.condition.ConditionalCopyValueTransform;
import org.datavec.api.transform.transform.string.AppendStringColumnTransform;
import tech.tablesaw.api.CategoryColumn;
import tech.tablesaw.api.Table;

/**
 * Created by agibsonccc on 11/11/16.
 */
public class DataFrameOps {

    public static void performTransform(Transform transform, Table table) {
        if (transform instanceof AppendStringColumnTransform) {
            AppendStringColumnTransform appendStringColumnTransform = (AppendStringColumnTransform) transform;
            CategoryColumn categoryColumn = (CategoryColumn) table.column(appendStringColumnTransform.getColumnName());
            table.addColumn(categoryColumn);
        } else if (transform instanceof CategoricalToIntegerTransform) {

        } else if (transform instanceof CategoricalToOneHotTransform) {

        } else if (transform instanceof ConditionalCopyValueTransform) {

        }


    }

}
