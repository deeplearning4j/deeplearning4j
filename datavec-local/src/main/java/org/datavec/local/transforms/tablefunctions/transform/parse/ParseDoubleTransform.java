package org.datavec.local.transforms.tablefunctions.transform.parse;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;

/**
 *  Convert string writables to doubles
 *
 *  @author Adam GIbson
 */
public class ParseDoubleTransform extends BaseTransform {

    @Override
    public String toString() {
        return getClass().getName();
    }

    /**
     * Get the output schema for this transformation, given an input schema
     *
     * @param inputSchema
     */
    @Override
    public Schema transform(Schema inputSchema) {
       Schema.Builder newSchema = new Schema.Builder();
       for(int i = 0; i < inputSchema.numColumns(); i++) {
           if(inputSchema.getType(i) == ColumnType.String) {
               newSchema.addColumnDouble(inputSchema.getMetaData(i).getName());
           }
           else
               newSchema.addColumn(inputSchema.getMetaData(i));

       }
        return newSchema.build();
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        List<Writable> transform = new ArrayList<>();
        for(Writable w : writables){
            if(w instanceof Text){
                transform.add(new DoubleWritable(w.toDouble()));
            } else {
                transform.add(w);
            }
        }
        return transform;
    }
}
