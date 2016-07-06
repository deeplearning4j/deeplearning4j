package org.nd4j.etl4j.api.transform.transform;

import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.schema.Schema;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**Map the values in a single column to new values.
 * For example: string -> string, or empty -> x type transforms for a single column
 */
@EqualsAndHashCode(callSuper = true)
@Data
public abstract class BaseColumnTransform extends BaseTransform {

    protected final String columnName;
    protected int columnNumber = -1;
    private static final long serialVersionUID = 0L;

    public BaseColumnTransform(String columnName) {
        this.columnName = columnName;
    }

    @Override
    public void setInputSchema(Schema inputSchema){
        this.inputSchema = inputSchema;
        columnNumber = inputSchema.getIndexOfColumn(columnName);
    }

    @Override
    public Schema transform(Schema schema) {
        if(columnNumber == -1) throw new IllegalStateException("columnNumber == -1 -> setInputSchema not called?");
        List<ColumnMetaData> oldMeta = schema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(oldMeta.size());

        Iterator<ColumnMetaData> typesIter = oldMeta.iterator();

        int i=0;
        while(typesIter.hasNext()){
            ColumnMetaData t = typesIter.next();
            if(i++ == columnNumber){
                newMeta.add(getNewColumnMetaData(t));
            } else {
                newMeta.add(t);
            }
        }

        return schema.newSchema(new ArrayList<>(schema.getColumnNames()),newMeta);
    }

    public abstract ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnType);

    @Override
    public List<Writable> map(List<Writable> writables) {
        if(writables.size() != inputSchema.numColumns() ){
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size() + ") does not " +
                    "match expected number of elements (schema: " + inputSchema.numColumns() + "). Transform = " + toString());
        }
        int n = writables.size();
        List<Writable> out = new ArrayList<>(n);

        int i=0;
        for(Writable w : writables){
            if(i++ == columnNumber){
                Writable newW = map(w);
                out.add(newW);
            } else {
                out.add(w);
            }
        }

        return out;
    }



    public abstract Writable map(Writable columnWritable);

    @Override
    public abstract String toString();

}
