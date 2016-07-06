package org.nd4j.etl4j.api.transform.filter;

import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.schema.Schema;

import java.util.List;

/**Abstract class for filtering examples based on the values in a single column
 */
public abstract class BaseColumnFilter implements Filter {

    protected Schema schema;
    protected final String column;
    protected int columnIdx;

    protected BaseColumnFilter(String column){
        this.column = column;
    }

    @Override
    public boolean removeExample(List<Writable> writables) {
        return removeExample(writables.get(columnIdx));
    }

    @Override
    public boolean removeSequence(List<List<Writable>> sequence) {
        for(List<Writable> c : sequence){
            if(removeExample(c)) return true;
        }
        return false;
    }

    @Override
    public void setInputSchema(Schema schema) {
        this.schema = schema;
        this.columnIdx = schema.getIndexOfColumn(column);
    }

    /** Should the example or sequence be removed, based on the values from the specified column? */
    public abstract boolean removeExample(Writable writable);
}
