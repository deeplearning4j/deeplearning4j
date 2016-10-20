package org.datavec.api.transform.filter;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Remove invalid records of a certain size.
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
public class InvalidNumColumns implements Filter {
    private Schema inputSchema;
    /**
     * @param writables Example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeExample(List<Writable> writables) {
        return writables.size() != inputSchema.numColumns();
    }

    /**
     * @param sequence sequence example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeSequence(List<List<Writable>> sequence) {
        for(List<Writable> record : sequence)
            if(record.size() != inputSchema.numColumns())
                return true;
        return false;
    }

    @Override
    public void setInputSchema(Schema schema) {
        this.inputSchema = schema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }
}
