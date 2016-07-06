package org.nd4j.etl4j.api.transform.transform.column;

import io.skymind.echidna.api.Transform;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.schema.Schema;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Duplicate one or more columns.
 * The duplicated columns are placed immediately after the original columns
 *
 * @author Alex Black
 */
public class DuplicateColumnsTransform implements Transform {

    private final List<String> columnsToDuplicate;
    private final List<String> newColumnNames;
    private final Set<String> columnsToDuplicateSet;
    private final Set<Integer> columnIndexesToDuplicateSet;
    private Schema inputSchema;

    /**
     * @param columnsToDuplicate List of columns to duplicate
     * @param newColumnNames     List of names for the new (duplicate) columns
     */
    public DuplicateColumnsTransform(List<String> columnsToDuplicate, List<String> newColumnNames) {
        if (columnsToDuplicate == null || newColumnNames == null)
            throw new IllegalArgumentException("Columns/names cannot be null");
        if (columnsToDuplicate.size() != newColumnNames.size())
            throw new IllegalArgumentException("Invalid input: columns to duplicate and the new names must have equal lengths");
        this.columnsToDuplicate = columnsToDuplicate;
        this.newColumnNames = newColumnNames;
        this.columnsToDuplicateSet = new HashSet<>(columnsToDuplicate);
        this.columnIndexesToDuplicateSet = new HashSet<>();
    }

    @Override
    public Schema transform(Schema inputSchema) {
        List<ColumnMetaData> oldMeta = inputSchema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(oldMeta.size() + newColumnNames.size());

        List<String> oldNames = inputSchema.getColumnNames();
        List<String> newNames = new ArrayList<>(oldNames.size() + newColumnNames.size());

        int dupCount = 0;
        for (int i = 0; i < oldMeta.size(); i++) {
            String current = oldNames.get(i);
            newNames.add(current);
            newMeta.add(oldMeta.get(i));

            if (columnsToDuplicateSet.contains(current)) {
                //Duplicate the current column, and place it after...
                String dupName = newColumnNames.get(dupCount);
                newNames.add(dupName);
                newMeta.add(oldMeta.get(i).clone());
                dupCount++;
            }
        }

        return inputSchema.newSchema(newNames, newMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        columnIndexesToDuplicateSet.clear();

        List<String> schemaColumnNames = inputSchema.getColumnNames();
        for (String s : columnsToDuplicate) {
            int idx = schemaColumnNames.indexOf(s);
            if (idx == -1)
                throw new IllegalStateException("Invalid state: column to duplicate \"" + s + "\" does not appear "
                        + "in input schema");
            columnIndexesToDuplicateSet.add(idx);
        }

        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if(writables.size() != inputSchema.numColumns() ){
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size() + ") does not " +
                    "match expected number of elements (schema: " + inputSchema.numColumns() + "). Transform = " + toString());
        }
        List<Writable> out = new ArrayList<>(writables.size() + columnsToDuplicate.size());
        int i = 0;
        for (Writable w : writables) {
            out.add(w);
            if (columnIndexesToDuplicateSet.contains(i++)) out.add(w);   //TODO safter to copy here...
        }
        return out;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>(sequence.size());
        for (List<Writable> l : sequence) {
            out.add(map(l));
        }
        return out;
    }

    @Override
    public String toString(){
        return "DuplicateColumnsTransform(toDuplicate=" + columnsToDuplicate + ",newNames=" + newColumnNames + ")";
    }
}
