package io.skymind.echidna.api.transform.column;

import io.skymind.echidna.api.metadata.ColumnMetaData;
import io.skymind.echidna.api.schema.Schema;
import io.skymind.echidna.api.transform.BaseTransform;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.canova.api.writable.Writable;

import java.util.*;

@EqualsAndHashCode(callSuper = true)
@Data
public class RemoveColumnsTransform extends BaseTransform {

    private int[] columnsToRemoveIdx;
    private String[] columnsToRemove;
    private Set<Integer> indicesToRemove;

    public RemoveColumnsTransform(String... columnsToRemove){
        this.columnsToRemove = columnsToRemove;
    }

    @Override
    public void setInputSchema(Schema schema){
        super.setInputSchema(schema);

        indicesToRemove = new HashSet<>();

        int i=0;
        columnsToRemoveIdx = new int[columnsToRemove.length];
        for( String s : columnsToRemove){
            int idx = schema.getIndexOfColumn(s);
            if(idx<0) throw new RuntimeException("Column \"" + s + "\" not found");
            columnsToRemoveIdx[i++] = idx;
            indicesToRemove.add(idx);
        }
    }

    @Override
    public Schema transform(Schema schema) {
        int nToRemove = columnsToRemove.length;
        int newNumColumns = schema.numColumns() - nToRemove;
        if(newNumColumns <= 0 ) throw new IllegalStateException("Number of columns after executing operation is " + newNumColumns + " (is <= 0). " +
            "origColumns = " + schema.getColumnNames() + ", toRemove = " + Arrays.toString(columnsToRemove));

        List<String> origNames = schema.getColumnNames();
        List<ColumnMetaData> origMeta = schema.getColumnMetaData();

        Set<String> set = new HashSet<>();
        Collections.addAll(set, columnsToRemove);


        List<String> newNames = new ArrayList<>(newNumColumns);
        List<ColumnMetaData> newMeta = new ArrayList<>(newNumColumns);

        Iterator<String> namesIter = origNames.iterator();
        Iterator<ColumnMetaData> metaIter = origMeta.iterator();

        while(namesIter.hasNext()){
            String n = namesIter.next();
            ColumnMetaData t = metaIter.next();
            if(!set.contains(n)){
                newNames.add(n);
                newMeta.add(t);
            }
        }

        return schema.newSchema(newNames,newMeta);
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if(writables.size() != inputSchema.numColumns() ){
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size() + ") does not " +
                    "match expected number of elements (schema: " + inputSchema.numColumns() + "). Transform = " + toString());
        }

        List<Writable> outList = new ArrayList<>(writables.size()-columnsToRemove.length);

        int i=0;
        for(Writable w : writables){
            if(indicesToRemove.contains(i++)) continue;
            outList.add(w);
        }
        return outList;
    }

    @Override
    public String toString(){
        return "RemoveColumnsTransform(" + Arrays.toString(columnsToRemove) + ")";
    }
}
