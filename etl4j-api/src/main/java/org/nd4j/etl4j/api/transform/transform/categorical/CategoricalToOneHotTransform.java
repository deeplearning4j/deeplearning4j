package org.nd4j.etl4j.api.transform.transform.categorical;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.canova.api.io.data.IntWritable;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.metadata.IntegerMetaData;
import org.nd4j.etl4j.api.transform.metadata.CategoricalMetaData;
import org.nd4j.etl4j.api.transform.schema.Schema;
import org.nd4j.etl4j.api.transform.transform.BaseTransform;

import java.util.*;

/**
 * Created by Alex on 4/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class CategoricalToOneHotTransform extends BaseTransform {

    private String columnName;
    private int columnIdx = -1;
    private List<String> stateNames;
    private Map<String,Integer> statesMap;

    public CategoricalToOneHotTransform(String columnName) {
        this.columnName = columnName;
    }

    @Override
    public void setInputSchema(Schema inputSchema){
        super.setInputSchema(inputSchema);

        columnIdx = inputSchema.getIndexOfColumn(columnName);
        ColumnMetaData meta = inputSchema.getMetaData(columnName);
        if(!(meta instanceof CategoricalMetaData)) throw new IllegalStateException("Cannot convert column \"" +
                columnName + "\" from categorical to one-hot: column is not categorical (is: " + meta.getColumnType() + ")");
        this.stateNames = ((CategoricalMetaData)meta).getStateNames();

        this.statesMap = new HashMap<>(stateNames.size());
        for( int i=0; i<stateNames.size(); i++ ){
            this.statesMap.put(stateNames.get(i), i);
        }
    }

    @Override
    public Schema transform(Schema schema) {

        List<String> origNames = schema.getColumnNames();
        List<ColumnMetaData> origMeta = schema.getColumnMetaData();

        int i=0;
        Iterator<String> namesIter = origNames.iterator();
        Iterator<ColumnMetaData> typesIter = origMeta.iterator();

        List<String> outNames = new ArrayList<>(origNames.size()+stateNames.size()-1);
        List<ColumnMetaData> newMeta = new ArrayList<>(outNames.size());

        while(namesIter.hasNext()){
            String s = namesIter.next();
            ColumnMetaData t = typesIter.next();

            if(i++ == columnIdx){
                //Convert this to one-hot:
                for (String stateName : stateNames) {
                    String newName = s + "[" + stateName + "]";
                    outNames.add(newName);
                    newMeta.add(new IntegerMetaData(0,1));
//                    newMeta.add(new CategoricalMetaData("0","1"));
                }
            } else {
                outNames.add(s);
                newMeta.add(t);
            }
        }

        return schema.newSchema(outNames,newMeta);
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if(writables.size() != inputSchema.numColumns() ){
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size() + ") does not " +
                    "match expected number of elements (schema: " + inputSchema.numColumns() + "). Transform = " + toString());
        }
        int idx = getColumnIdx();

        int n = stateNames.size();
        List<Writable> out = new ArrayList<>(writables.size() + n);

        int i=0;
        for( Writable w : writables){

            if(i++ == idx){
                //Do conversion
                String str = w.toString();
                Integer classIdx = statesMap.get(str);
                if(classIdx == null) throw new RuntimeException("Unknown state (index not found): " + str);
                for( int j=0; j<n; j++ ){
                    if(j == classIdx ) out.add(new IntWritable(1));
                    else out.add(new IntWritable(0));
                }
            } else {
                //No change to this column
                out.add(w);
            }
        }
        return out;
    }
}
