package io.skymind.echidna.api.transform.column;

import org.canova.api.writable.Writable;
import io.skymind.echidna.api.Transform;
import io.skymind.echidna.api.metadata.ColumnMetaData;
import io.skymind.echidna.api.schema.Schema;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Rearrange the order of the columns.
 * Note: A partial list of columns can be used here. Any columns that are not explicitly mentioned
 * will be placed after those that are in the output, without changing their relative order.
 *
 * @author Alex Black
 */
public class ReorderColumnsTransform implements Transform {

    private final List<String> newOrder;
    private Schema inputSchema;
    private int[] outputOrder;  //Mapping from in to out. so output[i] = input.get(outputOrder[i])

    /**
     *
     * @param newOrder    A partial or complete order of the columns in the output
     */
    public ReorderColumnsTransform(String... newOrder){
        this(Arrays.asList(newOrder));
    }

    /**
     *
     * @param newOrder    A partial or complete order of the columns in the output
     */
    public ReorderColumnsTransform(List<String> newOrder){
        this.newOrder = newOrder;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        for(String s : newOrder){
            if(!inputSchema.hasColumn(s)){
                throw new IllegalStateException("Input schema does not contain column with name \"" + s + "\"");
            }
        }
        if(inputSchema.numColumns() < newOrder.size()) throw new IllegalArgumentException("Schema has " + inputSchema.numColumns() +
            " column but newOrder has " + newOrder.size() + " columns");

        List<String> origNames = inputSchema.getColumnNames();
        List<ColumnMetaData> origMeta = inputSchema.getColumnMetaData();

        List<String> outNames = new ArrayList<>();
        List<ColumnMetaData> outMeta = new ArrayList<>();

        boolean[] taken = new boolean[origNames.size()];
        for(String s : newOrder){
            int idx = inputSchema.getIndexOfColumn(s);
            outNames.add(origNames.get(idx));
            outMeta.add(origMeta.get(idx));
            taken[idx] = true;
        }

        for( int i=0; i<taken.length; i++ ){
            if(taken[i]) continue;
            outNames.add(origNames.get(i));
            outMeta.add(origMeta.get(i));
        }

        return inputSchema.newSchema(outNames, outMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        for(String s : newOrder){
            if(!inputSchema.hasColumn(s)){
                throw new IllegalStateException("Input schema does not contain column with name \"" + s + "\"");
            }
        }
        if(inputSchema.numColumns() < newOrder.size()) throw new IllegalArgumentException("Schema has " + inputSchema.numColumns() +
                " columns but newOrder has " + newOrder.size() + " columns");

        List<String> origNames = inputSchema.getColumnNames();
        outputOrder = new int[origNames.size()];

        boolean[] taken = new boolean[origNames.size()];
        int j=0;
        for(String s : newOrder){
            int idx = inputSchema.getIndexOfColumn(s);
            taken[idx] = true;
            outputOrder[j++] = idx;
        }

        for( int i=0; i<taken.length; i++ ){
            if(taken[i]) continue;
            outputOrder[j++] = i;
        }
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        List<Writable> out = new ArrayList<>();
        for(int i : outputOrder){
            out.add(writables.get(i));
        }
        return out;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>();
        for(List<Writable> step : sequence){
            out.add(map(step));
        }
        return out;
    }
}
