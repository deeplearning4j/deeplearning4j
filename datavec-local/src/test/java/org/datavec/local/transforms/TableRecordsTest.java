package org.datavec.local.transforms;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import tech.tablesaw.api.Table;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 11/11/16.
 */
public class TableRecordsTest {
    @Test
    public void testToTable() {
        INDArray linspace = Nd4j.linspace(1, 8, 8).reshape(2, 4);
        List<List<Writable>> matrix = RecordConverter.toRecords(linspace);
        Schema.Builder schemaBuilder = new Schema.Builder();
        for (int i = 0; i < linspace.columns(); i++)
            schemaBuilder.addColumnDouble(String.valueOf(i));
        Table table = TableRecords.fromRecordsAndSchema(matrix, schemaBuilder.build());
        assertEquals(4, table.columns().size());
        assertEquals(2, table.rows().length);
        for (int i = 0; i < table.rows().length; i++) {
            for (int j = 0; j < table.columns().size(); j++) {
                assertEquals(linspace.getDouble(i, j), Double.valueOf(table.get(j, i)), 1e-1);
            }
        }

        assertEquals(linspace, TableRecords.arrayFromTable(table));
        System.out.println(table.printHtml());
    }

    @Test
    public void testRecordsFromTable() {
        int rows = 4;
        int columns = 2;
        List<List<Writable>> assertion = new ArrayList<>();

        for (int i = 0; i < rows; i++) {
            assertion.add(new ArrayList<>());
            for (int j = 0; j < columns; j++) {
                assertion.get(i).add(new DoubleWritable(i));
            }
        }

        Schema.Builder schemaBuilder = new Schema.Builder();
        for (int i = 0; i < columns; i++)
            schemaBuilder.addColumnDouble(String.valueOf(i));
        Table table = TableRecords.fromRecordsAndSchema(assertion, schemaBuilder.build());
        List<List<Writable>> test = TableRecords.fromTable(table);
        assertEquals(assertion, test);

    }


    @Test
    public void testTransform() {

    }



}
