package org.datavec.poi.excel;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ExcelRecordReaderTest {

    @Test
    public void testSimple() throws Exception {
        RecordReader excel = new ExcelRecordReader();
        excel.initialize(new FileSplit(new ClassPathResource("datavec-excel/testsheet.xlsx").getFile()));
        assertTrue(excel.hasNext());
        List<Writable> next = excel.next();
        assertEquals(3,next.size());

        RecordReader headerReader = new ExcelRecordReader(1);
        headerReader.initialize(new FileSplit(new ClassPathResource("datavec-excel/testsheetheader.xlsx").getFile()));
        assertTrue(excel.hasNext());
        List<Writable> next2 = excel.next();
        assertEquals(3,next2.size());


    }

}
