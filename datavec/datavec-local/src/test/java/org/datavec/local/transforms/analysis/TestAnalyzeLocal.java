package org.datavec.local.transforms.analysis;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.columns.NumericalColumnAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.AnalyzeLocal;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TestAnalyzeLocal {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testAnalysisBasic() throws Exception {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

        Schema s = new Schema.Builder()
                .addColumnsDouble("0", "1", "2", "3")
                .addColumnInteger("label")
                .build();

        DataAnalysis da = AnalyzeLocal.analyze(s, rr);

        System.out.println(da);

        //Compare:
        List<List<Writable>> list = new ArrayList<>();
        rr.reset();
        while(rr.hasNext()){
            list.add(rr.next());
        }

        INDArray arr = RecordConverter.toMatrix(list);
        INDArray mean = arr.mean(0);
        INDArray std = arr.std(0);

        for( int i=0; i<5; i++ ){
            double m = ((NumericalColumnAnalysis)da.getColumnAnalysis().get(i)).getMean();
            double stddev = ((NumericalColumnAnalysis)da.getColumnAnalysis().get(i)).getSampleStdev();
            assertEquals(mean.getDouble(i), m, 1e-3);
            assertEquals(std.getDouble(i), stddev, 1e-3);
        }

    }

}
