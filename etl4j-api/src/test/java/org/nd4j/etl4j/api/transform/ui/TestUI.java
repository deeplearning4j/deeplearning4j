package org.nd4j.etl4j.api.transform.ui;

import org.nd4j.etl4j.api.transform.analysis.DataAnalysis;
import org.nd4j.etl4j.api.transform.analysis.columns.ColumnAnalysis;
import org.nd4j.etl4j.api.transform.analysis.columns.IntegerAnalysis;
import org.nd4j.etl4j.api.transform.analysis.columns.StringAnalysis;
import org.nd4j.etl4j.api.transform.analysis.columns.TimeAnalysis;
import org.nd4j.etl4j.api.transform.schema.Schema;
import org.apache.commons.io.FilenameUtils;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 25/03/2016.
 */
public class TestUI {

    @Test
    public void testUI() throws Exception {
        Schema schema = new Schema.Builder()
                .addColumnString("StringColumn")
                .addColumnInteger("IntColumn")
                .addColumnInteger("IntColumn2")
                .addColumnInteger("IntColumn3")
                .addColumnTime("TimeColumn", DateTimeZone.UTC)
                .build();

        List<ColumnAnalysis> list = new ArrayList<>();
        list.add( new StringAnalysis.Builder()
                .countTotal(10)
                .countUnique(5)
                .maxLength(7)
                .countTotal(999999999L)
                .countUnique(999999999999L)
                .minLength(99999999).maxLength(99999999).meanLength(9999999999.0).sampleStdevLength(99999999.0).sampleVarianceLength(0.99999999999)
                .histogramBuckets(new double[]{0,1,2,3,4,5})
                .histogramBucketCounts(new long[]{50,30,10,12,3})
                .build());

        list.add( new IntegerAnalysis.Builder()
                .countTotal(10).countMaxValue(1).countMinValue(4).min(0).max(30)
                .countTotal(999999999).countMaxValue(99999999).countMinValue(999999999).min(-999999999).max(9999999)
                .min(99999999).max(99999999).mean(9999999999.0).sampleStdev(99999999.0).sampleVariance(0.99999999999)
                .histogramBuckets(new double[]{-3,-2,-1,0,1,2,3})
                .histogramBucketCounts(new long[]{100_000_000,20_000_000,30_000_000,40_000_000,50_000_000,60_000_000})
                .build());

        list.add( new IntegerAnalysis.Builder()
                .countTotal(10).countMaxValue(1).countMinValue(4).min(0).max(30)
                .histogramBuckets(new double[]{-3,-2,-1,0,1,2,3})
                .histogramBucketCounts(new long[]{15,20,35,40,55,60})
                .build());

        list.add( new IntegerAnalysis.Builder()
                .countTotal(10).countMaxValue(1).countMinValue(4).min(0).max(30)
                .histogramBuckets(new double[]{-3,-2,-1,0,1,2,3})
                .histogramBucketCounts(new long[]{10,2,3,4,5,6})
                .build());

        list.add(new TimeAnalysis.Builder()
                .min(1451606400000L)
                .max(1451606400000L + 60000L)
                .build());


        DataAnalysis da = new DataAnalysis(schema,list);

        String tempDir = System.getProperty("java.io.tmpdir");
        String outPath = FilenameUtils.concat(tempDir,"echidnaUITest.html");
        System.out.println(outPath);
        HtmlAnalysis.createHtmlAnalysisFile(da, new File(outPath));
    }
}
