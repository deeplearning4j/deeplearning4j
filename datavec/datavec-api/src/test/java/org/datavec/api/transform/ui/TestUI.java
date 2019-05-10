/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.transform.ui;

import com.tdunning.math.stats.TDigest;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.SequenceDataAnalysis;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.analysis.columns.IntegerAnalysis;
import org.datavec.api.transform.analysis.columns.StringAnalysis;
import org.datavec.api.transform.analysis.columns.TimeAnalysis;
import org.datavec.api.transform.analysis.sequence.SequenceLengthAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 25/03/2016.
 */
public class TestUI {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testUI() throws Exception {
        Schema schema = new Schema.Builder().addColumnString("StringColumn").addColumnInteger("IntColumn")
                        .addColumnInteger("IntColumn2").addColumnInteger("IntColumn3")
                        .addColumnTime("TimeColumn", DateTimeZone.UTC).build();

        List<ColumnAnalysis> list = new ArrayList<>();
        list.add(new StringAnalysis.Builder().countTotal(10).maxLength(7).countTotal(999999999L).minLength(99999999)
                        .maxLength(99999999).meanLength(9999999999.0).sampleStdevLength(99999999.0)
                        .sampleVarianceLength(0.99999999999).histogramBuckets(new double[] {0, 1, 2, 3, 4, 5})
                        .histogramBucketCounts(new long[] {50, 30, 10, 12, 3}).build());

        list.add(new IntegerAnalysis.Builder().countTotal(10).countMaxValue(1).countMinValue(4).min(0).max(30)
                        .countTotal(999999999).countMaxValue(99999999).countMinValue(999999999).min(-999999999)
                        .max(9999999).min(99999999).max(99999999).mean(9999999999.0).sampleStdev(99999999.0)
                        .sampleVariance(0.99999999999)
                        .histogramBuckets(new double[] {-3, -2, -1, 0, 1, 2, 3}).histogramBucketCounts(new long[] {
                                        100_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000})
                        .build());

        list.add(new IntegerAnalysis.Builder().countTotal(10).countMaxValue(1).countMinValue(4).min(0).max(30)
                        .histogramBuckets(new double[] {-3, -2, -1, 0, 1, 2, 3})
                        .histogramBucketCounts(new long[] {15, 20, 35, 40, 55, 60}).build());

        TDigest t = TDigest.createDigest(100);
        for( int i=0; i<100; i++ ){
            t.add(i);
        }
        list.add(new IntegerAnalysis.Builder().countTotal(10).countMaxValue(1).countMinValue(4).min(0).max(30)
                        .histogramBuckets(new double[] {-3, -2, -1, 0, 1, 2, 3})
                        .histogramBucketCounts(new long[] {10, 2, 3, 4, 5, 6})
                        .digest(t)
                .build());

        list.add(new TimeAnalysis.Builder().min(1451606400000L).max(1451606400000L + 60000L).build());


        DataAnalysis da = new DataAnalysis(schema, list);

        File fDir = testDir.newFolder();
        String tempDir = fDir.getAbsolutePath();
        String outPath = FilenameUtils.concat(tempDir, "datavec_transform_UITest.html");
        System.out.println(outPath);
        File f = new File(outPath);
        f.deleteOnExit();
        HtmlAnalysis.createHtmlAnalysisFile(da, f);


        //Test JSON:
        String json = da.toJson();
        DataAnalysis fromJson = DataAnalysis.fromJson(json);
        assertEquals( da, fromJson );



        //Test sequence analysis:
        SequenceLengthAnalysis sla = SequenceLengthAnalysis.builder()
                .totalNumSequences(100)
                .minSeqLength(1)
                .maxSeqLength(50)
                .countZeroLength(0)
                .countOneLength(10)
                .meanLength(20.0)
                .histogramBuckets(new double[]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0})
                .histogramBucketCounts(new long[]{1,2,3,4,5})
                .build();
        SequenceDataAnalysis sda = new SequenceDataAnalysis(da.getSchema(), da.getColumnAnalysis(), sla);


        //HTML:
        outPath = FilenameUtils.concat(tempDir, "datavec_transform_UITest_seq.html");
        System.out.println(outPath);
        f = new File(outPath);
        f.deleteOnExit();
        HtmlAnalysis.createHtmlAnalysisFile(sda, f);


        //JSON
        json = sda.toJson();
        SequenceDataAnalysis sFromJson = SequenceDataAnalysis.fromJson(json);

        String toStr1 = sda.toString();
        String toStr2 = sFromJson.toString();
        assertEquals(toStr1, toStr2);

        assertEquals(sda, sFromJson);
    }


    @Test
    @Ignore
    public void testSequencePlot() throws Exception {

        Schema schema = new SequenceSchema.Builder().addColumnDouble("sinx")
                        .addColumnCategorical("cat", "s0", "s1", "s2").addColumnString("stringcol").build();

        int nSteps = 100;
        List<List<Writable>> sequence = new ArrayList<>(nSteps);
        for (int i = 0; i < nSteps; i++) {
            String c = "s" + i % 3;
            sequence.add(Arrays.<Writable>asList(new DoubleWritable(Math.sin(i / 10.0)), new Text(c),
                            new Text(String.valueOf(i))));
        }

        String tempDir = System.getProperty("java.io.tmpdir");
        String outPath = FilenameUtils.concat(tempDir, "datavec_seqplot_test.html");
        //        System.out.println(outPath);
        File f = new File(outPath);
        f.deleteOnExit();
        HtmlSequencePlotting.createHtmlSequencePlotFile("Title!", schema, sequence, f);


    }
}
