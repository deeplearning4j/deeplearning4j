package org.deeplearning4j.spark.util;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.util.data.SparkDataValidation;
import org.deeplearning4j.spark.util.data.ValidationResult;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class TestValidation extends BaseSparkTest {

    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void testDataSetValidation() throws Exception {

        File f = folder.newFolder();

        for( int i=0; i<3; i++ ) {
            DataSet ds = new DataSet(Nd4j.create(10), Nd4j.create(10));
            ds.save(new File(f, i + ".bin"));
        }

        ValidationResult r = SparkDataValidation.validateDataSets(sc, f.toURI().toString());
        ValidationResult exp = ValidationResult.builder()
                .countTotal(3)
                .countTotalValid(3)
                .build();
        assertEquals(exp, r);

        //Add a DataSet that is corrupt (can't be loaded)
        File f3 = new File(f, "3.bin");
        FileUtils.writeStringToFile(f3, "This isn't a DataSet!");
        r = SparkDataValidation.validateDataSets(sc, f.toURI().toString());
        exp = ValidationResult.builder()
                .countTotal(4)
                .countTotalValid(3)
                .countTotalInvalid(1)
                .countLoadingFailure(1)
                .build();
        assertEquals(exp, r);
        f3.delete();


        //Add a DataSet with missing features:
        new DataSet(null, Nd4j.create(10)).save(f3);

        r = SparkDataValidation.validateDataSets(sc, f.toURI().toString());
        exp = ValidationResult.builder()
                .countTotal(4)
                .countTotalValid(3)
                .countTotalInvalid(1)
                .countMissingFeatures(1)
                .build();
        assertEquals(exp, r);

        r = SparkDataValidation.deleteInvalidDataSets(sc, f.toURI().toString());
        exp.setCountInvalidDeleted(1);
        assertEquals(exp, r);
        assertFalse(f3.exists());
        for( int i=0; i<3; i++ ){
            assertTrue(new File(f,i + ".bin").exists());
        }

        //Add DataSet with incorrect labels shape:
        new DataSet(Nd4j.create(10), Nd4j.create(20)).save(f3);
        r = SparkDataValidation.validateDataSets(sc, f.toURI().toString(), new int[]{-1,10}, new int[]{-1,10});
        exp = ValidationResult.builder()
                .countTotal(4)
                .countTotalValid(3)
                .countTotalInvalid(1)
                .countInvalidLabels(1)
                .build();

        assertEquals(exp, r);
    }

    @Test
    public void testMultiDataSetValidation() throws Exception {

        File f = folder.newFolder();

        for( int i=0; i<3; i++ ) {
            MultiDataSet ds = new MultiDataSet(Nd4j.create(10), Nd4j.create(10));
            ds.save(new File(f, i + ".bin"));
        }

        ValidationResult r = SparkDataValidation.validateMultiDataSets(sc, f.toURI().toString());
        ValidationResult exp = ValidationResult.builder()
                .countTotal(3)
                .countTotalValid(3)
                .build();
        assertEquals(exp, r);

        //Add a MultiDataSet that is corrupt (can't be loaded)
        File f3 = new File(f, "3.bin");
        FileUtils.writeStringToFile(f3, "This isn't a MultiDataSet!");
        r = SparkDataValidation.validateMultiDataSets(sc, f.toURI().toString());
        exp = ValidationResult.builder()
                .countTotal(4)
                .countTotalValid(3)
                .countTotalInvalid(1)
                .countLoadingFailure(1)
                .build();
        assertEquals(exp, r);
        f3.delete();


        //Add a MultiDataSet with missing features:
        new MultiDataSet(null, Nd4j.create(10)).save(f3);

        r = SparkDataValidation.validateMultiDataSets(sc, f.toURI().toString());
        exp = ValidationResult.builder()
                .countTotal(4)
                .countTotalValid(3)
                .countTotalInvalid(1)
                .countMissingFeatures(1)
                .build();
        assertEquals(exp, r);

        r = SparkDataValidation.deleteInvalidMultiDataSets(sc, f.toURI().toString());
        exp.setCountInvalidDeleted(1);
        assertEquals(exp, r);
        assertFalse(f3.exists());
        for( int i=0; i<3; i++ ){
            assertTrue(new File(f,i + ".bin").exists());
        }

        //Add MultiDataSet with incorrect labels shape:
        new MultiDataSet(Nd4j.create(10), Nd4j.create(20)).save(f3);
        r = SparkDataValidation.validateMultiDataSets(sc, f.toURI().toString(), Arrays.asList(new int[]{-1,10}),
                Arrays.asList(new int[]{-1,10}));
        exp = ValidationResult.builder()
                .countTotal(4)
                .countTotalValid(3)
                .countTotalInvalid(1)
                .countInvalidLabels(1)
                .build();
        f3.delete();
        assertEquals(exp, r);

        //Add a MultiDataSet with incorrect number of feature arrays:
        new MultiDataSet(new INDArray[]{Nd4j.create(10), Nd4j.create(10)},
                new INDArray[]{Nd4j.create(10)}).save(f3);
        r = SparkDataValidation.validateMultiDataSets(sc, f.toURI().toString(), Arrays.asList(new int[]{-1,10}),
                Arrays.asList(new int[]{-1,10}));
        exp = ValidationResult.builder()
                .countTotal(4)
                .countTotalValid(3)
                .countTotalInvalid(1)
                .countInvalidFeatures(1)
                .build();
        assertEquals(exp, r);


        r = SparkDataValidation.deleteInvalidMultiDataSets(sc, f.toURI().toString(), Arrays.asList(new int[]{-1,10}),
                Arrays.asList(new int[]{-1,10}));
        exp.setCountInvalidDeleted(1);
        assertEquals(exp, r);
        assertFalse(f3.exists());
    }

}
