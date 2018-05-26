package org.deeplearning4j.integration;

import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.SvhnDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.integration.testcases.*;
import org.junit.*;
import org.junit.rules.TemporaryFolder;

import java.io.File;

public class IntegrationTests extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @BeforeClass
    public static void beforeClass() throws Exception {
        //Initialize some of the iterators before the class... otherwise if these datasets are not available, the
        // tests could time out due to the downloading - not the actual test itself
        new MnistDataSetIterator(1, true, 12345);
        new MnistDataSetIterator(1, false, 12345);
        SvhnDataFetcher fetcher = new SvhnDataFetcher();
    }

    @AfterClass
    public static void afterClass(){
        IntegrationTestRunner.printCoverageInformation();
    }

    // ***** MLPTestCases *****
    @Test(timeout = 20000L)
    public void testMLPMnist() throws Exception {
        IntegrationTestRunner.runTest(MLPTestCases.getMLPMnist(), testDir);
    }

    @Test(timeout = 30000L)
    public void testMlpMoon() throws Exception {
        IntegrationTestRunner.runTest(MLPTestCases.getMLPMoon(), testDir);
    }

    // ***** RNNTestCases *****
    @Test(timeout = 30000L)
    public void testRnnSeqClassification1() throws Exception {
        IntegrationTestRunner.runTest(RNNTestCases.getRnnCsvSequenceClassificationTestCase1(), testDir);
    }

    @Test(timeout = 30000L)
    public void testRnnSeqClassification2() throws Exception {
        IntegrationTestRunner.runTest(RNNTestCases.getRnnCsvSequenceClassificationTestCase2(), testDir);
    }

    @Test(timeout = 120000L)
    public void testRnnCharacter() throws Exception {
        IntegrationTestRunner.runTest(RNNTestCases.getRnnCharacterTestCase(), testDir);
    }


    // ***** CNN1DTestCases *****
    @Test(timeout = 180000L)
    public void testCnn1dSynthetic() throws Exception {
        IntegrationTestRunner.runTest(CNN1DTestCases.getCnn1dTestCaseSynthetic(), testDir);
    }


    // ***** CNN2DTestCases *****
    @Test(timeout = 120000L)
    public void testLenetMnist() throws Exception {
        IntegrationTestRunner.runTest(CNN2DTestCases.getLenetMnist(), testDir);
    }

    @Test(timeout = 360000L)
    public void testVgg16Transfer() throws Exception {
        IntegrationTestRunner.runTest(CNN2DTestCases.getVGG16TransferTinyImagenet(), testDir);
    }

    @Ignore //TODO: NOT YET IMPLEMENTED
    @Test(timeout = 180000L)
    public void testYoloHouseNumbers() throws Exception {
        IntegrationTestRunner.runTest(CNN2DTestCases.getYoloHouseNumbers(), testDir);
    }

    @Ignore //TODO: NOT YET IMPLEMENTED
    @Test(timeout = 120000L)
    public void testCnn2DSynthetic() throws Exception {
        IntegrationTestRunner.runTest(CNN2DTestCases.getCnn2DSynthetic(), testDir);
    }

    @Test(timeout = 120000L)
    public void testCnn2DLenetTransferDropoutRepeatability() throws Exception {
        IntegrationTestRunner.runTest(CNN2DTestCases.testLenetTransferDropoutRepeatability(), testDir);
    }


    // ***** CNN3DTestCases *****

    @Ignore //TODO: NOT YET IMPLEMENTED
    @Test(timeout = 180000L)
    public void testCnn3dSynthetic() throws Exception {
        IntegrationTestRunner.runTest(CNN3DTestCases.getCnn3dTestCaseSynthetic(), testDir);
    }


    // ***** UnsupervisedTestCases *****
    @Test(timeout = 120000L)
    public void testVAEMnistAnomaly() throws Exception {
        IntegrationTestRunner.runTest(UnsupervisedTestCases.getVAEMnistAnomaly(), testDir);
    }

    // ***** TransferLearningTestCases *****
    @Ignore //TODO: NOT YET IMPLEMENTED
    @Test(timeout = 180000L)
    public void testTransferResnet() throws Exception {
        IntegrationTestRunner.runTest(TransferLearningTestCases.testPartFrozenResNet50(), testDir);
    }

    @Ignore //TODO: NOT YET IMPLEMENTED
    @Test(timeout = 180000L)
    public void testTransferNASNET() throws Exception {
        IntegrationTestRunner.runTest(TransferLearningTestCases.testPartFrozenNASNET(), testDir);
    }


    // ***** KerasImportTestCases *****


}
