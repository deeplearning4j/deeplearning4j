package integration;

import lombok.Data;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.List;

@Data
public abstract class TestCase {

    public enum TestType {
        PRETRAINED, RANDOM_INIT
    }

    protected String testName;
    protected TestType testType;
    protected boolean testPredictions = true;
    protected boolean testGradients = true;
    protected boolean testUnsupervisedTraining = false;
    protected boolean testTrainingCurves = true;
    protected boolean testParamsPostTraining = true;
    protected boolean testEvaluation = true;
    protected boolean testParallelInference = true;
    protected boolean testOverfitting = true;

    protected int[] unsupervisedTrainLayersMLN = null;
    protected String[] unsupervisedTrainLayersCG = null;

    //Relative errors for this test case:
    protected double maxRelativeErrorGradients = 1e-6;
    protected double minAbsErrorGradients = 1e-5;
    protected double maxRelativeErrorPretrainParams = 1e-5;
    protected double minAbsErrorPretrainParams = 1e-5;
    protected double maxRelativeErrorScores = 1e-6;
    protected double minAbsErrorScores = 1e-5;
    protected double maxRelativeErrorParamsPostTraining = 1e-4;
    protected double minAbsErrorParamsPostTraining = 1e-4;
    protected double maxRelativeErrorOverfit = 1e-2;
    protected double minAbsErrorOverfit = 1e-2;

    /**
     * Initialize the test case... many tests don't need this; others may use it to download or create data
     * @param testWorkingDir
     * @throws Exception
     */
    public void initialize(File testWorkingDir) throws Exception {
        //No op
    }

    /**
     * Required if NOT a pretrained model (testType == TestType.RANDOM_INIT)
     */
    public Object getConfiguration() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required for pretrained models (testType == TestType.PRETRAINED)
     */
    public Model getPretrainedModel() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }


    public List<Pair<INDArray[],INDArray[]>> getPredictionsTestData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    public MultiDataSet getGradientsTestData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required when testUnsupervisedTraining == true
     */
    public MultiDataSetIterator getUnsupervisedTrainData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * @return Training data - DataSetIterator or MultiDataSetIterator
     */
    public MultiDataSetIterator getTrainingData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testEvaluation == true
     */
    public IEvaluation[] getNewEvaluations() {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testEvaluation == true
     */
    public MultiDataSetIterator getEvaluationTestData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testOverfitting == true
     */
    public MultiDataSet getOverfittingData() throws Exception {
        throw new RuntimeException("Implementations must override this method if used");
    }

    /**
     * Required if testOverfitting == true
     */
    public int getOverfitNumIterations() {
        throw new RuntimeException("Implementations must override this method if used");
    }


}
