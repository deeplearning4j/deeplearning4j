package integration.testcases;

import integration.TestCase;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationCalibration;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

public class CNN2DTestCases {

    /**
     * Essentially: LeNet MNIST example
     */
    public static TestCase getLenetMnist(){

        throw new UnsupportedOperationException("Not yet implemented");
    }


    /**
     * VGG16 + transfer learning + tiny imagenet
     */
    public static TestCase getVGG16TransferTinyImagenet(){
        return new TestCase() {

            {
                testName = "VGG16TransferTinyImagenet_224";
                testType = TestType.PRETRAINED;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testModelSerialization = true;
                testOverfitting = true;
            }

            @Override
            public Model getPretrainedModel() throws Exception {
                VGG16 vgg16 = VGG16.builder()
                        .seed(12345)
                        .build();

                ComputationGraph pretrained = (ComputationGraph) vgg16.initPretrained(PretrainedType.IMAGENET);

                //Transfer learning
                ComputationGraph newGraph = new TransferLearning.GraphBuilder(pretrained)

                        .removeVertexKeepConnections("predictions")
                        .addLayer("predictions", new OutputLayer.Builder()
                                .nIn(4096)
                                .nOut(200)  //Tiny imagenet
                                .build(), "fc2")
                        .build();

                return newGraph;
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                List<Pair<INDArray[],INDArray[]>> out = new ArrayList<>();

                DataSetIterator iter = new TinyImageNetDataSetIterator(1, new int[]{224,224}, DataSetType.TRAIN, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                DataSet ds = iter.next();
                out.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));

                iter = new TinyImageNetDataSetIterator(3, new int[]{224,224}, DataSetType.TRAIN, null, 54321);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                ds = iter.next();
                out.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));

                return out;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                DataSet ds =  new TinyImageNetDataSetIterator(8, new int[]{224,224}, DataSetType.TRAIN, null, 12345).next();
                return new org.nd4j.linalg.dataset.MultiDataSet(ds.getFeatures(), ds.getLabels());
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new TinyImageNetDataSetIterator(8, new int[]{224,224}, DataSetType.TRAIN, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());

                iter = new EarlyTerminationDataSetIterator(iter, 10);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations(){
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass(),
                        new EvaluationCalibration()
                };
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                DataSetIterator iter = new TinyImageNetDataSetIterator(8, new int[]{224,224}, DataSetType.TEST, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                iter = new EarlyTerminationDataSetIterator(iter, 10);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public MultiDataSet getOverfittingData() throws Exception {
                DataSetIterator iter = new TinyImageNetDataSetIterator(1, new int[]{224,224}, DataSetType.TRAIN, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                return ComputationGraphUtil.toMultiDataSet(iter.next());
            }

            @Override
            public int getOverfitNumIterations(){
                return 200;
            }
        };
    }


    /**
     * Basically a cut-down version of the YOLO house numbers example
     */
    public static TestCase getYoloHouseNumbers(){

        throw new UnsupportedOperationException("Not yet implemented");
    }


    /**
     * A synthetic 2D CNN that uses all layers:
     * Convolution, Subsampling, Upsampling, Cropping, Depthwise conv, separable conv, deconv, space to batch,
     * space to depth, zero padding, batch norm, LRN
     */
    public static TestCase getCnn2DSynthetic(){

        throw new UnsupportedOperationException("Not yet implemented");
    }
}
