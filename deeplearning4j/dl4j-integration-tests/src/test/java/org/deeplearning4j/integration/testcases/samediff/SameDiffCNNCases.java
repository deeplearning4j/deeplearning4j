package org.deeplearning4j.integration.testcases.samediff;

import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.datasets.iterator.impl.SingletonMultiDataSetIterator;
import org.deeplearning4j.integration.ModelType;
import org.deeplearning4j.integration.TestCase;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.EvaluationCalibration;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.util.*;

public class SameDiffCNNCases {


    public static TestCase getLenetMnist() {
        return new TestCase() {
            {
                testName = "LenetMnistSD";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = false;
            }

            @Override
            public ModelType modelType() {
                return ModelType.SAMEDIFF;
            }

            public Object getConfiguration() throws Exception {
                int nChannels = 1; // Number of input channels
                int outputNum = 10; // The number of possible outcomes
                int seed = 123;

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.placeHolder("in", DataType.FLOAT, nChannels, 784);
                SDVariable label = sd.placeHolder("label", DataType.FLOAT, nChannels, outputNum);

                //input [minibatch, channels=1, Height = 28, Width = 28]
                SDVariable in4d = in.reshape(-1, nChannels, 28, 28);

                int kernelHeight = 5;
                int kernelWidth = 5;


                // w0 [kernelHeight = 5, kernelWidth = 5 , inputChannels = 1, outputChannels = 20]
                // b0 [20]
                SDVariable w0 = sd.var("w0", Nd4j.rand(DataType.FLOAT, kernelHeight, kernelWidth, nChannels, 20));
                SDVariable b0 = sd.var("b0", Nd4j.rand(DataType.FLOAT, 20));


                SDVariable layer0 = sd.nn.relu(sd.cnn.conv2d("layer0", in4d, w0, b0, Conv2DConfig.builder()
                        .kH(kernelHeight)
                        .kW(kernelWidth)
                        .sH(1)
                        .sW(1)
                        .dataFormat("NCHW")
                        .build()), 0);

                // outputSize = (inputSize - kernelSize + 2*padding) / stride + 1
                // outputsize_H(W) = ( 28 - 5 + 2*0 ) / 1 + 1 = 24
                // [minibatch,20,24,24]


                SDVariable layer1 = sd.cnn.maxPooling2d("layer1", layer0, Pooling2DConfig.builder()
                        .kH(2).kW(2)
                        .sH(2).sW(2)
                        .isNHWC(false)
                        .build());

                // outputSize = (inputSize - kernelSize + 2*padding) / stride + 1
                // outputsize_H(W) = ( 24 - 2 + 2*0 ) / 2 + 1 = 12
                // [minibatch,12,12,20]


                // w2 [kernelHeight = 5, kernelWidth = 5 , inputChannels = 20, outputChannels = 50]
                // b0 [50]
                SDVariable w2 = sd.var("w2", Nd4j.rand(DataType.FLOAT, kernelHeight, kernelWidth, 20, 50));
                SDVariable b2 = sd.var("b2", Nd4j.rand(DataType.FLOAT, 50));


                SDVariable layer2 = sd.nn.relu(sd.cnn.conv2d("layer2", layer1, w2, b2, Conv2DConfig.builder()
                        .kH(kernelHeight)
                        .kW(kernelWidth)
                        .sH(1)
                        .sW(1)
                        .dataFormat("NCHW")
                        .build()), 0);

                // outputSize = (inputSize - kernelSize + 2*padding) / stride + 1
                // outputsize_H(W) = ( 12 - 5 + 2*0 ) / 1 + 1 = 8
                // [minibatch,8,8,50]


                SDVariable layer3 = sd.cnn.maxPooling2d("layer3", layer2, Pooling2DConfig.builder()
                        .kH(2).kW(2)
                        .sH(2).sW(2)
                        .isNHWC(false)
                        .build());


                // outputSize = (inputSize - kernelSize + 2*padding) / stride + 1
                // outputsize_H(W) = ( 8 - 2 + 2*0 ) / 2 + 1 = 4
                // [minibatch,4,4,50]

                int channels_height_width = 4 * 4 * 50;
                SDVariable layer3_reshaped = layer3.reshape(-1, channels_height_width);

                SDVariable w4 = sd.var("w4", Nd4j.rand(DataType.FLOAT, channels_height_width, 500));
                SDVariable b4 = sd.var("b4", Nd4j.rand(DataType.FLOAT, 500));


                SDVariable layer4 = sd.nn.relu("layer4", layer3_reshaped.mmul(w4).add(b4), 0);

                SDVariable w5 = sd.var("w5", Nd4j.rand(DataType.FLOAT, 500, outputNum));
                SDVariable b5 = sd.var("b5", Nd4j.rand(DataType.FLOAT, outputNum));

                SDVariable out = sd.nn.softmax("out", layer4.mmul(w5).add(b5));
                SDVariable loss = sd.loss.logLoss("loss", label, out);

                //Also set the training configuration:
                sd.setTrainingConfig(TrainingConfig.builder()
                        .updater(new Nesterovs(0.01, 0.9))
                        .weightDecay(1e-3, true)
                        .dataSetFeatureMapping("in")            //features[0] -> "in" placeholder
                        .dataSetLabelMapping("label")           //labels[0]   -> "label" placeholder
                        .build());


                return sd;


            }

            @Override
            public Map<String, INDArray> getGradientsTestDataSameDiff() throws Exception {
                DataSet ds = new MnistDataSetIterator(8, true, 12345).next();
                Map<String, INDArray> map = new HashMap<>();
                map.put("in", ds.getFeatures());
                map.put("label", ds.getLabels());
                return map;
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(16, true, 12345);

                iter = new EarlyTerminationDataSetIterator(iter, 60);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                return new MultiDataSetIteratorAdapter(new EarlyTerminationDataSetIterator(new MnistDataSetIterator(32, false, 12345), 10));
            }

            @Override
            public List<Map<String, INDArray>> getPredictionsTestDataSameDiff() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(8, true, 12345);

                List<Map<String, INDArray>> list = new ArrayList<>();

                org.nd4j.linalg.dataset.DataSet ds = iter.next();
                ds = ds.asList().get(0);

                list.add(Collections.singletonMap("in", ds.getFeatures()));
                ds = iter.next();
                list.add(Collections.singletonMap("in", ds.getFeatures()));
                return list;
            }

            @Override
            public List<String> getPredictionsNamesSameDiff() {
                return Collections.singletonList("out");

            }

            @Override
            public IEvaluation[] getNewEvaluations() {
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass(),
                        new EvaluationCalibration()};
            }

            @Override
            public IEvaluation[] doEvaluationSameDiff(SameDiff sd, MultiDataSetIterator iter, IEvaluation[] evaluations) {
                sd.evaluate(iter, "out", 0, evaluations);
                return evaluations;
            }

        };
    }


    public static TestCase getCnn3dSynthetic() {
        return new TestCase() {
            {
                testName = "Cnn3dSynthetic";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = false;
            }

            @Override
            public ModelType modelType() {
                return ModelType.SAMEDIFF;
            }

            public Object getConfiguration() throws Exception {
                int nChannels = 3; // Number of input channels
                int outputNum = 10; // The number of possible outcomes

                SameDiff sd = SameDiff.create();


                //input in NCDHW [minibatch, channels=3, Height = 8, Width = 8, Depth = 8]
                SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, nChannels, 8, 8, 8);

                SDVariable label = sd.placeHolder("label", DataType.FLOAT, nChannels, outputNum);

                //input in NCDHW [minibatch, channels=3, Height = 8, Width = 8, Depth = 8]

                // Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels]
                // [kernelDepth = 3, kernelHeight = 3, kernelWidth = 3, inputChannels = 3, outputChannels = 8]
                SDVariable w0 = sd.var("w0", Nd4j.rand(DataType.FLOAT, 3, 3, 3, nChannels, 8));
                // Optional 1D bias array with shape [outputChannels]. May be null.
                SDVariable b0 = sd.var("b0", Nd4j.rand(DataType.FLOAT, 8));


                SDVariable layer0 = sd.nn.relu(sd.cnn.conv3d("layer0", in, w0, b0, Conv3DConfig.builder()
                        .kH(3)
                        .kW(3)
                        .kD(3)
                        .sH(2)
                        .sW(2)
                        .sD(2)
                        .dataFormat("NCDHW")
                        .build()), 0);

                // outputSize = (inputSize - kernelSize + 2*padding) / stride + 1
                // outputsize_H(W)(D) = (8 - 3 + 2*0 ) / 2 + 1 = ceil(3.5) = 4
                // [minibatch,8,4,4.4]


                SDVariable layer1 = sd.cnn.maxPooling3d("layer1", layer0, Pooling3DConfig.builder()
                        .kH(2).kW(2).kD(2)
                        .sH(2).sW(2).sD(2)
                        .isNCDHW(true)
                        .build());

                // outputSize = (inputSize - kernelSize + 2*padding) / stride + 1
                // outputsize_H(W)(D) = ( 4 - 2 + 2*0 ) / 2 + 1 = 2
                // [minibatch,8,2,2,2]


                int channels_height_width_depth = 8 * 1 * 1 * 1;

                SDVariable layer1_reshaped = layer1.reshape(-1, channels_height_width_depth);

                SDVariable w1 = sd.var("w4", Nd4j.rand(DataType.FLOAT, channels_height_width_depth, 10));
                SDVariable b1 = sd.var("b4", Nd4j.rand(DataType.FLOAT, 10));


                SDVariable out = sd.nn.softmax("out", layer1_reshaped.mmul(w1).add(b1));
                SDVariable loss = sd.loss.logLoss("loss", label, out);

                //Also set the training configuration:
                sd.setTrainingConfig(TrainingConfig.builder()
                        .updater(new Nesterovs(0.01, 0.9))
                        .dataSetFeatureMapping("in")            //features[0] -> "in" placeholder
                        .dataSetLabelMapping("label")           //labels[0]   -> "label" placeholder
                        .build());

                return sd;

            }

            @Override
            public Map<String,INDArray> getGradientsTestDataSameDiff() throws Exception {
                Nd4j.getRandom().setSeed(12345);
                //NCDHW format
                INDArray arr = Nd4j.rand(new int[]{2, 3, 8, 8, 8});
                INDArray labels = org.deeplearning4j.integration.TestUtils.randomOneHot(2, 10);

                Map<String, INDArray> map = new HashMap<>();
                map.put("in", arr);
                map.put("label", labels);
                return map;

            }



            @Override
            public List<String> getPredictionsNamesSameDiff() {

                return Collections.singletonList("out");

            }



            @Override
            public List<Map<String, INDArray>> getPredictionsTestDataSameDiff() throws Exception {

                List<Map<String, INDArray>> list = new ArrayList<>();
                INDArray arr = Nd4j.rand(new int[]{2, 3, 8, 8, 8});
                INDArray labels = org.deeplearning4j.integration.TestUtils.randomOneHot(2, 10);

                list.add(Collections.singletonMap("in", arr));
                list.add(Collections.singletonMap("label", labels));

                return list;
            }

            @Override
            public IEvaluation[] getNewEvaluations(){
                return new IEvaluation[]{new Evaluation()};
            }


        };

    }
}
