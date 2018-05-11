package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class TestDropout extends BaseDL4JTest {

    @Test
    public void testBasicConfig(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dropOut(0.6)
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).dropOut(0.7).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).dropOut(new AlphaDropout(0.5)).build())
                .build();

        assertEquals(new Dropout(0.6), conf.getConf(0).getLayer().getIDropout());
        assertEquals(new Dropout(0.7), conf.getConf(1).getLayer().getIDropout());
        assertEquals(new AlphaDropout(0.5), conf.getConf(2).getLayer().getIDropout());


        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .dropOut(0.6)
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                .addLayer("1", new DenseLayer.Builder().nIn(10).nOut(10).dropOut(0.7).build(), "0")
                .addLayer("2", new DenseLayer.Builder().nIn(10).nOut(10).dropOut(new AlphaDropout(0.5)).build(), "1")
                .setOutputs("2")
                .build();

        assertEquals(new Dropout(0.6), ((LayerVertex)conf2.getVertices().get("0")).getLayerConf().getLayer().getIDropout());
        assertEquals(new Dropout(0.7), ((LayerVertex)conf2.getVertices().get("1")).getLayerConf().getLayer().getIDropout());
        assertEquals(new AlphaDropout(0.5), ((LayerVertex)conf2.getVertices().get("2")).getLayerConf().getLayer().getIDropout());
    }

    @Test
    public void testCalls(){

        CustomDropout d1 = new CustomDropout();
        CustomDropout d2 = new CustomDropout();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(3).dropOut(d1).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).dropOut(d2).nIn(3).nOut(3).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        List<DataSet> l = new ArrayList<>();
        l.add(new DataSet(Nd4j.rand(5,4), Nd4j.rand(5,3)));
        l.add(new DataSet(Nd4j.rand(5,4), Nd4j.rand(5,3)));
        l.add(new DataSet(Nd4j.rand(5,4), Nd4j.rand(5,3)));

        DataSetIterator iter = new ExistingDataSetIterator(l);

        net.fit(iter);
        net.fit(iter);

        List<Triple<Integer,Integer,Boolean>> expList = Arrays.asList(
                new Triple<>(0, 0, true),
                new Triple<>(1, 0, true),
                new Triple<>(2, 0, true),
                new Triple<>(3, 1, true),
                new Triple<>(4, 1, true),
                new Triple<>(5, 1, true));

        assertEquals(expList, d1.getAllCalls());
        assertEquals(expList, d2.getAllCalls());


        d1 = new CustomDropout();
        d2 = new CustomDropout();
        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(4).nOut(3).dropOut(d1).build(), "in")
                .addLayer("1", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).dropOut(d2).nIn(3).nOut(3).build(), "0")
                .setOutputs("1")
                .build();

        ComputationGraph net2 = new ComputationGraph(conf2);
        net2.init();

        net2.fit(iter);
        net2.fit(iter);

        assertEquals(expList, d1.getAllCalls());
        assertEquals(expList, d2.getAllCalls());
    }

    @Data
    public static class CustomDropout implements IDropout{
        private List<Triple<Integer,Integer,Boolean>> allCalls = new ArrayList<>();

        @Override
        public INDArray applyDropout(INDArray inputActivations, int iteration, int epoch, boolean inPlace) {
            allCalls.add(new Triple<>(iteration, epoch, inPlace));
            return inputActivations;
        }

        @Override
        public IDropout clone() {
            throw new UnsupportedOperationException();
        }
    }

    @Test
    public void testSerialization(){

        IDropout[] dropouts = new IDropout[]{
                new Dropout(0.5),
                new AlphaDropout(0.5),
                new GaussianDropout(0.1),
                new GaussianNoise(0.1)};

        for(IDropout id : dropouts) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .dropOut(id)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(4).nOut(3).build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(3).nOut(3).build())
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            TestUtils.testModelSerialization(net);

            ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .dropOut(id)
                    .graphBuilder()
                    .addInputs("in")
                    .addLayer("0", new DenseLayer.Builder().nIn(4).nOut(3).build(), "in")
                    .addLayer("1", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(3).nOut(3).build(), "0")
                    .setOutputs("1")
                    .build();

            ComputationGraph net2 = new ComputationGraph(conf2);
            net2.init();

            TestUtils.testModelSerialization(net2);
        }
    }

    @Test
    public void testDropoutValues(){
        Nd4j.getRandom().setSeed(12345);

        Dropout d = new Dropout(0.5);

        INDArray in = Nd4j.ones(10, 10);
        INDArray out = d.applyDropout(in, 0, 0, false);

        assertEquals(in, Nd4j.ones(10, 10));

        int countZeros = Nd4j.getExecutioner().exec(new MatchCondition(out, Conditions.equals(0))).z().getInt(0);
        int countTwos = Nd4j.getExecutioner().exec(new MatchCondition(out, Conditions.equals(2))).z().getInt(0);

        assertEquals(100, countZeros + countTwos);  //Should only be 0 or 2
        //Stochastic, but this should hold for most cases
        assertTrue(countZeros >= 25 && countZeros <= 75);
        assertTrue(countTwos >= 25 && countTwos <= 75);

        //Test schedule:
        d = new Dropout(new MapSchedule.Builder(ScheduleType.ITERATION).add(0, 0.5).add(5, 0.1).build());
        for( int i=0; i<10; i++ ) {
            out = d.applyDropout(in, i, 0, false);
            assertEquals(in, Nd4j.ones(10, 10));
            countZeros = Nd4j.getExecutioner().exec(new MatchCondition(out, Conditions.equals(0))).z().getInt(0);

            if(i < 5){
                countTwos = Nd4j.getExecutioner().exec(new MatchCondition(out, Conditions.equals(2))).z().getInt(0);
                assertEquals(String.valueOf(i), 100, countZeros + countTwos);  //Should only be 0 or 2
                //Stochastic, but this should hold for most cases
                assertTrue(countZeros >= 25 && countZeros <= 75);
                assertTrue(countTwos >= 25 && countTwos <= 75);
            } else {
                int countInverse = Nd4j.getExecutioner().exec(new MatchCondition(out, Conditions.equals(1.0/0.1))).z().getInt(0);
                assertEquals(100, countZeros + countInverse);  //Should only be 0 or 10
                //Stochastic, but this should hold for most cases
                assertTrue(countZeros >= 80);
                assertTrue(countInverse <= 20);
            }
        }
    }

    @Test
    public void testGaussianDropoutValues(){
        Nd4j.getRandom().setSeed(12345);

        GaussianDropout d = new GaussianDropout(0.1);   //sqrt(0.1/(1-0.1)) = 0.3333 stdev

        INDArray in = Nd4j.ones(50, 50);
        INDArray out = d.applyDropout(in, 0, 0, false);

        assertEquals(in, Nd4j.ones(50, 50));

        double mean = out.meanNumber().doubleValue();
        double stdev = out.stdNumber().doubleValue();

        assertEquals(1.0, mean, 0.05);
        assertEquals(0.333, stdev, 0.02);
    }

    @Test
    public void testGaussianNoiseValues(){
        Nd4j.getRandom().setSeed(12345);

        GaussianNoise d = new GaussianNoise(0.1);   //sqrt(0.1/(1-0.1)) = 0.3333 stdev

        INDArray in = Nd4j.ones(50, 50);
        INDArray out = d.applyDropout(in, 0, 0, false);

        assertEquals(in, Nd4j.ones(50, 50));

        double mean = out.meanNumber().doubleValue();
        double stdev = out.stdNumber().doubleValue();

        assertEquals(1.0, mean, 0.05);
        assertEquals(0.1, stdev, 0.01);
    }

    @Test
    public void testAlphaDropoutValues(){
        Nd4j.getRandom().setSeed(12345);

        double p = 0.4;
        AlphaDropout d = new AlphaDropout(p);

        double SELU_ALPHA = 1.6732632423543772;
        double SELU_LAMBDA = 1.0507009873554804;
        double alphaPrime = - SELU_LAMBDA * SELU_ALPHA;
        double a = 1.0 / Math.sqrt((p + alphaPrime * alphaPrime * p * (1-p)));
        double b = -1.0 / Math.sqrt(p + alphaPrime * alphaPrime * p * (1-p)) * (1-p) * alphaPrime;

        double actA = d.a(p);
        double actB = d.b(p);

        assertEquals(a, actA, 1e-6);
        assertEquals(b, actB, 1e-6);

        INDArray in = Nd4j.ones(10, 10);
        INDArray out = d.applyDropout(in, 0, 0, false);

        int countValueDropped = 0;
        int countEqn = 0;
        double eqn = a * 1 + b;
        double valueDropped = a * alphaPrime + b;
        for(int i=0; i<100; i++ ){
            double v = out.getDouble(i);
            if(v >= valueDropped - 1e-6 && v <= valueDropped + 1e-6){
                countValueDropped++;
            } else if(v >= eqn - 1e-6 && v <= eqn + 1e-6){
                countEqn++;
            }

        }

        assertEquals(100, countValueDropped+ countEqn);
        assertTrue(countValueDropped >= 25 && countValueDropped <= 75);
        assertTrue(countEqn >= 25 && countEqn <= 75);
    }


    @Test
    public void testSpatialDropoutValues(){
        Nd4j.getRandom().setSeed(12345);

        SpatialDropout d = new SpatialDropout(0.5);

        INDArray in = Nd4j.ones(10, 10, 5, 5);
        INDArray out = d.applyDropout(in, 0, 0, false);

        assertEquals(in, Nd4j.ones(10, 10, 5, 5));

        //Now, we expect all values for a given depth to be the same... 0 or 2
        int countZero = 0;
        int countTwo = 0;
        for( int i=0; i<10; i++ ){
            for( int j=0; j<10; j++ ){
                double value = out.getDouble(i,j,0,0);
                assertTrue( value == 0 || value == 2.0);
                INDArray exp = Nd4j.valueArrayOf(new int[]{5,5,}, value);
                INDArray act = out.get(point(i), point(j), all(), all());
                assertEquals(exp, act);

                if(value == 0.0){
                    countZero++;
                } else {
                    countTwo++;
                }
            }
        }

        //Stochastic, but this should hold for most cases
        assertTrue(countZero >= 25 && countZero <= 75);
        assertTrue(countTwo >= 25 && countTwo <= 75);

        //Test schedule:
        d = new SpatialDropout(new MapSchedule.Builder(ScheduleType.ITERATION).add(0, 0.5).add(5, 0.1).build());
        for( int i=0; i<10; i++ ) {
            out = d.applyDropout(in, i, 0, false);
            assertEquals(in, Nd4j.ones(10, 10, 5, 5));

            if(i < 5){
                countZero = 0;
                countTwo = 0;
                for( int m=0; m<10; m++ ){
                    for( int j=0; j<10; j++ ){
                        double value = out.getDouble(m,j,0,0);
                        assertTrue( value == 0 || value == 2.0);
                        INDArray exp = Nd4j.valueArrayOf(new int[]{5,5,}, value);
                        INDArray act = out.get(point(m), point(j), all(), all());
                        assertEquals(exp, act);

                        if(value == 0.0){
                            countZero++;
                        } else {
                            countTwo++;
                        }
                    }
                }

                //Stochastic, but this should hold for most cases
                assertTrue(countZero >= 25 && countZero <= 75);
                assertTrue(countTwo >= 25 && countTwo <= 75);
            } else {
                countZero = 0;
                int countInverse = 0;
                for( int m=0; m<10; m++ ){
                    for( int j=0; j<10; j++ ){
                        double value = out.getDouble(m,j,0,0);
                        assertTrue( value == 0 || value == 10.0);
                        INDArray exp = Nd4j.valueArrayOf(new int[]{5,5,}, value);
                        INDArray act = out.get(point(m), point(j), all(), all());
                        assertEquals(exp, act);

                        if(value == 0.0){
                            countZero++;
                        } else {
                            countInverse++;
                        }
                    }
                }

                //Stochastic, but this should hold for most cases
                assertTrue(countZero >= 80);
                assertTrue(countInverse <= 20);
            }
        }
    }

    @Test
    public void testSpatialDropoutValues3D(){
        Nd4j.getRandom().setSeed(12345);

        SpatialDropout d = new SpatialDropout(0.5);

        INDArray in = Nd4j.ones(10, 8, 12);
        INDArray out = d.applyDropout(in, 0, 0, false);

        assertEquals(in, Nd4j.ones(10, 8, 12));

        //Now, we expect all values for a given depth to be the same... 0 or 2
        int countZero = 0;
        int countTwo = 0;
        for( int i=0; i<10; i++ ){
            for( int j=0; j<8; j++ ){
                double value = out.getDouble(i,j,0);
                assertTrue( value == 0 || value == 2.0);
                INDArray exp = Nd4j.valueArrayOf(new int[]{1,12}, value);
                INDArray act = out.get(point(i), point(j), all());
                assertEquals(exp, act);

                if(value == 0.0){
                    countZero++;
                } else {
                    countTwo++;
                }
            }
        }

        //Stochastic, but this should hold for most cases
        assertTrue(countZero >= 20 && countZero <= 60);
        assertTrue(countTwo >= 20 && countTwo <= 60);

        //Test schedule:
        d = new SpatialDropout(new MapSchedule.Builder(ScheduleType.ITERATION).add(0, 0.5).add(5, 0.1).build());
        for( int i=0; i<10; i++ ) {
            out = d.applyDropout(in, i, 0, false);
            assertEquals(in, Nd4j.ones(10, 8, 12));

            if(i < 5){
                countZero = 0;
                countTwo = 0;
                for( int m=0; m<10; m++ ){
                    for( int j=0; j<8; j++ ){
                        double value = out.getDouble(m,j,0);
                        assertTrue( value == 0 || value == 2.0);
                        INDArray exp = Nd4j.valueArrayOf(new int[]{1, 12}, value);
                        INDArray act = out.get(point(m), point(j), all());
                        assertEquals(exp, act);

                        if(value == 0.0){
                            countZero++;
                        } else {
                            countTwo++;
                        }
                    }
                }

                //Stochastic, but this should hold for most cases
                assertTrue(countZero >= 20 && countZero <= 60);
                assertTrue(countTwo >= 20 && countTwo <= 60);
            } else {
                countZero = 0;
                int countInverse = 0;
                for( int m=0; m<10; m++ ){
                    for( int j=0; j<8; j++ ){
                        double value = out.getDouble(m,j,0);
                        assertTrue( value == 0 || value == 10.0);
                        INDArray exp = Nd4j.valueArrayOf(new int[]{1,12}, value);
                        INDArray act = out.get(point(m), point(j), all());
                        assertEquals(exp, act);

                        if(value == 0.0){
                            countZero++;
                        } else {
                            countInverse++;
                        }
                    }
                }

                //Stochastic, but this should hold for most cases
                assertTrue(countZero >= 60);
                assertTrue(countInverse <= 15);
            }
        }
    }

    @Test
    public void testSpatialDropoutJSON(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DropoutLayer.Builder(new SpatialDropout(0.5)).build())
                .build();

        String asJson = conf.toJson();
        MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(asJson);

        assertEquals(conf, fromJson);
    }

}
