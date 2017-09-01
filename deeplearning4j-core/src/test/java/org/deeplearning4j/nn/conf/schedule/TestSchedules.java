package org.deeplearning4j.nn.conf.schedule;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.AlphaDropout;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestSchedules {

    @Test
    public void testJson(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRateSchedule()
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).dropOut(0.7).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).dropOut(new AlphaDropout(0.5)).build())
                .build();

        fail();
    }

    @Test
    public void testMapSchedule(){

        ISchedule schedule = new MapSchedule.Builder(ScheduleType.ITERATION)
                .add(0, 0.5)
                .add(5, 0.1)
                .build();

        for( int i=0; i<10; i++ ){
            if(i < 5){
                assertEquals(0.5, schedule.valueAt(i, 0), 1e-6);
            } else {
                assertEquals(0.1, schedule.valueAt(i, 0), 1e-6);
            }
        }

    }

}
