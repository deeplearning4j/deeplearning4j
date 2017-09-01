package org.deeplearning4j.nn.conf.schedule;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestSchedules {

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
