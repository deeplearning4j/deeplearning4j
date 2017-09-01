package org.nd4j.linalg.schedule;

import org.junit.Test;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import static org.junit.Assert.assertEquals;

public class TestSchedules {

    @Test
    public void testJson() throws Exception {

        ObjectMapper om = new ObjectMapper();
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        om.enable(SerializationFeature.INDENT_OUTPUT);

        ISchedule[] schedules = new ISchedule[]{
                new ExponentialSchedule(ScheduleType.ITERATION, 1.0, 0.5),
                new InverseSchedule(ScheduleType.ITERATION, 1.0, 0.5, 2),
                new MapSchedule.Builder(ScheduleType.ITERATION).add(0, 1.0).add(10,0.5).build(),
                new PolySchedule(ScheduleType.ITERATION, 1.0, 2, 100),
                new SigmoidSchedule(ScheduleType.ITERATION, 1.0, 0.5, 10),
                new StepSchedule(ScheduleType.ITERATION, 1.0, 0.9, 100)};


        for(ISchedule s : schedules){
            String json = om.writeValueAsString(s);
            ISchedule fromJson = om.readValue(json, ISchedule.class);
            assertEquals(s, fromJson);
        }
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
