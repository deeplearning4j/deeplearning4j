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
                new StepSchedule(ScheduleType.ITERATION, 1.0, 0.9, 100),
                new CycleSchedule(ScheduleType.ITERATION, 1.5, 100)
        };


        for(ISchedule s : schedules){
            String json = om.writeValueAsString(s);
            ISchedule fromJson = om.readValue(json, ISchedule.class);
            assertEquals(s, fromJson);
        }
    }

    @Test
    public void testScheduleValues(){

        double lr = 0.8;
        double decay = 0.9;
        double power = 2;
        double gamma = 0.5;
        int step = 20;

        for(ScheduleType st : ScheduleType.values()) {

            ISchedule[] schedules = new ISchedule[]{
                    new ExponentialSchedule(st, lr, gamma),
                    new InverseSchedule(st, lr, gamma, power),
                    new PolySchedule(st, lr, power, step),
                    new SigmoidSchedule(st, lr, gamma, step),
                    new StepSchedule(st, lr, decay, step)};

            for(ISchedule s : schedules) {

                for (int i = 0; i < 9; i++) {
                    int epoch = i / 3;
                    int x;
                    if (st == ScheduleType.ITERATION) {
                        x = i;
                    } else {
                        x = epoch;
                    }

                    double now = s.valueAt(i, epoch);
                    double e;
                    if(s instanceof ExponentialSchedule){
                        e = calcExponentialDecay(lr, gamma, x);
                    } else if(s instanceof InverseSchedule){
                        e = calcInverseDecay(lr, gamma, x, power);
                    } else if(s instanceof PolySchedule){
                        e = calcPolyDecay(lr, x, power, step);
                    } else if(s instanceof SigmoidSchedule){
                        e = calcSigmoidDecay(lr, gamma, x, step);
                    } else if(s instanceof StepSchedule){
                        e = calcStepDecay(lr, decay, x, step);
                    } else {
                        throw new RuntimeException();
                    }

                    assertEquals(s.toString() + ", " + st, e, now, 1e-6);
                }
            }
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
    @Test
    public void testCycleSchedule(){
        ISchedule schedule = new CycleSchedule(ScheduleType.ITERATION, 1.5, 100);
        assertEquals(0.15, schedule.valueAt(0, 0), 1e-6);
        assertEquals(1.5, schedule.valueAt(45, 0), 1e-6);
        assertEquals(0.15, schedule.valueAt(90, 0), 1e-6);
        assertEquals(0.015, schedule.valueAt(91, 0), 1e-6);

        schedule = new CycleSchedule(ScheduleType.ITERATION, 0.95, 0.85, 100, 10, 1);
        assertEquals(0.95, schedule.valueAt(0, 0), 1e-6);
        assertEquals(0.85, schedule.valueAt(45, 0), 1e-6);
        assertEquals(0.95, schedule.valueAt(90, 0), 1e-6);
        assertEquals(0.95, schedule.valueAt(91, 0), 1e-6);
    }

    private static double calcExponentialDecay(double lr, double decayRate, double iteration) {
        return lr * Math.pow(decayRate, iteration);
    }

    private static double calcInverseDecay(double lr, double decayRate, double iteration, double power) {
        return lr / Math.pow((1 + decayRate * iteration), power);
    }

    private static double calcStepDecay(double lr, double decayRate, double iteration, double steps) {
        return lr * Math.pow(decayRate, Math.floor(iteration / steps));
    }

    private static double calcPolyDecay(double lr, double iteration, double power, double maxIterations) {
        return lr * Math.pow(1 + iteration / maxIterations, power);
    }

    private static double calcSigmoidDecay(double lr, double decayRate, double iteration, double steps) {
        return lr / (1 + Math.exp(-decayRate * (iteration - steps)));
    }

}
