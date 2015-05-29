package org.nd4j.linalg.benchmark;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.benchmark.api.BenchMarkPerformer;
import org.nd4j.linalg.benchmark.elementwise.AddiRowVectorBenchmarkPerformer;
import org.nd4j.linalg.benchmark.gemm.GemmBenchmarkPerformer;
import org.nd4j.linalg.benchmark.linearview.LinearViewBenchmarkPerformer;
import org.nd4j.linalg.benchmark.linearview.getput.GetPutScalarLinearViewBenchmarkPerformer;
import org.nd4j.linalg.benchmark.scalar.ScalarBenchmarkPerformer;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.jblas.JblasBackend;

/**
 * @author Adam Gibson
 */
public class JblasTest {
    private Nd4jBackend backend = new JblasBackend();

    @Test
    public void testJblas() {
        int trials = 10;
        BenchMarkPerformer getPutScalar = new GetPutScalarLinearViewBenchmarkPerformer(trials);
        getPutScalar.run(backend);
        System.out.println("Get put linear view scalar " + getPutScalar.averageTime());


        BenchMarkPerformer performer = new AddiRowVectorBenchmarkPerformer(trials);
        assertEquals(trials, performer.nTimes());
        performer.run(backend);
        System.out.println("Addirowvector " + performer.averageTime());


        BenchMarkPerformer scalarPerformer = new ScalarBenchmarkPerformer(trials);
        scalarPerformer.run(backend);
        System.out.println("Scalar " + scalarPerformer.averageTime());


        BenchMarkPerformer linearPerformer = new LinearViewBenchmarkPerformer(trials);
        linearPerformer.run(backend);
        System.out.println("Linear view " + linearPerformer.averageTime());

        BenchMarkPerformer mmulPerf = new GemmBenchmarkPerformer(trials);
        mmulPerf.run(backend);
        System.out.println("MMUL " + mmulPerf.averageTime());









    }

}
