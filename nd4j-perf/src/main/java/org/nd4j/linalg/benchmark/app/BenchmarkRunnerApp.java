package org.nd4j.linalg.benchmark.app;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.benchmark.api.BenchMarkPerformer;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.reflections.Reflections;

import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Discovers all sub classes of benchmark performer on the class
 * path and runs each backend on the class path with each performer.
 *
 * You can specify the number of trials to run for each benchmark.
 *
 * @author Adam Gibson
 */
public class BenchmarkRunnerApp {
    @Option(name="--nTrials",usage="Number of trials to run",aliases = "-n")
    private int nTrials = 1000;
    @Option(name="--run",usage="Trials to run",aliases   = "-r")
    private String benchmarksToRun;

    /**
     * Do the main method
     * @param args the arguments for the method
     * @throws Exception if an exception is thrown
     */
    public void doMain(String[] args) throws Exception {
        Reflections reflections = new Reflections();
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch(CmdLineException e) {
            System.err.println(e.getMessage());
            return;
        }

        ServiceLoader<Nd4jBackend> backends = ServiceLoader.load(Nd4jBackend.class);
        Iterator<Nd4jBackend> backendIterator = backends.iterator();
        List<Nd4jBackend> allBackends = new ArrayList<>();
        Set<String> run = new HashSet<>();
        if(benchmarksToRun != null) {
            String[] split = benchmarksToRun.split(",");
            for(String s : split)
                run.add(s);
        }
        while(backendIterator.hasNext())
            allBackends.add(backendIterator.next());


        Set<Class<? extends BenchMarkPerformer>> performers = reflections.getSubTypesOf(BenchMarkPerformer.class);
        for(Class<? extends BenchMarkPerformer> perfClazz : performers) {
            if(Modifier.isAbstract(perfClazz.getModifiers()) || !run.isEmpty() && !run.contains(perfClazz.getName()))
                continue;

            Constructor<BenchMarkPerformer> performerConstructor = (Constructor<BenchMarkPerformer>) perfClazz.getConstructor(int.class);
            BenchMarkPerformer performer = performerConstructor.newInstance(nTrials);
            String begin = "=========================";
            String end = "===========================";
            System.out.println(begin + " Benchmark: " + perfClazz.getName() + " " + end);
            for(Nd4jBackend backend : backends) {
                performer.run(backend);
                System.out.println("Backend " + backend.getClass().getName() + " took (in nanoseconds) " + performer.averageTime() + " (in milliseconds) " + TimeUnit.MILLISECONDS.convert(performer.averageTime(),TimeUnit.NANOSECONDS));
            }

            System.out.println(begin + end);
        }

    }

    public static void main(String[] args) throws Exception {
        new BenchmarkRunnerApp().doMain(args);
    }

}
