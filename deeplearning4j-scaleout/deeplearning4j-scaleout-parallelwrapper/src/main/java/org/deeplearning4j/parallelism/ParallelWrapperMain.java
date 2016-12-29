package org.deeplearning4j.parallelism;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;

/**
 * Created by agibsonccc on 12/29/16.
 */
public class ParallelWrapperMain {
    @Parameter(names={"--modelPath"},
            description = "Path to the model"
            , arity = 1,
            required = true)
    private String modelPath;
    @Parameter(names={"--workers"},
            description = "Number of workers"
            , arity = 1,
            required = true)
    private int workers = 2;
    @Parameter(names={"--prefetchSize"},
            description = "The number of datasets to prefetch"
            , arity = 1,
            required = true)
    private int prefetchSize = 16;
    @Parameter(names={"--averagingFrequency"},
            description = "The frequency for averaging parameters"
            , arity = 1,
            required = true)
    private int averagingFrequency = 1;
    @Parameter(names={"--reportScore"},
            description = "The subcommand to run"
            , arity = 1,
            required = true)
    private boolean reportScore = false;
    @Parameter(names={"--averageUpdaters"},
            description = "Whether to average updaters"
            , arity = 1,
            required = true)
    private boolean averageUpdaters = true;
    @Parameter(names={"--legacyAveraging"},
            description = "Whether to use legacy averaging"
            , arity = 1,
            required = true)
    private boolean legacyAveraging = true;



    public static void main(String[] args) {
        new ParallelWrapperMain().runMain(args);
    }

    public void runMain(String...args) {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch(ParameterException e) {
            System.err.println(e.getMessage());
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try{ Thread.sleep(500); } catch(Exception e2){ }
            System.exit(1);
        }
    }




    // ParallelWrapper will take care of load balancing between GPUs.
    ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(prefetchSize)

            // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
            .workers(workers)

            // rare averaging improves performance, but might reduce model accuracy
            .averagingFrequency(averagingFrequency)

            // if set to TRUE, on every averaging model score will be reported
            .reportScoreAfterAveraging(reportScore)

            // optional parameter, set to false ONLY if your system has support P2P memory access across PCIe (hint: AWS do not support P2P)
            .useLegacyAveraging(legacyAveraging)

            .build();


}
