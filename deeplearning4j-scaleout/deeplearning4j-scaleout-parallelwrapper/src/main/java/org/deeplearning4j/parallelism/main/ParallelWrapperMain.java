package org.deeplearning4j.parallelism.main;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.Data;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.util.ModelGuesser;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.io.File;

/**
 * Parallelwrapper main class.
 * Configure a {@link ParallelWrapper}
 * instance from the command line.
 *
 *
 *
 * @author Adam Gibson
 */
@Data
public class ParallelWrapperMain {
    @Parameter(names = {"--modelPath"}, description = "Path to the model", arity = 1, required = true)
    private String modelPath = null;
    @Parameter(names = {"--workers"}, description = "Number of workers", arity = 1)
    private int workers = 2;
    @Parameter(names = {"--prefetchSize"}, description = "The number of datasets to prefetch", arity = 1)
    private int prefetchSize = 16;
    @Parameter(names = {"--averagingFrequency"}, description = "The frequency for averaging parameters", arity = 1)
    private int averagingFrequency = 1;
    @Parameter(names = {"--reportScore"}, description = "The subcommand to run", arity = 1)
    private boolean reportScore = false;
    @Parameter(names = {"--averageUpdaters"}, description = "Whether to average updaters", arity = 1)
    private boolean averageUpdaters = true;
    @Parameter(names = {"--legacyAveraging"}, description = "Whether to use legacy averaging", arity = 1)
    private boolean legacyAveraging = true;
    @Parameter(names = {"--dataSetIteratorFactoryClazz"},
            description = "The fully qualified class name of the multi data set iterator class to use.",
            arity = 1)
    private String dataSetIteratorFactoryClazz = null;
    @Parameter(names = {"--multiDataSetIteratorFactoryClazz"},
            description = "The fully qualified class name of the multi data set iterator class to use.",
            arity = 1)
    private String multiDataSetIteratorFactoryClazz = null;
    @Parameter(names = {"--modelOutputPath"},
            description = "The fully qualified class name of the multi data set iterator class to use.",
            arity = 1, required = true)
    private String modelOutputPath = null;
    @Parameter(names = {"--uiUrl"}, description = "The host:port of the ui to use (optional)", arity = 1)
    private String uiUrl = null;



    public static void main(String[] args) throws Exception {
        new ParallelWrapperMain().runMain(args);
    }

    public void runMain(String... args) throws Exception {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            System.err.println(e.getMessage());
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            System.exit(1);
        }

        run();

    }


    public void run() throws Exception {

        Model model = ModelGuesser.loadModelGuess(modelPath);
        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                // DataSets prefetching options. Set this value with respect to number of actual devices
                .prefetchBuffer(prefetchSize)

                // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
                .workers(workers)

                // rare averaging improves performance, but might reduce model accuracy
                .averagingFrequency(averagingFrequency).averageUpdaters(averageUpdaters)

                // if set to TRUE, on every averaging model score will be reported
                .reportScoreAfterAveraging(reportScore)

                // optional parameter, set to false ONLY if your system has support P2P memory access across PCIe (hint: AWS do not support P2P)
                .useLegacyAveraging(legacyAveraging)

                .build();

        if (dataSetIteratorFactoryClazz != null) {
            DataSetIteratorProviderFactory dataSetIteratorProviderFactory =
                    (DataSetIteratorProviderFactory) Class.forName(dataSetIteratorFactoryClazz).newInstance();
            DataSetIterator dataSetIterator = dataSetIteratorProviderFactory.create();
            if (uiUrl != null) {
                // it's important that the UI can report results from parallel training
                // there's potential for StatsListener to fail if certain properties aren't set in the model
                StatsStorageRouter remoteUIRouter = new RemoteUIStatsStorageRouter("http://" + uiUrl);
                wrapper.setListeners(remoteUIRouter, new StatsListener(null));

            }
            wrapper.fit(dataSetIterator);
            ModelSerializer.writeModel(model, new File(modelOutputPath), true);


        } else if (multiDataSetIteratorFactoryClazz != null) {
            MultiDataSetProviderFactory multiDataSetProviderFactory =
                    (MultiDataSetProviderFactory) Class.forName(multiDataSetIteratorFactoryClazz).newInstance();
            MultiDataSetIterator iterator = multiDataSetProviderFactory.create();
            if (uiUrl != null) {
                // it's important that the UI can report results from parallel training
                // there's potential for StatsListener to fail if certain properties aren't set in the model
                StatsStorageRouter remoteUIRouter = new RemoteUIStatsStorageRouter("http://" + uiUrl);
                wrapper.setListeners(remoteUIRouter, new StatsListener(null));

            }
            wrapper.fit(iterator);
            ModelSerializer.writeModel(model, new File(modelOutputPath), true);

        } else {
            throw new IllegalStateException("Please provide a datasetiteraator or multi datasetiterator class");
        }


    }
}
