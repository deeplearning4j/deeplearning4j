package org.deeplearning4j.optimize.listeners;

import lombok.Builder;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

/**
 * An iteration listener that provides details on parameters and gradients at each iteration during traning.
 * Attempts to provide much of the same information as the UI histogram iteration listener, but in a text-based
 * format (for example, when learning on a system accessed via SSH etc).
 * i.e., is intended to aid network tuning and debugging<br>
 * This iteration listener is set up to calculate mean, min, max, and mean absolute value
 * of each type of parameter and gradient in the network at each iteration.<br>
 * These
 *
 *
 * @author Alex Black
 */
public class ParamAndGradientIterationListener extends BaseTrainingListener {
    private static final int MAX_WRITE_FAILURE_MESSAGES = 10;
    private static final Logger logger = LoggerFactory.getLogger(ParamAndGradientIterationListener.class);

    private int iterations;
    private long totalIterationCount = 0;
    private boolean printMean = true;
    private boolean printHeader = true;
    private boolean printMinMax = true;
    private boolean printMeanAbsValue = true;
    private File file;
    private Path filePath;
    private boolean outputToConsole;
    private boolean outputToFile;
    private boolean outputToLogger;
    private String delimiter = "\t";


    private int writeFailureCount = 0;


    /** Default constructor for output to console only every iteration, tab delimited */
    public ParamAndGradientIterationListener() {
        this(1, true, true, true, true, true, false, false, null, "\t");
    }

    /**Full constructor with all options.
     * Note also: ParamAndGradientIterationListener.builder() can be used instead of this constructor.
     * @param iterations calculate and report values every 'iterations' iterations
     * @param printHeader Whether to output a header row (i.e., names for each column)
     * @param printMean Calculate and display the mean of parameters and gradients
     * @param printMinMax Calculate and display the min/max of the parameters and gradients
     * @param printMeanAbsValue Calculate and display the mean absolute value
     * @param outputToConsole If true, display the values to the console (System.out.println())
     * @param outputToFile If true, write the values to a file, one per line
     * @param outputToLogger If true, log the values
     * @param file File to write values to. May be null, not used if outputToFile == false
     * @param delimiter delimiter (for example, "\t" or "," etc)
     */
    @Builder
    public ParamAndGradientIterationListener(int iterations, boolean printHeader, boolean printMean,
                    boolean printMinMax, boolean printMeanAbsValue, boolean outputToConsole, boolean outputToFile,
                    boolean outputToLogger, File file, String delimiter) {
        this.printHeader = printHeader;
        this.printMean = printMean;
        this.printMinMax = printMinMax;
        this.printMeanAbsValue = printMeanAbsValue;
        this.iterations = iterations;
        this.file = file;
        if (this.file != null) {
            this.filePath = file.toPath();
        }
        this.outputToConsole = outputToConsole;
        this.outputToFile = outputToFile;
        this.outputToLogger = outputToLogger;
        this.delimiter = delimiter;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        totalIterationCount++;

        if (totalIterationCount == 1 && printHeader) {
            Map<String, INDArray> params = model.paramTable();
            model.conf().getVariables();

            StringBuilder sb = new StringBuilder();

            sb.append("n");
            sb.append(delimiter);
            sb.append("score");

            for (String s : params.keySet()) {
                //Parameters:
                if (printMean)
                    sb.append(delimiter).append(s).append("_mean");
                //Min, max
                if (printMinMax) {
                    sb.append(delimiter).append(s).append("_min").append(delimiter).append(s).append("_max");
                }
                if (printMeanAbsValue)
                    sb.append(delimiter).append(s).append("_meanAbsValue");

                //Gradients:
                if (printMean)
                    sb.append(delimiter).append(s).append("_meanG");
                //Min, max
                if (printMinMax) {
                    sb.append(delimiter).append(s).append("_minG").append(delimiter).append(s).append("_maxG");
                }
                if (printMeanAbsValue)
                    sb.append(delimiter).append(s).append("_meanAbsValueG");
            }
            sb.append("\n");

            if (outputToFile) {
                try {
                    Files.write(filePath, sb.toString().getBytes(), StandardOpenOption.CREATE,
                                    StandardOpenOption.TRUNCATE_EXISTING);
                } catch (IOException e) {
                    if (writeFailureCount++ < MAX_WRITE_FAILURE_MESSAGES) {
                        //Print error message
                        logger.warn("Error writing to file: {}", e);
                    }
                    if (writeFailureCount == MAX_WRITE_FAILURE_MESSAGES) {
                        logger.warn("Max file write messages displayed. No more failure messages will be printed");
                    }
                }
            }

            if (outputToLogger)
                logger.info(sb.toString());
            if (outputToConsole)
                System.out.println(sb.toString());
        }

        if (totalIterationCount % iterations != 0)
            return; //No op this iteration

        Map<String, INDArray> params = model.paramTable();
        Map<String, INDArray> grads = model.gradient().gradientForVariable();

        StringBuilder sb = new StringBuilder();
        sb.append(totalIterationCount);
        sb.append(delimiter);
        sb.append(model.score());


        //Calculate actual values for parameters and gradients
        for (Map.Entry<String, INDArray> entry : params.entrySet()) {
            INDArray currParams = entry.getValue();
            INDArray currGrad = grads.get(entry.getKey());

            //Parameters:
            if (printMean) {
                sb.append(delimiter);
                sb.append(currParams.meanNumber().doubleValue());
            }
            if (printMinMax) {
                sb.append(delimiter);
                sb.append(currParams.minNumber().doubleValue());
                sb.append(delimiter);
                sb.append(currParams.maxNumber().doubleValue());
            }
            if (printMeanAbsValue) {
                sb.append(delimiter);
                INDArray abs = Transforms.abs(currParams.dup());
                sb.append(abs.meanNumber().doubleValue());
            }

            //Gradients:
            if (printMean) {
                sb.append(delimiter);
                sb.append(currGrad.meanNumber().doubleValue());
            }
            if (printMinMax) {
                sb.append(delimiter);
                sb.append(currGrad.minNumber().doubleValue());
                sb.append(delimiter);
                sb.append(currGrad.maxNumber().doubleValue());
            }
            if (printMeanAbsValue) {
                sb.append(delimiter);
                INDArray abs = Transforms.abs(currGrad.dup());
                sb.append(abs.meanNumber().doubleValue());
            }
        }
        sb.append("\n");

        String out = sb.toString();
        if (outputToLogger)
            logger.info(out);
        if (outputToConsole)
            System.out.print(out);

        if (outputToFile) {
            try {
                Files.write(filePath, out.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            } catch (IOException e) {
                if (writeFailureCount++ < MAX_WRITE_FAILURE_MESSAGES) {
                    //Print error message
                    logger.warn("Error writing to file: {}", e);
                }
                if (writeFailureCount == MAX_WRITE_FAILURE_MESSAGES) {
                    logger.warn("Max file write messages displayed. No more failure messages will be printed");
                }
            }
        }

    }
}
