package org.deeplearning4j.optimize.listeners.callbacks;

import lombok.NonNull;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;

/**
 * This callback will save model after each EvaluativeListener invocation.
 *
 * Filename template respects %d pattern, which will be replaced with integer value representing invocation number (not iteration!).
 * I.e. if EvaluativeListener has frequency set to 50, it will be invoked once every 50 iterations, each invocation will increment number by 1. So, after 500 epochs there will be 10 invocations in total, and 10 models will be saved.
 *
 * PLEASE NOTE:
 * @author raver119@gmail.com
 */
public class ModelSavingCallback implements EvaluationCallback {
    protected File rootFolder;
    protected String template;

    /**
     * This constructor will create ModelSavingCallback instance that will save models in current folder
     *
     * PLEASE NOTE: Make sure you have write access to the current folder
     *
     * @param fileNameTemplate
     */
    public ModelSavingCallback(@NonNull String fileNameTemplate) {
        this(new File("./"), fileNameTemplate);
    }

    /**
     * This constructor will create ModelSavingCallback instance that will save models in specified folder
     *
     * PLEASE NOTE: Make sure you have write access to the target folder
     *
     * @param rootFolder File object referring to target folder
     * @param fileNameTemplate
     */
    public ModelSavingCallback(@NonNull File rootFolder, @NonNull String fileNameTemplate) {
        if (!rootFolder.isDirectory())
            throw new DL4JInvalidConfigException("rootFolder argument should point to valid folder");

        if (fileNameTemplate.isEmpty())
            throw new DL4JInvalidConfigException("Filename template can't be empty String");

        this.rootFolder = rootFolder;
        this.template = fileNameTemplate;
    }

    @Override
    public void call(EvaluativeListener listener, Model model, long invocationsCount, IEvaluation[] evaluations) {

        String temp = template.replaceAll("%d", "" + invocationsCount);

        String finalName = FilenameUtils.concat(rootFolder.getAbsolutePath(), temp);
        save(model, finalName);
    }


    /**
     * This method saves model
     *
     * @param model
     * @param filename
     */
    protected void save(Model model, String filename) {
        try {
            ModelSerializer.writeModel(model, filename, true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
