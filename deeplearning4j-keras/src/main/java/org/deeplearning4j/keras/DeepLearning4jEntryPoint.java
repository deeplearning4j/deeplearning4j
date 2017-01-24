package org.deeplearning4j.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.keras.api.*;
import org.deeplearning4j.keras.hdf5.HDF5MiniBatchDataSetIterator;
import org.deeplearning4j.keras.model.KerasModelSerializer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * The API exposed to the Python side.
 *
 * The python wrapper reads these methods and instantiates DL4J objects and operations
 * accordingly. The strategy is to capture a Keras model and convert it to a DL4J instance
 * when necessary. This includes hijacking fit, evaluate, predict, and related methods.
 */
@Slf4j
public class DeepLearning4jEntryPoint {

    private final KerasModelSerializer kerasModelSerializer = new KerasModelSerializer();

    /**
     * Keras compile is hijacked to return a DL4J Model instance.
     *
     * @param compileParams Parameters for the model compile operation
     * @throws Exception
     */
    public void compile(CompileParams compileParams) throws Exception {
        // TODO
        ComputationGraph model = kerasModelSerializer.read(writeParams.getModelFilePath(), writeParams.getType());
    }

    /**
     * Performs fitting of the model which is referenced in the parameters according to learning parameters specified.
     *
     * @param fitParams Parameters for a fit operation
     */
    public void fit(FitParams fitParams) throws Exception {
        try {
            ComputationGraph model = fitParams.getModel();

            DataSetIterator dataSetIterator = new HDF5MiniBatchDataSetIterator(
                fitParams.getTrainFeaturesDirectory(),
                fitParams.getTrainLabelsDirectory()
            );

            for (int i = 0; i < fitParams.getNbEpoch(); i++) {
                log.info("Fitting: " + i);

                model.fit(dataSetIterator);
            }

            log.info("model.fit() operation complete.");

        } catch (Throwable e) {
            log.error("Error while performing model.fit()", e);
            throw e;
        }
    }

    /**
     * Evaluates a model using a given dataset.
     *
     * @param evaluateParams Parameters for a Keras evaluate operation
     */
    public void evaluate(EvaluateParams evaluateParams) throws Exception {
        try {
            ComputationGraph model = evaluateParams.getModel();

            DataSetIterator dataSetIterator = new HDF5MiniBatchDataSetIterator(
                fitParams.getTrainFeaturesDirectory(),
                fitParams.getTrainLabelsDirectory()
            );

            model.evaluate(dataSetIterator);

            log.info("model.evaluate() operation complete.");

        } catch (Throwable e) {
            log.error("Error while performing model.evaluate()", e);
            throw e;
        }
    }

    /**
     * Predict the label of a single feature.
     *
     * @param predictParams A single feature and associated parameters
     */
    public void predict(PredictParams predictParams) throws Exception {
        try {
            ComputationGraph model = predictParams.getModel();

            model.predict(x);

            log.info("model.predict() operation complete.");

        } catch (Throwable e) {
            log.error("Error while performing model.predict()", e);
            throw e;
        }
    }

    /**
     * Predict the labels of features in a dataset.
     *
     * @param predictParams A dataset and assocated parameters
     */
    public void predict_on_batch(PredictBatchParams predictParams) throws Exception {
        try {
            ComputationGraph model = predictParams.getModel();

            model.predict(dataSet);

            log.info("model.predict_on_batch() operation complete.");

        } catch (Throwable e) {
            log.error("Error while performing model.predict_on_batch()", e);
            throw e;
        }
    }

    /**
     * Load a Keras model config into the current model.
     *
     * @param configParams Keras model config and associated parameters.
     */
    public void from_config(FromConfigParams configParams) throws Exception {
        ComputationGraph model;
        try {
            model = kerasModelSerializer.readConfig(configParams);

            log.info("model.from_config() operation complete.");

        } catch (Throwable e) {
            log.error("Error while performing model.from_config()", e);
            throw e;
        }

        return model;
    }

    /**
     * Save a model into the DL4J format.
     *
     * @param saveParams Current model in scope and assocaited save parameters
     * @throws Exception
     */
    public void save_model(SaveParams saveParams) throws Exception {
        try {
            ModelSerializer.writeModel(
                saveParams.getModel(),
                new File(writeParams.getWritePath()),
                true
            );

        } catch (Throwable e) {
            log.error("Error while performing model.save_model()", e);
            throw e;
        }
    }

}
