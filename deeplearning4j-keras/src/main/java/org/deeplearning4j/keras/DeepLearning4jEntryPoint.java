package org.deeplearning4j.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.keras.api.*;
import org.deeplearning4j.keras.hdf5.HDF5MiniBatchDataSetIterator;
import org.deeplearning4j.keras.model.KerasModelSerializer;
import org.deeplearning4j.keras.model.KerasModelType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

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

    /*
        Model serialization.
     */

    /**
     * Converts a Keras model to a DL4J reference.
     *
     * @param modelRef Reference to sequential model serialized on disk.
     */
    public MultiLayerNetwork sequentialToMultilayerNetwork(KerasModelRef modelRef) throws Exception {
        MultiLayerNetwork model;
        try {
            if (KerasModelType.SEQUENTIAL == modelRef.getType()) {
                model = kerasModelSerializer.readSequential(modelRef.getModelFilePath(), modelRef.getType());
            }
            else {
                throw new IllegalArgumentException("Only SEQUENTIAL and FUNCTIONAL model types are supported.");
            }

            log.info("model.sequential_to_multilayernetwork() operation complete.");

        } catch (Throwable e) {
            log.error("Error while performing model.sequential_to_multilayernetwork()", e);
            throw e;
        }

        return model;
    }

    /**
     * Converts a Keras model to a DL4J reference.
     *
     * @param modelRef Reference to functional model serialized on disk.
     */
    public ComputationGraph functionalToComputationGraph(KerasModelRef modelRef) throws Exception {
        ComputationGraph model;
        try {
            if(KerasModelType.FUNCTIONAL == modelRef.getType()) {
                model = kerasModelSerializer.readFunctional(modelRef.getModelFilePath(), modelRef.getType());
            }
            else {
                throw new IllegalArgumentException("Only SEQUENTIAL and FUNCTIONAL model types are supported.");
            }

            log.info("model.functional_to_computationgraph() operation complete.");

        } catch (Throwable e) {
            log.error("Error while performing model.functional_to_computationgraph()", e);
            throw e;
        }

        return model;
    }



    /*
        Sequential model methods.
     */

    /**
     * Performs fitting of the model which is referenced in the parameters according to learning parameters specified.
     *
     * @param fitParams Parameters for a fit operation
     */
    public void sequentialFit(FitParams fitParams) throws Exception {
        try {
            MultiLayerNetwork model = fitParams.getSequentialModel();

            DataSetIterator trainIterator;
            DataSetIterator testIterator;

            trainIterator = new HDF5MiniBatchDataSetIterator(
                fitParams.getTrainXPath(),
                fitParams.getTrainYPath()
            );

            if(fitParams.isDoValidation()) {
                testIterator = new HDF5MiniBatchDataSetIterator(
                    fitParams.getValidationXPath(),
                    fitParams.getValidationYPath()
                );
            } else {
                testIterator = null;
            }

            // loop through specified epochs
            for (int i = 0; i < fitParams.getNbEpoch(); i++) {
                log.info("Fitting epoch: " + i);

                model.fit(trainIterator);
                double score = model.score();
                log.info("Score is "+score);

                if(fitParams.isDoValidation()) {
                    log.info("Evaluating epoch: " + i);
                    Evaluation eval = model.evaluate(testIterator);
                    log.info(eval.toString());
                }
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
    public Evaluation sequentialEvaluate(EvaluateParams evaluateParams) throws Exception {
        try {
            MultiLayerNetwork model = evaluateParams.getSequentialModel();

            DataSetIterator dataSetIterator = new HDF5MiniBatchDataSetIterator(
                evaluateParams.getFeaturesDirectory(),
                evaluateParams.getLabelsDirectory()
            );

            Evaluation ret = model.evaluate(dataSetIterator);

            log.info("model.evaluate() operation complete.");

            return ret;

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
    public INDArray sequentialPredict(PredictParams predictParams) throws Exception {
        try {
            MultiLayerNetwork model = predictParams.getSequentialModel();

            DataSetIterator dataSetIterator = new HDF5MiniBatchDataSetIterator(
                predictParams.getFeaturesDirectory()
            );

            INDArray ret = model.output(dataSetIterator);

            log.info("model.predict() operation complete.");

            return ret;

        } catch (Throwable e) {
            log.error("Error while performing model.predict()", e);
            throw e;
        }
    }

    /**
     * Predict the labels of features in a dataset.
     *
     * @param predictParams A dataset and associated parameters
     */
    public INDArray sequentialPredictOnBatch(PredictOnBatchParams predictParams) throws Exception {
        try {
            MultiLayerNetwork model = predictParams.getSequentialModel();

            DataSetIterator dataSetIterator = new HDF5MiniBatchDataSetIterator(
                predictParams.getFeaturesDirectory()
            );

            INDArray ret = model.output(dataSetIterator);

            log.info("model.predict_on_batch() operation complete.");

            return ret;

        } catch (Throwable e) {
            log.error("Error while performing model.predict_on_batch()", e);
            throw e;
        }
    }

    /**
     * Save a model into the DL4J format.
     *
     * @param saveParams Current model in scope and associated save parameters
     * @throws Exception
     */
    public void sequentialSave(SaveParams saveParams) throws Exception {
        try {
            ModelSerializer.writeModel(
                saveParams.getSequentialModel(),
                new File(saveParams.getWritePath()),
                true
            );

        } catch (Throwable e) {
            log.error("Error while performing model.save_model()", e);
            throw e;
        }
    }



    /*
        Functional model methods.
     */

    /**
     * Performs fitting of the model which is referenced in the parameters according to learning parameters specified.
     *
     * @param fitParams Parameters for a fit operation
     */
    public void functionalFit(FitParams fitParams) throws Exception {
        try {
            ComputationGraph model = fitParams.getFunctionalModel();

            DataSetIterator trainIterator;
            DataSetIterator testIterator;

            trainIterator = new HDF5MiniBatchDataSetIterator(
                fitParams.getTrainXPath(),
                fitParams.getTrainYPath()
            );

            if(fitParams.isDoValidation()) {
                testIterator = new HDF5MiniBatchDataSetIterator(
                    fitParams.getValidationXPath(),
                    fitParams.getValidationYPath()
                );
            } else {
                testIterator = null;
            }

            // loop through specified epochs
            for (int i = 0; i < fitParams.getNbEpoch(); i++) {
                log.info("Fitting epoch: " + i);

                model.fit(trainIterator);
                double score = model.score();
                log.info("Score is "+score);

                if(fitParams.isDoValidation()) {
                    log.info("Evaluating epoch: " + i);
                    Evaluation eval = model.evaluate(testIterator);
                    log.info(eval.toString());
                }
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
    public Evaluation functionalEvaluate(EvaluateParams evaluateParams) throws Exception {
        try {
            ComputationGraph model = evaluateParams.getFunctionalModel();

            DataSetIterator dataSetIterator = new HDF5MiniBatchDataSetIterator(
                evaluateParams.getFeaturesDirectory(),
                evaluateParams.getLabelsDirectory()
            );

            Evaluation ret = model.evaluate(dataSetIterator);

            log.info("model.evaluate() operation complete.");

            return ret;

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
    public List<INDArray[]> functionalPredict(PredictParams predictParams) throws Exception {
        try {
            ComputationGraph model = predictParams.getFunctionalModel();

            DataSetIterator dataSetIterator = new HDF5MiniBatchDataSetIterator(
                predictParams.getFeaturesDirectory()
            );

            List<INDArray[]> ret = new LinkedList<>();

            while(dataSetIterator.hasNext()) {
                DataSet data = dataSetIterator.next();
                ret.add(model.output(data.getFeatures()));
            }

            log.info("model.predict() operation complete.");

            return ret;

        } catch (Throwable e) {
            log.error("Error while performing model.predict()", e);
            throw e;
        }
    }

    /**
     * Predict the labels of features in a dataset.
     *
     * @param predictParams A dataset and associated parameters
     */
    public List<INDArray[]> functionalPredictOnBatch(PredictOnBatchParams predictParams) throws Exception {
        try {
            ComputationGraph model = predictParams.getFunctionalModel();

            DataSetIterator dataSetIterator = new HDF5MiniBatchDataSetIterator(
                predictParams.getFeaturesDirectory()
            );

            List<INDArray[]> ret = new LinkedList<>();

            while(dataSetIterator.hasNext()) {
                DataSet data = dataSetIterator.next();
                ret.add(model.output(data.getFeatures()));
            }

            log.info("model.predict_on_batch() operation complete.");

            return ret;

        } catch (Throwable e) {
            log.error("Error while performing model.predict_on_batch()", e);
            throw e;
        }
    }

    /**
     * Save a model into the DL4J format.
     *
     * @param saveParams Current model in scope and associated save parameters
     * @throws Exception
     */
    public void functionalSave(SaveParams saveParams) throws Exception {
        try {
            ModelSerializer.writeModel(
                saveParams.getFunctionalModel(),
                new File(saveParams.getWritePath()),
                true
            );

            log.info("model.save() operation complete.");

        } catch (Throwable e) {
            log.error("Error while performing model.save()", e);
            throw e;
        }
    }

}
