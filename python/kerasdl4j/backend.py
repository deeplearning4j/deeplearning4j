import h5py
import tempfile
import sys
import numpy
import xxhash

# Python-specific imports
if sys.version_info > (3, 0):
    from types import MethodType as instancemethod
else:
    from new import instancemethod

from keras import backend as K
from os import path, mkdir
from py4j.java_gateway import JavaGateway

# cache objects for hijack
batch_file_template = "batch_{id}.h5"
hijack_cache = {}
gateway = JavaGateway()



########
# installs the DL4J backend via hijack
########

def install_dl4j_backend(model):
    """
    Hijacks the `fit` method call in the model object. Detects
    if model is Sequential or Functional.
    :param model: Model in which fit will be hijacked
    """
    # append special methods
    model._old_save = model.save
    model.save = instancemethod(_save_model, model)

    # hijack Keras API
    if model.__class__.__name__ == 'Sequential':
        # compile()
        model._old_compile = model.compile
        model.compile = instancemethod(_sequential_compile, model)
        # fit()
        model._old_fit = model.fit
        model.fit = instancemethod(_sequential_fit, model)
        # evaluate()
        model._old_evaluate = model.evaluate
        model.evaluate = instancemethod(_sequential_evaluate, model)
        # predict()
        model._old_predict = model.predict
        model.predict = instancemethod(_sequential_predict, model)
        # predict_on_batch()
        model._old_predict_on_batch = model.predict_on_batch
        model.predict_on_batch = instancemethod(_sequential_predict_on_batch, model)

    elif model.__class__.__name__ == 'Model':
        # compile()
        model._old_compile = model.compile
        model.compile = instancemethod(_functional_compile, model)
        # fit()
        model._old_fit = model.fit
        model.fit = instancemethod(_functional_fit, model)
        # evaluate()
        model._old_evaluate = model.evaluate
        model.evaluate = instancemethod(_functional_evaluate, model)
        # predict()
        model._old_predict = model.predict
        model.predict = instancemethod(_functional_predict, model)
        # predict_on_batch()
        model._old_predict_on_batch = model.predict_on_batch
        model.predict_on_batch = instancemethod(_functional_predict_on_batch, model)

    else:
        raise ValueError('DL4J Keras only works with Sequential and Functional models')

    print("Deeplearning4j backend installed to model instance")
    print("Please check main stdout for DL4J operation output")


def generate_tmp_path():
    tmp_file = tempfile.NamedTemporaryFile(prefix="dl4j")
    tmp_file.close()

    return tmp_file.name


def dump_h5(dataset, batch_size, directory_name):
    """
    Dumps the data from dataset to a series of HDF5 files. Each of them will contain at most batch_size samples.

    :param dataset: Dataset to store
    :param batch_size: Size of the batch
    :param directory_name: Directory where the batch files are going to be saved
    """

    if path.exists(directory_name):
        raise IOError("Path exists: " + directory_name)
        return

    mkdir(directory_name)

    batch_id = 0
    samples_count = dataset.shape[0]

    begin = 0
    end = batch_size

    while begin < samples_count:
        batch_file_name = batch_file_template.format(id=batch_id)
        f = h5py.File(path.join(directory_name, batch_file_name), 'w')
        f.create_dataset("data", data=dataset[begin:end])
        f.flush()
        f.close()

        begin = end
        end += batch_size
        batch_id += 1


def hash_ndarray(array):
    """
    Calculates a hash of contents of ndarray
    :param array: Array to calculate hash
    :return: hex digest of the hash (as string)
    """

    hsh = xxhash.xxh64()
    hsh.update(array.view(numpy.uint8))
    return hsh.hexdigest()


def dump_ndarray(batch_size, dataset):
    dataset_hash = hash_ndarray(dataset)
    if not dataset_hash in hijack_cache:
        directory_name = generate_tmp_path()
        dump_h5(dataset, batch_size, directory_name)
        hijack_cache[dataset_hash] = directory_name
    else:
        print("Dataset already dumped")

    return hijack_cache[dataset_hash]


def slice_X(X, start=None, stop=None):
    """This takes an array-like, or a list of
    array-likes, and outputs:
        - X[start:stop] if X is an array-like
        - [x[start:stop] for x in X] if X in a list
    Can also work on list/array of indices: `slice_X(x, indices)`
    # Arguments
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.
    """
    if isinstance(X, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return X[start]
        else:
            return X[start:stop]


def arrayhelper_to_array(
        arrayhelper):
    """
    DL4J gateway will return a flattened array with shape info in C order. This
    converts it into a numpy array.

    :param arrayhelper:
    :return:
    """
    array = numpy.array(arrayhelper[0], dtype=float, order="C").reshape(arrayhelper[1])

    return array


def arrayhelper_from_array(
        numpy_matrix):
    """
    DL4J gateway can accept a flattened array buffer, which speeds up transfer
    and avoids the tricky situation of writing to disk.

    :param numpy_matrix:
    :return:
    """
    headerlength = numpy.array('i', list(len(numpy_matrix.shape)))
    header = numpy.array('i', list(numpy_matrix.shape))
    body = numpy.array('i', numpy_matrix.flatten().tolist());
    if sys.byteorder != 'big':
        header.byteswap()
        body.byteswap()
    buf = bytearray(headerlength.tostring() + header.tostring() + body.tostring())
    return buf


def check_dl4j_model(
        self):
    """
    Checks the current Keras model object in scope
    and installs a reference to DL4J MultiLayerNetwork
    if it doesn't exist.
    """
    if hasattr(self, '_dl4j_model'):
        return self
    else:
        model_file_path = generate_tmp_path()
        self.save(model_file_path, useLegacySave=True)

        gateway = JavaGateway()
        modelType = None

        if self.__class__.__name__ == 'Sequential':
            print("New Sequential instance detected, backing with DL4J MultiLayerNetwork...")
            modelType = gateway.jvm.org.deeplearning4j.keras.model.KerasModelType.SEQUENTIAL
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.KerasModelRef.builder()
            params_builder.type(modelType)
            params_builder.modelFilePath(model_file_path)

            self._dl4j_model = gateway.sequentialToMultilayerNetwork(params_builder.build())
            self._dl4j_type = modelType

        elif self.__class__.__name__ == 'Model':
            print("New Model instance detected, backing with DL4J ComputationGraph...")
            modelType = gateway.jvm.org.deeplearning4j.keras.model.KerasModelType.FUNCTIONAL
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.KerasModelRef.builder()
            params_builder.type(modelType)
            params_builder.modelFilePath(model_file_path)

            self._dl4j_model = gateway.functionalToComputationGraph(params_builder.build())
            self._dl4j_type = modelType
        else:
            raise ValueError('DL4J Keras only works with Sequential and Functional models')

        return self


########
# hijacked functions in model class
########

########
# for Sequential(Model) instances
########

def _sequential_compile(
        self,
        optimizer,
        loss,
        metrics=None,
        sample_weight_mode=None,
        **kwargs):
    """
    Configures the learning process.
    """
    # first call the old compile() method
    self._old_compile(optimizer, loss, metrics, sample_weight_mode)

    # then convert to DL4J instance
    check_dl4j_model(self) # enforces dl4j model for model.fn()


def _sequential_fit(
        self,
        x,
        y,
        batch_size=32,
        nb_epoch=10,
        verbose=1,
        callbacks=[],
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        **kwargs):
    """
    Executes fitting of the model by using DL4J as backend
    :param model: Model to use
    :param nb_epoch: Number of learning epochs
    :param features_directory: Directory with feature batch files
    :param labels_directory: Directory with label batch files
    :return:
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    print("Performing fit() operation...")

    training_x = None
    training_y = None
    validation_x = None
    validation_y = None
    do_validation = True

    if validation_data:
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)

        if len(validation_data) == 2:
            val_x, val_y = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y)
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y)
        else:
            raise ValueError('Incorrect configuration for validation_data. Must be a tuple of length 2 or 3.')
    elif validation_split and 0. < validation_split < 1.:
        split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
        y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)
        validation_x = dump_ndarray(batch_size, val_x)
        validation_y = dump_ndarray(batch_size, val_y)
    else:
        do_validation = False
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.FitParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.nbEpoch(nb_epoch)
    params_builder.trainXPath(training_x)
    params_builder.trainYPath(training_y)
    if not validation_x == None:
        params_builder.validationXPath(validation_x)
        params_builder.validationYPath(validation_y)
    params_builder.dimOrdering(K.image_dim_ordering())
    params_builder.doValidation(do_validation)
    gateway.sequentialFit(params_builder.build())

    print("fit() operation complete")


def _sequential_evaluate(
        self,
        x,
        y,
        batch_size=32,
        verbose=1,
        sample_weight=None,
        **kwargs):
    """
    Computes the loss on some input data, batch by batch.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    print("Performing evaluate() operation...")

    features_directory = dump_ndarray(batch_size, x)
    labels_directory = dump_ndarray(batch_size, y)

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.EvaluateParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    params_builder.labelsDirectory(labels_directory)
    params_builder.batchSize(batch_size)
    ret = gateway.sequentialEvaluate(params_builder.build())

    print("evaluate() operation complete")

    return ret


def _sequential_predict(
        self,
        x,
        batch_size=32,
        verbose=0):
    """
    Generates output predictions for the input samples,
    processing the samples in a batched way.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    print("Performing predict() operation...")

    features_directory = dump_ndarray(batch_size, x)

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    params_builder.batchSize(batch_size)
    arrayhelper = gateway.sequentialPredict(params_builder.build())

    print("predict() operation complete")

    array = arrayhelper_to_array(arrayhelper)
    return array


def _sequential_predict_on_batch(
        self,
        x):
    """
    Returns predictions for a single batch of samples.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    print("Performing predict_on_batch() operation...")

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictOnBatchParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.data(arrayhelper_from_array(x)) # input becomes array buffer
    arrayhelper = gateway.sequentialPredictOnBatch(params_builder.build())

    print("predict_on_batch() operation complete")

    array = arrayhelper_to_array(arrayhelper)
    return array



########
# for Functional(Model) instances
########

def _functional_compile(
        self,
        optimizer,
        loss,
        metrics=None,
        loss_weights=None,
        sample_weight_mode=None,
        **kwargs):
    """
    Configures the model for training.
    """
    # first call the old compile() method
    self._old_compile(optimizer, loss, metrics, loss_weights, sample_weight_mode)

    # then convert to DL4J instance
    check_dl4j_model(self) # enforces dl4j model for model.fn()


def _functional_fit(
        self,
        x,
        y,
        batch_size=32,
        nb_epoch=10,
        verbose=1,
        callbacks=[],
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        **kwargs):
    """
    Executes fitting of the model by using DL4J as backend
    :param model: Model to use
    :param nb_epoch: Number of learning epochs
    :param features_directory: Directory with feature batch files
    :param labels_directory: Directory with label batch files
    :return:
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    print("Performing fit() operation...")

    training_x = None
    training_y = None
    validation_x = None
    validation_y = None
    do_validation = True

    if validation_data:
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)

        if len(validation_data) == 2:
            val_x, val_y = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y)
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y)
        else:
            raise ValueError('Incorrect configuration for validation_data. Must be a tuple of length 2 or 3.')
    elif validation_split and 0. < validation_split < 1.:
        split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
        y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)
        validation_x = dump_ndarray(batch_size, val_x)
        validation_y = dump_ndarray(batch_size, val_y)
    else:
        do_validation = False
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.FitParams.builder()
    params_builder.functionalModel(self._dl4j_model)
    params_builder.nbEpoch(nb_epoch)
    params_builder.trainXPath(training_x)
    params_builder.trainYPath(training_y)
    if not validation_x == None:
        params_builder.validationXPath(validation_x)
        params_builder.validationYPath(validation_y)
    params_builder.dimOrdering(K.image_dim_ordering())
    params_builder.doValidation(do_validation)
    gateway.functionalFit(params_builder.build())

    print("fit() operation complete")


def _functional_evaluate(
        self,
        x,
        y,
        batch_size=32,
        verbose=1,
        sample_weight=None):
    """
    Returns the loss value and metrics values for the model.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    print("Performing evaluate() operation...")

    features_directory = dump_ndarray(batch_size, x)
    labels_directory = dump_ndarray(batch_size, y)

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.EvaluateParams.builder()
    params_builder.functionalModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    params_builder.labelsDirectory(labels_directory)
    params_builder.batchSize(batch_size)
    ret = gateway.functionalEvaluate(params_builder.build())

    print("evaluate() operation complete")

    return ret


def _functional_predict(
        self,
        x,
        batch_size=32,
        verbose=0):
    """
    Generates output predictions for the input samples,
    processing the samples in a batched way.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    print("Performing predict() operation...")

    features_directory = dump_ndarray(batch_size, x)

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictParams.builder()
    params_builder.functionalModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    params_builder.batchSize(batch_size)
    gateway.functionalPredict(params_builder.build())

    print("predict() operation complete")


def _functional_predict_on_batch(
        self,
        x):
    """
    Returns predictions for a single batch of samples.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    print("Performing predict_on_batch() operation...")

    features_directory = dump_ndarray(len(x), x)

    # gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictOnBatchParams.builder()
    params_builder.functionalModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    gateway.functionalPredictOnBatch(params_builder.build())

    print("predict_on_batch() operation complete")



########
# for any Model/Sequential instances
########


def _save_model(
        self,
        filepath,
        overwrite=True,
        saveUpdaterState=False,
        useLegacySave=False):
    """
    Save model to disk in DL4J format.
    """

    if useLegacySave:
        print("Saving model using legacy methods...")
        self._old_save(filepath,overwrite)

    else:
        check_dl4j_model(self) # enforces dl4j model for model.fn()

        print("Performing save() operation...")

        if model.__class__.__name__ == 'Sequential':
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.SaveParams.builder()
            params_builder.sequentialModel(self._dl4j_model)
            params_builder.writePath(filepath)
            params_builder.saveUpdaterState(saveUpdaterState)
            gateway.sequentialSave(params_builder.build())

        elif model.__class__.__name__ == 'Model':
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.SaveParams.builder()
            params_builder.functionalModel(self._dl4j_model)
            params_builder.writePath(filepath)
            params_builder.saveUpdaterState(saveUpdaterState)
            gateway.functionalSave(params_builder.build())

        else:
            raise ValueError('DL4J Keras only works with Sequential and Functional models')

    print("save() operation complete")