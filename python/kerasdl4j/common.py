import h5py
import tempfile
import new
import numpy
import xxhash

from keras import backend as K
from os import path, mkdir
from py4j.java_gateway import JavaGateway

batch_file_template = "batch_{id}.h5"
hijack_cache = {}
gateway = JavaGateway()


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
            modelType = gateway.jvm.org.deeplearning4j.keras.model.KerasModelType.SEQUENTIAL
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.KerasModelRef.builder()
            params_builder.type(modelType)
            params_builder.modelFilePath(model_file_path)

            self._dl4j_model = gateway.sequential_to_multilayernetwork(params_builder.build())
            self._dl4j_type = modelType

        elif self.__class__.__name__ == 'Model':
            modelType = gateway.jvm.org.deeplearning4j.keras.model.KerasModelType.FUNCTIONAL
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.KerasModelRef.builder()
            params_builder.type(modelType)
            params_builder.modelFilePath(model_file_path)

            self._dl4j_model = gateway.functional_to_computationgraph(params_builder.build())
            self._dl4j_type = modelType
        else:
            raise ValueError('DL4J Keras only works with Sequential and Functional models')

        return self



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
        self._old_save(filepath,overwrite)

    else:
        check_dl4j_model(self) # enforces dl4j model for model.fn()

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
