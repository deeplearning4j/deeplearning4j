#include <graph/GraphExecutioner.h>
#include <graph/GraphHolder.h>
#include <helpers/ConstantTadHelper.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>

#include "helpers/OpTracker.h"

static long lengthInBytes(OpaqueDataBuffer *buffer) {
  return buffer->dataBuffer()->getLenInBytes();
}

template <typename T>
static sd::Pointer _numpyHeaderForNd4j(sd::Pointer data, const sd::Pointer shapeBuffer, sd::LongType wordSize,
                                       sd::LongType* headerSize) {
  sd::LongType const* shapeBufferCast = reinterpret_cast<const sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  const sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  char* ret = new char[npHeader.size() + 1];
  int count = 0;
  for (int i = 0; i < npHeader.size(); i++) {
    ret[count] = npHeader[i];
    count++;
  }

  ret[count] = '\0';
  count++;

  *headerSize = count;
  return reinterpret_cast<sd::Pointer>(ret);
}


 sd::Pointer numpyHeaderForNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize,
                                      sd::LongType* headerSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);
  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderForNd4j, (data, shapeBuffer, wordSize, headerSize), SD_COMMON_TYPES);
}

/**
 * Load numpy from a header
 * based on the cnpy parse from header method.
 * @param data the header data to parse
 * @return a pointer to a numpy cnpy:NpyArray struct
 */
 sd::Pointer loadNpyFromHeader(sd::Pointer data) {
  char* header = reinterpret_cast<char*>(data);

  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(header);
  cnpy::NpyArray* ret = new cnpy::NpyArray();
  int totalLengthOfShape = 1;
  for (int i = 0; i < arr.shape.size(); i++) {
    totalLengthOfShape *= arr.shape[i];
  }

  ret->data = arr.data;
  ret->wordSize = arr.wordSize;
  ret->shape = arr.shape;
  return reinterpret_cast<sd::Pointer>(ret);
}


/**
 * Create a numpy array from an nd4j
 * array
 * @param data a pointer to the data
 * @param shapeBuffer  the shapebuffer for the nd4j array
 * @param wordSize  the word size (4 for float, 8 for doubles)
 * @return a pointer to a numpy array
 */

template <typename T>
 sd::Pointer _numpyFromNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize) {
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>( npShape, rank, wordSize);
  char* dataChar = reinterpret_cast<char*>(data);
  char* npHeaderData = npHeader.data();
  char* ret = new char[(wordSize * length) + npHeader.size()];
  char* cursorStart = ret + npHeader.size();
  std::memcpy(ret, npHeaderData,
              npHeader.size());
  std::memcpy(cursorStart, dataChar,length  * wordSize);
  sd::Pointer rettPointer = reinterpret_cast<sd::Pointer>(ret);
  return rettPointer;
}
template<typename T>
 long _numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer) {
  sd::LongType wordSize = opaqueDataBuffer->dataBuffer()->getLenInBytes() / opaqueDataBuffer->dataBuffer()->getNumElements();
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  long ret = npHeader.size();
  return ret;
}

template<typename  T>
 long _numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize) {
  sd::LongType* shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  int rank = shape::rank(shapeBufferCast);
  sd::LongType* shape = shape::shapeOf(shapeBufferCast);
  unsigned int* npShape = new unsigned int[rank];
  for (int i = 0; i < rank; i++) {
    npShape[i] = shape[i];
  }

  sd::LongType length = shape::prodLong(shape, rank);
  auto npHeader = cnpy::createNpyHeader<T>(npShape, rank, wordSize);
  long ret = npHeader.size();
  return ret;
}



 long numpyHeaderLengthWordSize(sd::Pointer shapeBuffer,long wordSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);
  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderLengthWordSize, (shapeBuffer, wordSize), SD_COMMON_TYPES);

}

 long numpyHeaderLength(OpaqueDataBuffer *opaqueDataBuffer,sd::Pointer shapeBuffer) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);

  BUILD_SINGLE_SELECTOR(type, return _numpyHeaderLength, (opaqueDataBuffer, shapeBuffer), SD_COMMON_TYPES);

}



 sd::Pointer numpyFromNd4j(sd::Pointer data, sd::Pointer shapeBuffer, sd::LongType wordSize) {
  auto shapeBufferCast = reinterpret_cast<sd::LongType*>(shapeBuffer);
  auto type = sd::ArrayOptions::dataType(shapeBufferCast);

  BUILD_SINGLE_SELECTOR(type, return _numpyFromNd4j, (data, shapeBuffer, wordSize), SD_COMMON_TYPES);
}


sd::Pointer shapeBufferForNumpy(sd::Pointer npyArray) {
  try {
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    unsigned int shapeSize = arr.shape.size();
    std::vector<sd::LongType> shape(shapeSize);
    bool _empty = false;
    for (unsigned int i = 0; i < shapeSize; i++) {
      shape[i] = arr.shape[i];

      if (arr.shape[i] == 0) _empty = true;
    }

    auto dtype = cnpy::dataTypeFromHeader(reinterpret_cast<char *>(npyArray));

    sd::LongType *shapeBuffer;
    if (shape.size() == 1 && shape[0] == 0) {
      // scalar case
      shapeBuffer = sd::ShapeBuilders::createScalarShapeInfo(dtype);
    } else if (_empty) {
      if (shapeSize > 0)
        shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
      else
        shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype);
    } else {
      shapeBuffer = sd::ShapeBuilders::createShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
    }
    return (sd::Pointer)(sd::ConstantShapeHelper::getInstance().createFromExisting(
        shapeBuffer, true));  // TO DO: this can lead to unpleasant crash sometimes
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

OpaqueNDArray getOutputArrayNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return nullptr;
  return ptr->outputArray(idx);
}


OpaqueNDArray getInputArrayNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return nullptr;
  return ptr->array(idx);
}


sd::LongType dataTypeNativeAt(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return 0;
  return static_cast<sd::LongType>(ptr->dataType(idx));

}


bool bArgAtNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return false;
  return ptr->getBArguments()->at(idx);

}

sd::LongType iArgumentAtNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return 0;
  return ptr->getIArguments()->at(idx);

}

sd::LongType numDNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numD();
}

sd::LongType numBNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numB();
}

sd::LongType numOutputsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->outputWidth();
}
sd::LongType numInputsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->width();
}

double tArgumentNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return 0.0;
  return ptr->getTArguments()->at(idx);
}

sd::LongType numTArgumentsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numT();
}

sd::LongType numIArgumentsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numI();
}




void setGraphContextOutputArray(OpaqueContext* ptr, int index,OpaqueNDArray arr) {
  if(arr == nullptr)
    THROW_EXCEPTION("setGraphContextOutputArray: Input arrays were null!");

  ptr->setOutputArray(index,arr,false);


}

void setGraphContextInputArray(OpaqueContext* ptr,int index,OpaqueNDArray arr) {
  if(arr == nullptr)
    THROW_EXCEPTION("setGraphContextInputArray: Input arrays were null!");

  ptr->setInputArray(index, arr, false);

}

//note here for javacpp mapping we have to use this odd type alias as a pointer
//to make the typedef work properly.
void setGraphContextOutputArraysArr(OpaqueContext* ptr, int numArrays,OpaqueNDArrayArr *arr) {
  if (arr == nullptr) THROW_EXCEPTION("setGraphContextOutputArraysArr: Input arrays were null!");
  for (int i = 0; i < numArrays; i++) {
    if (arr[i] == nullptr) {
      std::string errorMessage;
      errorMessage += "setGraphContextOutputArraysArr: Input array at index ";
      errorMessage += std::to_string(i);
      errorMessage += " was null!";
      THROW_EXCEPTION(errorMessage.c_str());
    }
    for (int i = 0; i < numArrays; i++) {
      ptr->setOutputArray(i, *arr[i], false);
    }
  }
}


sd::Pointer createUtf8String(sd::Pointer *extraPointers, const char *string, int length) {
  auto u = new sd::utf8string(string, length);
  return reinterpret_cast<sd::Pointer>(u);
}

sd::LongType getUtf8StringLength(sd::Pointer *extraPointers, sd::Pointer ptr) {
  return reinterpret_cast<sd::utf8string *>(ptr)->_length;
}
char *getUtf8StringBuffer(sd::Pointer *extraPointers, sd::Pointer ptr) {
  return reinterpret_cast<sd::utf8string *>(ptr)->_buffer;
}

void deleteUtf8String(sd::Pointer *extraPointers, sd::Pointer ptr) { delete (reinterpret_cast<sd::utf8string *>(ptr)); }

int dataTypeFromNpyHeader(void *header) { return (int)cnpy::dataTypeFromHeader(reinterpret_cast<char *>(header)); }



OpaqueConstantShapeBuffer shapeBufferEx(int rank, sd::LongType *shape, sd::LongType *strides, sd::DataType dtype,
                                        char order,
                                        sd::LongType ews, sd::LongType extras) {
  try {

    auto desc = new sd::ShapeDescriptor(dtype, order, shape, strides, rank, extras);
    auto buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    return buffer;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

void deleteConstantShapeBuffer(OpaqueConstantShapeBuffer *ptr) { }

void deleteConstantDataBuffer(OpaqueConstantDataBuffer *ptr) {
  delete ptr;
}


sd::LongType  *mmapFile(sd::Pointer *extraPointers, const char *fileName, sd::LongType  length) { return nullptr; }

void munmapFile(sd::Pointer *extraPointers, sd::LongType  *ptrMap, sd::LongType  length) {}

ResultWrapper *executeFlatGraph(sd::Pointer *extraPointers, sd::Pointer flatBufferPointer) {
  try {
    return sd::graph::GraphExecutioner::executeFlatBuffer(flatBufferPointer);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::LongType  getResultWrapperSize(ResultWrapper *ptr) { return ptr->size(); }
sd::Pointer getResultWrapperPointer(ResultWrapper *ptr) { return ptr->pointer(); }

const char *getAllCustomOps() { return sd::ops::OpRegistrator::getInstance().getAllCustomOperations(); }

sd::ShapeList *_calculateOutputShapes(sd::Pointer *extraPointers, sd::ops::DeclarableOp *op, sd::Pointer *inputBuffers,
                                      sd::Pointer *inputShapes, int numInputShapes, double *tArgs, int numTArgs,
                                  sd::LongType  *iArgs, int numIArgs, bool *bArgs, int numBArgs, int *dArgs, int numDArgs,
                                  sd::LongType  *offsets) {

  sd::graph::VariableSpace varSpace;
  Context block(2, &varSpace);
  sd::ShapeList inShapes;

  for (int e = 0; e < numIArgs; e++) block.getIArguments()->push_back(iArgs[e]);

  for (int e = 0; e < numTArgs; e++) block.getTArguments()->push_back(tArgs[e]);

  for (int e = 0; e < numBArgs; e++) block.getBArguments()->push_back(bArgs[e]);

  for (int e = 0; e < numDArgs; e++) block.getDArguments()->push_back(sd::DataTypeUtils::fromInt(dArgs[e]));

  for (int e = 0; e < numInputShapes; e++) {
    auto shape_ = reinterpret_cast<sd::LongType  *>(inputShapes[e]);
    if(shape_ == nullptr) {
      THROW_EXCEPTION("Input shape was null!");
    }

    if((shape_ != nullptr && shape_[0] > SD_MAX_RANK) || shape_[0] < 0) {
      THROW_EXCEPTION("Input shape rank is invalid. Either > 32 or < 0. Likely corrupt. Please check your input shapes.");
    }



    // we shouldn't copy buffer if that's empty array
    void *buffer_ = sd::ArrayOptions::arrayType(shape_) == sd::ArrayType::EMPTY ? nullptr : inputBuffers[e];

    auto array = new sd::NDArray(buffer_, shape_, block.launchContext(), 0, offsets[e]);


    // block should contain references to proper variable
    varSpace.putVariable(1, e, array);
    block.pickInput(1, e);

    inShapes.push_back(shape_);
  }

  auto status = op->validateDataTypes(block);
  if (status != sd::Status::OK) THROW_EXCEPTION("Data types validation failed");

  auto shapeList = op->calculateOutputShape(&inShapes, block);

  if (varSpace.launchContext() != nullptr) shapeList->detach();

  return shapeList;
}

sd::ShapeList *calculateOutputShapes2(sd::Pointer *extraPointers, sd::LongType  hash, sd::Pointer *inputBuffers, sd::Pointer *inputShapes,
                                  int numInputShapes, double *tArgs, int numTArgs, sd::LongType  *iArgs, int numIArgs,
                                  bool *bArgs, int numBArgs, int *dArgs, int numDArgs,
                                  sd::LongType  *offsets) {
  try {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    return _calculateOutputShapes(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs,
                                  numIArgs, bArgs, numBArgs, dArgs, numDArgs, nullptr);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::ShapeList *_calculateOutputShapes(sd::Pointer *extraPointers, sd::ops::DeclarableOp *op, sd::Pointer *inputShapes,
                                  int numInputShapes, double *tArgs, int numTArgs, sd::LongType  *iArgs, int numIArgs) {
  Context block(1);
  sd::ShapeList inShapes;

  for (int e = 0; e < numIArgs; e++) block.getIArguments()->push_back(iArgs[e]);

  for (int e = 0; e < numTArgs; e++) block.getTArguments()->push_back(tArgs[e]);

  for (int e = 0; e < numInputShapes; e++) inShapes.push_back(reinterpret_cast<sd::LongType  *>(inputShapes[e]));

  auto shapeList = op->calculateOutputShape(&inShapes, block);

  return shapeList;
}

sd::ShapeList *calculateOutputShapes(sd::Pointer *extraPointers, sd::LongType  hash, sd::Pointer *inputShapes, int numInputShapes,
                                 double *tArgs, int numTArgs, sd::LongType  *iArgs, int numIArgs) {
  try {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);

    return _calculateOutputShapes(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::LongType  getShapeListSize(sd::ShapeList *list) { return list->size(); }

sd::LongType  const *getShape(sd::ShapeList *list, sd::LongType  i) { return list->at(i); }

sd::Status realExec(sd::ops::DeclarableOp *op, sd::Pointer *extraPointers, sd::LongType  hash, sd::Pointer *inputBuffers,
                    sd::Pointer *inputShapes, int numInputs, sd::Pointer *outputBuffers, sd::Pointer *outputShapes,
                int numOutputs, double *tArgs, int numTArgs, sd::LongType  *iArgs, int numIArgs, bool *bArgs,
                int numBArgs, bool isInplace) {
  if (op == nullptr) sd_printf("Can't find requested operation: [%lld]\n", hash);

  // we're using the same fake nodeId everywhere here

  std::vector<sd::NDArray *> inputs(numInputs);
  std::vector<sd::NDArray *> outputs(numOutputs);
  std::vector<double> ttArgs(numTArgs);
  std::vector<sd::LongType > iiArgs(numIArgs);
  std::vector<bool> biArgs(numBArgs);

  // filling block now with inputs
  for (int e = 0; e < numInputs; e++) {
    auto shape = reinterpret_cast<sd::LongType  *>(inputShapes[e]);
    void *buffer = sd::ArrayOptions::arrayType(shape) == sd::ArrayType::EMPTY ? nullptr : inputBuffers[e];

    inputs[e] = new sd::NDArray(buffer, shape, nullptr, 0, 0);
  }

  // if not inplace - transferring output arrays

  if (!isInplace)
    for (int e = 0; e < numOutputs; e++) {
      // we want to keep original output shape intact
      auto shape = shape::copyShape(reinterpret_cast<sd::LongType  *>(outputShapes[e]));
      void *buffer = sd::ArrayOptions::arrayType(shape) == sd::ArrayType::EMPTY ? nullptr : outputBuffers[e];

      // FIXME: revisit this.
      bool canNullify = true;
      for (int i = 0; i < numInputs; i++) {
        void *ibuffer = sd::ArrayOptions::arrayType(shape) == sd::ArrayType::EMPTY ? nullptr : inputBuffers[i];
        if (ibuffer == buffer) {
          canNullify = false;
          break;
        }
      }

      if (canNullify)
        memset((uint8_t *)buffer, '\0',
               shape::length(shape) * sd::DataTypeUtils::sizeOfElement(sd::ArrayOptions::dataType(shape)));

      auto array = new sd::NDArray(buffer, shape, nullptr, 0, 0);
      outputs[e] = array;

      // and we want to release shape copy once we're done
      delete[] shape;
    }

  for (int e = 0; e < numIArgs; e++) iiArgs[e] = iArgs[e];

  for (int e = 0; e < numTArgs; e++) ttArgs[e] = tArgs[e];

  for (int e = 0; e < numBArgs; e++) biArgs[e] = bArgs[e];

  // hypothetically at this point we have everything filled
  auto hZ = op->execute(inputs, outputs, ttArgs, iiArgs, biArgs, std::vector<sd::DataType>(), isInplace);

  if (!isInplace)
    for (int e = 0; e < numOutputs; e++) {
      if (outputs[e]->ordering() != shape::order(reinterpret_cast<sd::LongType  *>(outputShapes[e])))
        outputs[e]->streamline(shape::order(reinterpret_cast<sd::LongType  *>(outputShapes[e])));
    }

  for (auto v : inputs) delete v;

  for (auto v : outputs) delete v;

  return hZ;
}


// Function to execute a custom operation
sd::Status execCustomOp(sd::Pointer *extraPointers, sd::LongType  hash, OpaqueNDArray *inputs, int numInputs,
                    OpaqueNDArray *outputs, int numOutputs, double *tArgs, int numTArgs,
                    sd::LongType  *iArgs, int numIArgs, bool *bArgs, int numBArgs, bool isInplace) {
  try {
    // Convert NDArray** inputs and outputs to std::vector<NDArray*>
    const std::vector<sd::NDArray*> inputVec(inputs, inputs + numInputs);
    const std::vector<sd::NDArray*> outputVec(outputs, outputs + numOutputs);
    const std::vector<double> tArgsVec(tArgs, tArgs + numTArgs);
    const std::vector<sd::LongType > iArgsVec(iArgs, iArgs + numIArgs);
    const std::vector<bool> bArgsVec(bArgs, bArgs + numBArgs);

    // Retrieve the operation based on the hash
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    if (op == nullptr) {
      throw std::invalid_argument("Operation not found for the given hash.");
    }

    // Execute the custom operation
    return op->execute(inputVec, outputVec, tArgsVec, iArgsVec, bArgsVec, {}, isInplace);
  }
  catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::KERNEL_FAILURE;
  }
}

void toggleOpTrace(bool opTrace) { sd::ops::OpRegistrator::getInstance().toggleTraceOps(opTrace);
}

void purgeOpTrace() { sd::ops::OpRegistrator::getInstance().purgeOpExecs();
}




void printOpTrace() {
  auto execTrace = *sd::ops::OpRegistrator::getInstance().execTrace();
  for(int i = 0; i < execTrace.size(); i++) {
    auto curr = execTrace[i];
    if(curr->opName != nullptr) {
      sd_printf("Op name: %s\n", curr->opName->c_str());
    }
    sd_printf(" Input buffers:\n",0);
    if(curr->inputShapeBuffers == nullptr || curr->inputShapeBuffers->size() == 0) {
      sd_printf("No input buffers\n",0);
      continue;
    } else {
      auto currInputShapeBuffers = *(curr->inputShapeBuffers);
      for(int j = 0; j < currInputShapeBuffers.size(); j++) {
        auto buff = currInputShapeBuffers[j];
        shape::printShapeInfo(buff);
        sd_printf("\n",0);
      }
    }

    if(curr->outputShapeBuffers == nullptr || curr->outputShapeBuffers->size() == 0) {
      sd_printf("No output buffers\n",0);
      continue;
    } else {
      auto currOutputShapeBuffers = *(curr->outputShapeBuffers);
      for(int j = 0; j < curr->outputShapeBuffers->size(); j++) {
        shape::printShapeInfo(currOutputShapeBuffers[j]);
        sd_printf("\n",0);
      }

    }


  }

}


std::vector<ExecTrace*> * listOpTraces() {
  return sd::ops::OpRegistrator::getInstance().execTrace();
}

void copyBuffer(OpaqueDataBuffer *target, long n,  OpaqueDataBuffer *from, long fromOffset, long targetOffset) {
  OpaqueDataBuffer *copyFrom = dbCreateView(from, n);
  OpaqueDataBuffer *targetView = dbCreateView(target, n);
  sd::DataBuffer *targetBuf = copyFrom->dataBuffer();
  sd::DataBuffer *srcBuf = targetView->dataBuffer();
  sd::DataBuffer::memcpy(targetBuf, srcBuf, 0, 0);
}



int contextNumInputs(void *contextPointer) {
  Context *context = (Context *) contextPointer;
  return context->width();
}

int contextNumOutputs(void *contextPointer) {
  Context *context = (Context *) contextPointer;
  return context->outputWidth();
}



int numInputs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->inputShapeBuffers->size();
}

int numOutputs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->outputShapeBuffers->size();
}

std::vector<bool> * bArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return &trace->bArgs;
}

std::vector<std::string> * sArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return (&trace->sArguments);
}
std::vector<double> * tArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return (&trace->tArgs);

}

std::vector<int> * dArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  std::vector<int> *dArgs = new std::vector<int>();
  for (int e = 0; e < trace->dArgs.size(); e++) {
    dArgs->push_back(trace->dArgs[e]);
  }
  return dArgs;
}

std::vector<sd::LongType> * iArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return &(trace->iArgs);
}

std::vector<const sd::LongType *> *inputShapeBuffers(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->inputShapeBuffers;
}

std::vector<const sd::LongType *> *outputShapeBuffers(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->outputShapeBuffers;
}

char *opName(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return const_cast<char *>(trace->opName->c_str());
}

sd::Status registerGraph(sd::Pointer *extraPointers, sd::LongType  graphId, sd::Pointer flatBufferPointer) {
  try {
    auto graph = sd::graph::GraphExecutioner::importFromFlatPointer(flatBufferPointer);

    GraphHolder::getInstance().registerGraph(graphId, graph);

    return sd::Status::OK;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::BAD_INPUT;
  }
}

static VariablesSet *executeStoredGraphT(sd::Pointer *extraPointers, sd::LongType  graphId, sd::Pointer *inputBuffers,
                                         sd::Pointer *inputShapes, int *inputIndices, int numInputs) {
  auto graph = sd::graph::GraphHolder::getInstance().cloneGraph(graphId);
  auto varSpace = graph->getVariableSpace();

  std::vector<sd::NDArray *> handles;

  for (int e = 0; e < numInputs; e++) {
    auto idx = inputIndices[e];

    // we'll delete this array later, together with cloned VariableSpace
    auto array = new sd::NDArray(inputBuffers[e], reinterpret_cast<sd::LongType  *>(inputShapes[e]), nullptr, 0, 0);
    handles.emplace_back(array);

    if (varSpace->hasVariable(idx)) {
      auto var = varSpace->getVariable(idx);
      if (var->hasNDArray()) delete var->getNDArray();

      var->setNDArray(array);
    } else
      varSpace->putVariable(idx, array);
  }

  auto hZ = sd::graph::GraphExecutioner::execute(graph, varSpace);
  auto varSet = new sd::graph::VariablesSet(hZ);

  if (hZ == sd::Status::OK) {
    // pull back results, and provide them
    auto outputs = graph->fetchOutputs();
    for (int e = 0; e < outputs->size(); e++) {
      // we're only getting variable ID/Index from original grap. values will be taken from cloned workspace
      std::pair<int, int> varId(outputs->at(e)->id(), outputs->at(e)->index());

      auto var = varSpace->getVariable(varId);

      varSet->push_back(var->clone());
    }

    delete outputs;
  }

  delete graph;

  return varSet;
}


VariablesSet *executeStoredGraph(sd::Pointer *extraPointers, sd::LongType  graphId, sd::Pointer *inputBuffers, sd::Pointer *inputShapes,
                                 int *inputIndices, int numInputs) {
  try {
    return executeStoredGraphT(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::LongType  getVariablesSetSize(VariablesSet *set) { return set->size(); }

sd::Status getVariablesSetStatus(VariablesSet *set) { return set->status(); }

Variable *getVariable(VariablesSet *set, sd::LongType  i) { return set->at(i); }

int getVariableId(Variable *variable) { return variable->id(); }

int getVariableIndex(Variable *variable) { return variable->index(); }

const char *getVariableName(Variable *variable) { return variable->getName()->c_str(); }

sd::LongType  const *getVariableShape(Variable *variable) { return variable->getNDArray()->shapeInfo(); }

void *getVariableBuffer(Variable *variable) { return variable->getNDArray()->buffer(); }

sd::Status unregisterGraph(sd::Pointer *extraPointers, sd::LongType  graphId) {
  try {
    GraphHolder::getInstance().dropGraphAny(graphId);

    return sd::Status::OK;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::BAD_INPUT;
  }
}

void deletePointerArray(sd::Pointer pointer) {
  sd::Pointer *ptr = reinterpret_cast<sd::Pointer *>(pointer);
  delete[] ptr;
}

void deleteCharArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<char *>(pointer);
  delete[] ptr;
}

void deleteIntArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<int *>(pointer);
  delete[] ptr;
}

void deleteLongArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<sd::LongType  *>(pointer);
  delete[] ptr;
}

void deleteVariablesSet(VariablesSet *pointer) {
  delete pointer;
}

void deleteShapeList(sd::Pointer shapeList) {
  sd::ShapeList *list = reinterpret_cast<sd::ShapeList *>(shapeList);
  delete list;
}

const char *getAllOperations() { return sd::OpTracker::getInstance().exportOperations(); }

sd::Pointer getGraphState(sd::LongType  id) { return (sd::Pointer) new GraphState(id); }

void deleteGraphState(sd::Pointer state) {
  auto stateP = reinterpret_cast<GraphState *>(state);
  delete stateP;
}

sd::Status execCustomOpWithScope_(sd::Pointer *extraPointers, sd::graph::GraphState *state, sd::LongType  opHash,
                              sd::LongType  *scopes, int numScopes, sd::Pointer *inputBuffers,
                              sd::Pointer *inputShapes, int numInputs, sd::Pointer *outputBuffers,
                              sd::Pointer *outputShapes, int numOutputs) {
  /**
   * That's basically exec, with VariableSpace provided in GraphState:
   * depending on operation (i.e. while of if), different logic executors could be used
   */

  auto graph = state->graph();
  auto varSpace = state->variableSpace();

  // Node is dynamically created, and has nothing beyond it: only inputs and outputs
  // this node has id of 0, and inputs are
  Node node(OpType_LOGIC, opHash, 0);

  // mapping inputs
  for (int e = 0; e < numInputs; e++) {
    auto buffer = inputBuffers[e];
    auto shapeInfo = reinterpret_cast<sd::LongType  *>(inputShapes[e]);

    auto array = new sd::NDArray(buffer, shapeInfo, varSpace->launchContext(), 0, 0);

    // now we just put array to VarSpace
    varSpace->putVariable(0, e, array);
    node.pickInput(0, e);
  }

  // mapping scopes
  for (int e = 0; e < numScopes; e++) {
    // we should check scope existence in GraphState/Graph
    int scopeId = (int)scopes[e];
    if (!state->hasScope(scopeId)) {
      return sd::Logger::logKernelFailureMsg();
    }
    node.pickInput(scopeId, 0);
  }

  auto hZ = LogicExecutor::processNode(graph, &node);
  if (hZ != sd::Status::OK) return hZ;

  // mapping outputs

  for (int e = 0; e < numOutputs; e++) {
    auto buffer = outputBuffers[e];
    auto shapeInfo = reinterpret_cast<sd::LongType  *>(outputShapes[e]);

    sd::NDArray array(buffer, shapeInfo, varSpace->launchContext(), 0, 0);

    // now we just put array to VarSpace to the same ID
    // varSpace->putVariable(0, e, array);

    auto t = varSpace->getVariable(0, e)->getNDArray();
    array.assign(*t);
  }

  // removing input variables
  for (int e = 0; e < numInputs; e++) {
    varSpace->dropVariable(0, e);
  }

  // after some bla-bla-bla we should have Graph and Node for current op
  return sd::Status::OK;
}

sd::Status execCustomOpWithScope(sd::Pointer *extraPointers, sd::Pointer state, sd::LongType  opHash, sd::LongType  *scopes, int numScopes,
                             sd::Pointer *inputBuffers, sd::Pointer *inputShapes, int numInputs, sd::Pointer *outputBuffers,
                             sd::Pointer *outputShapes, int numOutputs) {
  try {
    return execCustomOpWithScope(extraPointers, reinterpret_cast<GraphState *>(state), opHash, scopes,
                                 numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes,
                                 numOutputs);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::BAD_INPUT;
  }
}

void deleteResultWrapper(sd::Pointer ptr) {
  auto p = reinterpret_cast<ResultWrapper *>(ptr);
  delete p;
}

int estimateThreshold(sd::Pointer *extraPointers, sd::Pointer dX, sd::LongType  const *dXShapeInfo, int N,
                      float threshold) {
  THROW_EXCEPTION("estimateThreshold: Not implemented yet");
}



void deleteTadPack(sd::TadPack *ptr) {
  delete ptr;
}

OpaqueConstantDataBuffer constantBufferLong(sd::DataType dtype, sd::LongType  *data, int length) {
  return sd::ConstantHelper::getInstance().constantBuffer(sd::ConstantDescriptor(data, length), dtype);
}

OpaqueConstantDataBuffer constantBufferDouble(sd::DataType dtype, double *data, int length) {
  return sd::ConstantHelper::getInstance().constantBuffer(sd::ConstantDescriptor(data, length), dtype);
}

OpaqueConstantDataBuffer constantBuffer(sd::DataType dtype, sd::ConstantDescriptor *descriptor) {
  return sd::ConstantHelper::getInstance().constantBuffer(*descriptor, dtype);
}

sd::Pointer getConstantDataBufferPrimary(OpaqueConstantDataBuffer dbf) { return dbf->primary(); }
sd::Pointer getConstantDataBufferSpecial(OpaqueConstantDataBuffer dbf) { return dbf->special(); }
sd::LongType getConstantDataBufferLength(OpaqueConstantDataBuffer dbf) { return dbf->length(); }
sd::LongType getConstantDataBufferSizeOf(OpaqueConstantDataBuffer dbf) { return dbf->sizeOf(); }

sd::Pointer getConstantShapeBufferPrimary(OpaqueConstantShapeBuffer dbf) { return const_cast<sd::LongType *>(dbf->primary()); }

sd::Pointer getConstantShapeBufferSpecial(OpaqueConstantShapeBuffer dbf) { return const_cast<sd::LongType *>(dbf->special()); }

Context *createGraphContext(int nodeId) { return new Context(nodeId); }

OpaqueRandomGenerator getGraphContextRandomGenerator(Context *ptr) { return &ptr->randomGenerator(); }

void markGraphContextInplace(Context *ptr, bool reallyInplace) { ptr->markInplace(reallyInplace); }


//note here for javacpp mapping we have to use this odd type alias as a pointer
//to make the typedef work properly.
void setGraphContextInputArraysArr(OpaqueContext* ptr, int numArrays,OpaqueNDArrayArr *arr) {
  if(arr == nullptr)
    THROW_EXCEPTION("setGraphContextInputArraysArr: Input arrays were null!");
  for (int i = 0; i < numArrays; i++) {
    if(arr[i] == nullptr) {
      std::string errorMessage;
      errorMessage += "setGraphContextInputArraysArr: Input array at index ";
      errorMessage += std::to_string(i);
      errorMessage += " was null!";
      THROW_EXCEPTION(errorMessage.c_str());
    }

    OpaqueNDArray &ref = *arr[i];
    ptr->setInputArray(i, ref, false);
  }
}



void setGraphContextTArguments(Context *ptr, double *arguments, int numberOfArguments) {
  ptr->setTArguments(arguments, numberOfArguments);
}

void setGraphContextIArguments(Context *ptr, sd::LongType *arguments, int numberOfArguments) {
  ptr->setIArguments(arguments, numberOfArguments);
}

void setGraphContextBArguments(Context *ptr, bool *arguments, int numberOfArguments) {
  ptr->setBArguments(arguments, numberOfArguments);
}

void setGraphContextDArguments(OpaqueContext *ptr, int *arguments, int numberOfArguments) {
  std::vector<sd::DataType> dtypes(numberOfArguments);
  for (int e = 0; e < numberOfArguments; e++) dtypes[e] = (sd::DataType)arguments[e];

  ptr->setDArguments(dtypes);
}

void deleteGraphContext(Context *ptr) {}

OpaqueRandomGenerator createRandomGenerator(sd::LongType rootSeed, sd::LongType nodeSeed) {
  try {
    return new RandomGenerator(rootSeed, nodeSeed);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::LongType getRandomGeneratorRootState(OpaqueRandomGenerator ptr) { return ptr->rootState(); }

sd::LongType getRandomGeneratorNodeState(OpaqueRandomGenerator ptr) { return ptr->nodeState(); }

void setRandomGeneratorStates(OpaqueRandomGenerator ptr, sd::LongType rootSeed, sd::LongType nodeSeed) {
  ptr->setStates(rootSeed, nodeSeed);
}

float getRandomGeneratorRelativeFloat(OpaqueRandomGenerator ptr, sd::LongType index) {
  return ptr->relativeT<float>(index);
}

double getRandomGeneratorRelativeDouble(OpaqueRandomGenerator ptr, sd::LongType index) {
  return ptr->relativeT<double>(index);
}

int getRandomGeneratorRelativeInt(OpaqueRandomGenerator ptr, sd::LongType index) { return ptr->relativeInt(index); }

sd::LongType getRandomGeneratorRelativeLong(OpaqueRandomGenerator ptr, sd::LongType index) {
  return ptr->relativeLong(index);
}

int getRandomGeneratorNextInt(OpaqueRandomGenerator ptr) {
  // to nullify  _nodeState._long ^= (steps ^ 0xdeadbeef);
  // we will use step = 0xdeadbeef
  auto result = ptr->relativeInt(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

sd::LongType getRandomGeneratorNextLong(OpaqueRandomGenerator ptr) {
  auto result = ptr->relativeLong(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

float getRandomGeneratorNextFloat(OpaqueRandomGenerator ptr) {
  auto result = ptr->relativeT<float>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

double getRandomGeneratorNextDouble(OpaqueRandomGenerator ptr) {
  auto result = ptr->relativeT<double>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

void deleteRandomGenerator(OpaqueRandomGenerator ptr) { delete ptr; }



/**
 * Get the shape buffer from a
 * numpy array.
 * **Warning** this allocates memory
 * @param npyArray
 * @return
 */
 sd::Pointer shapeBufferForNumpyHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  auto shape = new sd::LongType[arr.shape.size()];
  for (unsigned int i = 0; i < arr.shape.size(); i++) {
    shape[i] = arr.shape[i];
  }

  auto shapeBuffer = shape::shapeBufferOfNpy(arr.shape.size(), shape, arr.fortranOrder);
  delete[] shape;
  return reinterpret_cast<sd::Pointer>(shapeBuffer);
}

/**
 *
 * @param npyArray
 * @return
 */
 sd::Pointer dataPointForNumpyHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  unsigned char* dataToPrint = reinterpret_cast<unsigned char*>(arr.data);
  return dataToPrint;
}

/**
 *
 * @param npyArray
 * @return
 */
 sd::Pointer dataPointForNumpyStruct(sd::Pointer npyArrayStruct) {
  cnpy::NpyArray* arrPointer = reinterpret_cast<cnpy::NpyArray*>(npyArrayStruct);
  unsigned char* dataToPrint = reinterpret_cast<unsigned char*>(arrPointer->data);
  return reinterpret_cast<sd::Pointer>(dataToPrint);
}

/**
 *
 * @param npyArray
 * @param fromFile
 * @return
 */
sd::Pointer dataPointForNumpy(sd::Pointer npyArray) {
  char* npyArrayBuffer = reinterpret_cast<char*>(npyArray);
  cnpy::NpyArray arr = cnpy::loadNpyFromPointer(npyArrayBuffer);
  return dataPointForNumpyStruct(reinterpret_cast<sd::Pointer>(&arr));
}

/**
 * Load a numpy array from a file
 * and return it as an sd::Pointer
 * @param path
 * @return
 */
 sd::Pointer numpyFromFile(std::string path) {
  char* numpyBuffer = cnpy::loadFile(path.data());
  return reinterpret_cast<sd::Pointer>(numpyBuffer);
}

////// NPZ //////

 void* mapFromNpzFile(std::string path) {
  cnpy::npz_t* mapPtr = new cnpy::npz_t();
  cnpy::npz_t map = cnpy::npzLoad(path);
  mapPtr->insert(map.begin(), map.end());
  return reinterpret_cast<void*>(mapPtr);
}

 int getNumNpyArraysInMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  int n = arrays->size();
  return n;
}

 const char* getNpyArrayNameFromMap(void* map, int index, char* nameBuffer) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      size_t len_of_str = strlen(it->first.c_str());
      memcpy(nameBuffer, it->first.c_str(), len_of_str);
    }
  }
  THROW_EXCEPTION("No array at index.");
}

 void* getNpyArrayFromMap(void* map, int index) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  cnpy::NpyArray* arr = new cnpy::NpyArray();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      *arr = it->second;
      return arr;
    }
  }
  THROW_EXCEPTION("No array at index.");
}


 void* getNpyArrayData(void* npArray) {
  cnpy::NpyArray* npyArray2 = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return reinterpret_cast<void*>(npyArray2->data);
}

 int getNpyArrayRank(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int rank = arr->shape.size();
  return rank;
}

 sd::LongType* getNpyArrayShape(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int ndim = arr->shape.size();
  sd::LongType* shape = new sd::LongType[ndim];
  for (int i = 0; i < ndim; i++) {
    shape[i] = arr->shape.at(i);
  }
  return shape;
}

 char getNpyArrayOrder(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return (arr->fortranOrder) ? 'f' : 'c';
}

 int getNpyArrayElemSize(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return arr->wordSize;
}

 void deleteNPArrayStruct(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  delete arr;
}

 void deleteNPArrayMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  delete arrays;
}
//////

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
 int elementSizeForNpyArray(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  // arrPointer->destruct();
  return size;
}

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
 int elementSizeForNpyArrayHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  return size;
}

 void releaseNumpy(sd::Pointer npyArray) { free(reinterpret_cast<void*>(npyArray)); }

#if defined(SD_GCC_FUNCTRACE)
// this is mainly a c based function.
extern "C" {

//note this is a c++ 17 feature
#ifndef INSTRUMENT_FILE_DEF
#pragma once
#define INSTRUMENT_FILE_DEF 1
FILE* instrumentFile = nullptr;
#endif


//we need to tell -finstrument-functions not to include the logger otherwise it will recursively
// stack overflow and segfault.
__attribute__((no_instrument_function)) SD_LIB_EXPORT  void writeLog(bool enter,void *this_fn,void *call_site) {
  if(instrumentFile == nullptr) {
    return;
  }
  Dl_info info;
  if (dladdr(this_fn, &info)) {
    int status;
    const char *funcName;
    char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, 0, &status);
    if (status == 0) {
      funcName = demangled  != nullptr ? demangled : "null_demangled";
    } else {
      funcName = info.dli_sname ? info.dli_sname : "null_dli_sname";
    }

    printf(" %s %s (%s)\n",enter ? "enter" : "exit", funcName, info.dli_fname);
    fprintf( instrumentFile," %s %s (%s)\n",enter ? "enter" : "exit", funcName, info.dli_fname);
    if (demangled != nullptr) {
      delete demangled;
      demangled = nullptr;
    }
  } else {
    printf("%s %s\n", enter ? "enter" : "exit","unknown");
    fprintf(instrumentFile, "%s %s\n", enter ? "enter" : "exit","unknown");
    fflush(instrumentFile);
  }
}
//we need to tell -finstrument-functions not to include the logger otherwise it will recursively
// stack overflow and segfault.
__attribute__((no_instrument_function)) SD_LIB_EXPORT void __cyg_profile_func_enter(void *this_fn,
                                                                                    void *call_site) {
  writeLog(true,this_fn, call_site);
}


//we need to tell -finstrument-functions not to include the logger otherwise it will recursively
// stack overflow and segfault.
__attribute__((no_instrument_function)) SD_LIB_EXPORT void __cyg_profile_func_exit  (void *this_fn,
                                                                                   void *call_site) {
  writeLog(false,this_fn, call_site);

}



}

#endif

void ctxAllowHelpers(OpaqueContext *ptr, bool reallyAllow) { ptr->allowHelpers(reallyAllow); }

void ctxSetExecutionMode(OpaqueContext *ptr, int execMode) {
  if (execMode < 0 || execMode > 2) execMode = 0;

  ptr->setExecutionMode((samediff::ExecutionMode)execMode);
}

sd::LongType getCachedMemory(int deviceId) { return sd::ConstantHelper::getInstance().getCachedAmount(deviceId); }


void ctxShapeFunctionOverride(OpaqueContext *ptr, bool reallyOverride) {
  ptr->setShapeFunctionOverride(reallyOverride);
}

void ctxPurge(OpaqueContext *ptr) { ptr->clearFastPath(); }

int lastErrorCode() { return sd::LaunchContext::defaultContext()->errorReference()->errorCode(); }

const char *lastErrorMessage() { return sd::LaunchContext::defaultContext()->errorReference()->errorMessage(); }


sd::LaunchContext *defaultLaunchContext() { return sd::LaunchContext::defaultContext(); }

sd::Pointer lcScalarPointer(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcReductionPointer(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcAllocationPointer(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcExecutionStream(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcCopyStream(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcBlasHandle(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcSolverHandle(OpaqueLaunchContext *lc) { return nullptr; }



void dbPrintAllocationTrace(OpaqueDataBuffer *db) {
  db->printDbAllocationTrace();
}



void setIntermediateResult(OpaqueContext *contextPointer,
                           int index,
                           OpaqueDataBuffer *buffer,
                           OpaqueDataBuffer *shapeInfo,
                           sd::LongType dataOffset) {
  if(shapeInfo == nullptr) {
    THROW_EXCEPTION("Set Intermediate Result: shapeInfo is null");
  }
  auto casted = reinterpret_cast<sd::LongType *>(shapeInfo->primary());
  auto desc = new sd::ShapeDescriptor(casted, false);
  auto arr = new sd::NDArray(buffer->dataBuffer(),
                         desc,
                             sd::LaunchContext::defaultContext(),
                         dataOffset);
  contextPointer->setIntermediateResult(index, arr);
}


std::vector<const sd::LongType *> intermediateResultsShapeInfo(OpaqueContext *contextPointer) {
  std::vector<const sd::LongType *> intermediates;
  for (auto v: contextPointer->intermediateResults()) {
    const sd::LongType *buff = v->shapeInfo();
    intermediates.push_back(buff);
  }

  return intermediates;
}

std::vector<OpaqueDataBuffer *> intermediateResults(OpaqueContext *contextPointer) {
  std::vector<OpaqueDataBuffer *> intermediates;
  for (auto v: contextPointer->intermediateResults()) {
    OpaqueDataBuffer *buff = new OpaqueDataBuffer (v->dataBuffer());
    intermediates.push_back(buff);
  }

  return intermediates;
}

int numIntermediateResults(OpaqueContext *contextPointer) {
  return contextPointer->numIntermediates();
}

void pushIntermediateResult(OpaqueContext *contextPointer,
                            OpaqueDataBuffer *buffer,
                            OpaqueDataBuffer *shapeInfo,
                            sd::LongType offset) {
  auto shapeInfoCast = reinterpret_cast<sd::LongType *>(shapeInfo->primary());
  auto desc = new sd::ShapeDescriptor(shapeInfoCast, false);
  auto arr = new sd::NDArray(buffer->dataBuffer(), desc, sd::LaunchContext::defaultContext(), offset);
  contextPointer->pushIntermediateResult(arr);
}

OpaqueDataBuffer  * intermediateResultDataAt(int index, OpaqueContext *contextPointer) {
  auto arr = contextPointer->intermediateResult(index);
  return new OpaqueDataBuffer(arr->dataBuffer());
}

const sd::LongType * intermediateResultShapeInfoAt(int index, OpaqueContext *contextPointer) {
  auto context = reinterpret_cast<sd::graph::Context *>(contextPointer);
  auto arr = context->intermediateResult(index);
  return arr->shapeInfo();
}


sd::TadPack *tadOnlyShapeInfo(OpaqueDataBuffer *hXShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength) {
  try {
    auto buffPrim = reinterpret_cast<sd::LongType *>(hXShapeInfo->primary());
    auto rankVal = buffPrim[0];
    if(rankVal == 0) {
      //detect when the shape buffer values are unset.
      auto len = shape::shapeInfoLength(rankVal);
      //min number of values in a shape info buffer
      bool allZero = true;
      for(int i = 0; i < len; i++) {
        if(buffPrim[i] != 0) {
          allZero = false;
          break;
        }
      }

      if(allZero) {
        THROW_EXCEPTION("Found shape buffer with all zero values. Values likely unset.");
      }
    }

    auto pack = sd::ConstantTadHelper::getInstance().tadForDimensions(reinterpret_cast<sd::LongType *>(hXShapeInfo->primary()), dimension, dimensionLength);
    return pack;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }


}


OpaqueConstantShapeBuffer shapeBuffer(int rank, sd::LongType *shape, sd::LongType *strides, sd::DataType dtype,
                                      char order, sd::LongType ews, bool empty) {
  return shapeBufferEx(rank, shape, strides, dtype, order, ews, empty ? ARRAY_EMPTY : 0);
}

sd::LongType dbBufferLength(OpaqueDataBuffer *dataBuffer) {
  return dataBuffer->dataBuffer()->getNumElements();
}


OpaqueDataBuffer *dbAllocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth) {
  return allocateDataBuffer(elements, dataType, allocateBoth);
}

OpaqueDataBuffer *allocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth) {
  try {
    auto dtype = sd::DataTypeUtils::fromInt(dataType);
    sd::LongType totalElementSize = elements == 0 ? sd::DataTypeUtils::sizeOf(dtype) : elements * sd::DataTypeUtils::sizeOf(dtype);
    return new sd::InteropDataBuffer(totalElementSize, dtype, allocateBoth);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

OpaqueDataBuffer *dbCreateExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary, sd::Pointer special) {
  auto buffer = dbAllocateDataBuffer(0, dataType, false);
  buffer->markOwner(false);

  if (primary != nullptr) buffer->setPrimary(primary, elements);

  if (special != nullptr) buffer->setSpecial(special, elements);

  return buffer;
}

sd::Pointer dbPrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) THROW_EXCEPTION("dbPrimaryBuffer: dataBuffer is null");
  return dataBuffer->primary();
}

sd::Pointer dbSpecialBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSpecialBuffer: dataBuffer is null");
  return dataBuffer->special();
}

void deleteDataBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbPrimaryBuffer: dataBuffer is null");
  delete dataBuffer;
}

void dbSetPrimaryBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer primaryBuffer, sd::LongType numBytes) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetPrimaryBuffer: dataBuffer is null");
  dataBuffer->setPrimary(primaryBuffer, numBytes);
}

void dbSetSpecialBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer specialBuffer, sd::LongType numBytes) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetSpecialBuffer: dataBuffer is null");
  dataBuffer->setSpecial(specialBuffer, numBytes);
}

void dbAllocatePrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbAllocatePrimaryBuffer: dataBuffer is null");
  dataBuffer->dataBuffer()->allocatePrimary();
}

void dbAllocateSpecialBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbAllocateSpecialBuffer: dataBuffer is null");
  dataBuffer->dataBuffer()->allocateSpecial();
}

void dbExpandBuffer(OpaqueDataBuffer *dataBuffer, sd::LongType elements) {
  try {
    if(dataBuffer == nullptr)
      THROW_EXCEPTION("dbExpandBuffer: dataBuffer is null");
    dataBuffer->dataBuffer()->expand(elements * sd::DataTypeUtils::sizeOf(dataBuffer->dataBuffer()->getDataType()));
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

OpaqueDataBuffer *dbCreateView(OpaqueDataBuffer *dataBuffer, sd::LongType length) {
  return new OpaqueDataBuffer(dataBuffer, length);
}


int dbUseCount(OpaqueDataBuffer* dataBuffer) {
  if(dataBuffer) return dataBuffer->useCount();
  return 0;
}

void dbSyncToSpecial(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSyncToSpecial: dataBuffer is null");
  if(dataBuffer->dataBuffer() != nullptr  && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToSpecial();
}

void dbSyncToPrimary(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSyncToPrimary: dataBuffer is null");
  if(dataBuffer->dataBuffer() != nullptr  && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToPrimary(sd::LaunchContext::defaultContext(),false);

}

void dbTickHostRead(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickHostRead: dataBuffer is null");
  dataBuffer->dataBuffer()->readPrimary();
}

void dbTickHostWrite(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickHostWrite: dataBuffer is null");
  dataBuffer->dataBuffer()->writePrimary();
}

void dbTickDeviceRead(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickDeviceRead: dataBuffer is null");
  dataBuffer->dataBuffer()->readSpecial();
}

void dbTickDeviceWrite(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickDeviceWrite: dataBuffer is null");
  dataBuffer->dataBuffer()->writeSpecial();

}

void dbExpand(OpaqueDataBuffer *dataBuffer, sd::LongType elements) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbExpand: dataBuffer is null");
  dataBuffer->expand(elements);
}

void dbClose(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbClose: dataBuffer is null");

  auto ret = dataBuffer->getDataBuffer();
  if(ret != nullptr)
    dataBuffer->getDataBuffer()->close();
}

int dbDeviceId(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbDeviceId: dataBuffer is null");
  return dataBuffer->deviceId();
}

void dbSetDeviceId(OpaqueDataBuffer *dataBuffer, int deviceId) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetDeviceId: dataBuffer is null");
  dataBuffer->setDeviceId(deviceId);
}

int dbLocality(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbLocality: dataBuffer is null");
  auto p = dataBuffer->dataBuffer()->isPrimaryActual();
  auto d = dataBuffer->dataBuffer()->isSpecialActual();

  if (p && d)
    return 0;
  else if (p)
    return -1;
  else
    return 1;
}

