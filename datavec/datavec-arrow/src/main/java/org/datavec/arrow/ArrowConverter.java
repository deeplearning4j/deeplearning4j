package org.datavec.arrow;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.arrow.flatbuf.Tensor;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.dictionary.Dictionary;
import org.apache.arrow.vector.dictionary.DictionaryProvider;
import org.apache.arrow.vector.holders.VarBinaryHolder;
import org.apache.arrow.vector.ipc.ArrowFileReader;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.ipc.SeekableReadChannel;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;
import org.datavec.api.records.Buffer;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.*;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.conversion.TypeConversion;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.*;
import org.datavec.arrow.recordreader.ArrowWritableRecordBatch;
import org.datavec.arrow.recordreader.ArrowWritableRecordTimeSeriesBatch;
import org.nd4j.arrow.ArrowSerde;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

import static java.nio.channels.Channels.newChannel;

/**
 * Interop between datavec primitives and arrow.
 * This allows for datavec schemas and primitives
 * to be converted to the arrow format.
 *
 * @author Adam Gibson
 */
@Slf4j
public class ArrowConverter {




    /**
     * Create an ndarray from a matrix.
     * The included batch must be all the same number of rows in order
     * to work. The reason for this is {@link INDArray} must be all the same dimensions.
     * Note that the input columns must also be numerical. If they aren't numerical already,
     * consider using an {@link org.datavec.api.transform.TransformProcess} to transform the data
     * output from {@link org.datavec.arrow.recordreader.ArrowRecordReader} in to the proper format
     * for usage with this method for direct conversion.
     *
     * @param arrowWritableRecordBatch the incoming batch. This is typically output from
     *                                 an {@link org.datavec.arrow.recordreader.ArrowRecordReader}
     * @return an {@link INDArray} representative of the input data
     */
    public static INDArray toArray(ArrowWritableRecordTimeSeriesBatch arrowWritableRecordBatch) {
        return RecordConverter.toTensor(arrowWritableRecordBatch);
    }




    /**
     * Create an ndarray from a matrix.
     * The included batch must be all the same number of rows in order
     * to work. The reason for this is {@link INDArray} must be all the same dimensions.
     * Note that the input columns must also be numerical. If they aren't numerical already,
     * consider using an {@link org.datavec.api.transform.TransformProcess} to transform the data
     * output from {@link org.datavec.arrow.recordreader.ArrowRecordReader} in to the proper format
     * for usage with this method for direct conversion.
     *
     * @param arrowWritableRecordBatch the incoming batch. This is typically output from
     *                                 an {@link org.datavec.arrow.recordreader.ArrowRecordReader}
     * @return an {@link INDArray} representative of the input data
     */
    public static INDArray toArray(ArrowWritableRecordBatch arrowWritableRecordBatch) {
        List<FieldVector> columnVectors = arrowWritableRecordBatch.getList();
        Schema schema = arrowWritableRecordBatch.getSchema();
        for(int i = 0; i < schema.numColumns(); i++) {
            switch(schema.getType(i)) {
                case Integer:
                    break;
                case Float:
                    break;
                case Double:
                    break;
                case Long:
                    break;
                default:
                    throw new ND4JIllegalArgumentException("Illegal data type found for column " + schema.getName(i));
            }
        }

        int rows  = arrowWritableRecordBatch.getList().get(0).getValueCount();
        int cols = schema.numColumns();
        INDArray arr  = Nd4j.create(rows,cols);
        for(int i = 0; i < cols; i++) {
            INDArray put = ArrowConverter.convertArrowVector(columnVectors.get(i),schema.getType(i));
            switch(arr.data().dataType()) {
                case FLOAT:
                    arr.putColumn(i,Nd4j.create(put.data().asFloat()).reshape(rows,1));
                    break;
                case DOUBLE:
                    arr.putColumn(i,Nd4j.create(put.data().asDouble()).reshape(rows,1));
                    break;
            }

        }

        return arr;
    }

    /**
     * Convert a field vector to a column vector
     * @param fieldVector the field vector to convert
     * @param type the type of the column vector
     * @return the converted ndarray
     */
    public static INDArray convertArrowVector(FieldVector fieldVector,ColumnType type) {
        DataBuffer buffer = null;
        int cols = fieldVector.getValueCount();
        ByteBuffer direct = ByteBuffer.allocateDirect(fieldVector.getDataBuffer().capacity());
        direct.order(ByteOrder.nativeOrder());
        fieldVector.getDataBuffer().getBytes(0,direct);
        direct.rewind();
        switch(type) {
            case Integer:
                buffer = Nd4j.createBuffer(direct, DataBuffer.Type.INT,cols,0);
                break;
            case Float:
                buffer = Nd4j.createBuffer(direct, DataBuffer.Type.FLOAT,cols);
                break;
            case Double:
                buffer = Nd4j.createBuffer(direct, DataBuffer.Type.DOUBLE,cols);
                break;
            case Long:
                buffer =  Nd4j.createBuffer(direct, DataBuffer.Type.LONG,cols);
                break;
        }

        return Nd4j.create(buffer,new int[] {cols,1});
    }


    /**
     * Convert an {@link INDArray}
     * to a list of column vectors or a singleton
     * list when either a row vector or a column vector
     * @param from the input array
     * @param name the name of the vector
     * @param type the type of the vector
     * @param bufferAllocator the allocator to use
     * @return the list of field vectors
     */
    public static List<FieldVector> convertToArrowVector(INDArray from,List<String> name,ColumnType type,BufferAllocator bufferAllocator) {
        List<FieldVector> ret = new ArrayList<>();
        if(from.isVector()) {
            long cols = from.length();
            switch(type) {
                case Double:
                    double[] fromData = from.isView() ? from.dup().data().asDouble() : from.data().asDouble();
                    ret.add(vectorFor(bufferAllocator,name.get(0),fromData));
                    break;
                case Float:
                    float[] fromDataFloat = from.isView() ? from.dup().data().asFloat() : from.data().asFloat();
                    ret.add(vectorFor(bufferAllocator,name.get(0),fromDataFloat));
                    break;
                case Integer:
                    int[] fromDataInt = from.isView() ? from.dup().data().asInt() : from.data().asInt();
                    ret.add(vectorFor(bufferAllocator,name.get(0),fromDataInt));
                    break;
                default:
                    throw new IllegalArgumentException("Illegal type " + type);
            }

        }
        else {
            long cols = from.size(1);
            for(int i = 0; i < cols; i++) {
                INDArray column = from.getColumn(i);

                switch(type) {
                    case Double:
                        double[] fromData = column.isView() ? column.dup().data().asDouble() : from.data().asDouble();
                        ret.add(vectorFor(bufferAllocator,name.get(i),fromData));
                        break;
                    case Float:
                        float[] fromDataFloat = column.isView() ? column.dup().data().asFloat() : from.data().asFloat();
                        ret.add(vectorFor(bufferAllocator,name.get(i),fromDataFloat));
                        break;
                    case Integer:
                        int[] fromDataInt = column.isView() ? column.dup().data().asInt() : from.data().asInt();
                        ret.add(vectorFor(bufferAllocator,name.get(i),fromDataInt));
                        break;
                    default:
                        throw new IllegalArgumentException("Illegal type " + type);
                }
            }
        }


        return ret;
    }



    /**
     * Write the records to the given output stream
     * @param recordBatch the record batch to write
     * @param inputSchema the input schema
     * @param outputStream the output stream to write to
     */
    public static void writeRecordBatchTo(List<List<Writable>> recordBatch, Schema inputSchema,OutputStream outputStream) {
        BufferAllocator bufferAllocator = new RootAllocator(Long.MAX_VALUE);
        writeRecordBatchTo(bufferAllocator,recordBatch,inputSchema,outputStream);
    }

    /**
     * Write the records to the given output stream
     * @param recordBatch the record batch to write
     * @param inputSchema the input schema
     * @param outputStream the output stream to write to
     */
    public static void writeRecordBatchTo(BufferAllocator bufferAllocator ,List<List<Writable>> recordBatch, Schema inputSchema,OutputStream outputStream) {
        if(!(recordBatch instanceof ArrowWritableRecordBatch)) {
            val convertedSchema = toArrowSchema(inputSchema);
            val columns  = toArrowColumns(bufferAllocator,inputSchema,recordBatch);
            try {
                VectorSchemaRoot root = new VectorSchemaRoot(convertedSchema,columns,recordBatch.size());

                ArrowFileWriter writer = new ArrowFileWriter(root, providerForVectors(columns,convertedSchema.getFields()),
                        newChannel(outputStream));
                writer.start();
                writer.writeBatch();
                writer.end();


            } catch (IOException e) {
                throw new IllegalStateException(e);
            }

        }
        else {
            val convertedSchema = toArrowSchema(inputSchema);
            val pair = toArrowColumns(bufferAllocator,inputSchema,recordBatch);
            try {
                VectorSchemaRoot root = new VectorSchemaRoot(convertedSchema,pair,recordBatch.size());

                ArrowFileWriter writer = new ArrowFileWriter(root, providerForVectors(pair,convertedSchema.getFields()),
                        newChannel(outputStream));
                writer.start();
                writer.writeBatch();
                writer.end();


            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        }

    }


    /**
     * Convert the input field vectors (the input data) and
     * the given schema to a proper list of writables.
     * @param fieldVectors the field vectors to use
     * @param schema the schema to use
     * @param timeSeriesLength the length of the time series
     * @return the equivalent datavec batch given the input data
     */
    public static List<List<List<Writable>>> toArrowWritablesTimeSeries(List<FieldVector> fieldVectors,Schema schema,int timeSeriesLength) {
        ArrowWritableRecordTimeSeriesBatch arrowWritableRecordBatch = new ArrowWritableRecordTimeSeriesBatch(fieldVectors,schema,timeSeriesLength);
        return arrowWritableRecordBatch;
    }


    /**
     * Convert the input field vectors (the input data) and
     * the given schema to a proper list of writables.
     * @param fieldVectors the field vectors to use
     * @param schema the schema to use
     * @return the equivalent datavec batch given the input data
     */
    public static ArrowWritableRecordBatch toArrowWritables(List<FieldVector> fieldVectors,Schema schema) {
        ArrowWritableRecordBatch arrowWritableRecordBatch = new ArrowWritableRecordBatch(fieldVectors,schema);
        return arrowWritableRecordBatch;
    }

    /**
     * Return a singular record based on the converted
     * writables result.
     * @param fieldVectors the field vectors to use
     * @param schema the schema to use for input
     * @return
     */
    public static List<Writable> toArrowWritablesSingle(List<FieldVector> fieldVectors,Schema schema) {
        return toArrowWritables(fieldVectors,schema).get(0);
    }


    /**
     * Read a datavec schema and record set
     * from the given arrow file.
     * @param input the input to read
     * @return the associated datavec schema and record
     */
    public static Pair<Schema,ArrowWritableRecordBatch> readFromFile(FileInputStream input) throws IOException {
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        Schema retSchema = null;
        ArrowWritableRecordBatch ret = null;
        SeekableReadChannel channel = new SeekableReadChannel(input.getChannel());
        ArrowFileReader reader = new ArrowFileReader(channel, allocator);
        reader.loadNextBatch();
        retSchema = toDatavecSchema(reader.getVectorSchemaRoot().getSchema());
        //load the batch
        VectorUnloader unloader = new VectorUnloader(reader.getVectorSchemaRoot());
        VectorLoader vectorLoader = new VectorLoader(reader.getVectorSchemaRoot());
        ArrowRecordBatch recordBatch = unloader.getRecordBatch();

        vectorLoader.load(recordBatch);
        ret = asDataVecBatch(recordBatch,retSchema,reader.getVectorSchemaRoot());
        ret.setUnloader(unloader);

        return Pair.of(retSchema,ret);

    }

    /**
     * Read a datavec schema and record set
     * from the given arrow file.
     * @param input the input to read
     * @return the associated datavec schema and record
     */
    public static Pair<Schema,ArrowWritableRecordBatch> readFromFile(File input) throws IOException {
        return readFromFile(new FileInputStream(input));
    }

    /**
     * Read a datavec schema and record set
     * from the given bytes (usually expected to be an arrow format file)
     * @param input the input to read
     * @return the associated datavec schema and record
     */
    public static Pair<Schema,ArrowWritableRecordBatch> readFromBytes(byte[] input) throws IOException {
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        Schema retSchema = null;
        ArrowWritableRecordBatch ret = null;
        SeekableReadChannel channel = new SeekableReadChannel(new ByteArrayReadableSeekableByteChannel(input));
        ArrowFileReader reader = new ArrowFileReader(channel, allocator);
        reader.loadNextBatch();
        retSchema = toDatavecSchema(reader.getVectorSchemaRoot().getSchema());
        //load the batch
        VectorUnloader unloader = new VectorUnloader(reader.getVectorSchemaRoot());
        VectorLoader vectorLoader = new VectorLoader(reader.getVectorSchemaRoot());
        ArrowRecordBatch recordBatch = unloader.getRecordBatch();

        vectorLoader.load(recordBatch);
        ret = asDataVecBatch(recordBatch,retSchema,reader.getVectorSchemaRoot());
        ret.setUnloader(unloader);

        return Pair.of(retSchema,ret);

    }

    /**
     * Convert a data vec {@link Schema}
     * to an arrow {@link org.apache.arrow.vector.types.pojo.Schema}
     * @param schema the input schema
     * @return the schema for arrow
     */
    public static org.apache.arrow.vector.types.pojo.Schema toArrowSchema(Schema schema) {
        List<Field> fields = new ArrayList<>(schema.numColumns());
        for(int i = 0; i < schema.numColumns(); i++) {
            fields.add(getFieldForColumn(schema.getName(i),schema.getType(i)));
        }

        return new org.apache.arrow.vector.types.pojo.Schema(fields);
    }

    /**
     * Convert an {@link org.apache.arrow.vector.types.pojo.Schema}
     * to a datavec {@link Schema}
     * @param schema the input arrow schema
     * @return the equivalent datavec schema
     */
    public static Schema toDatavecSchema(org.apache.arrow.vector.types.pojo.Schema schema) {
        Schema.Builder schemaBuilder = new Schema.Builder();
        for (int i = 0; i < schema.getFields().size(); i++) {
            schemaBuilder.addColumn(metaDataFromField(schema.getFields().get(i)));
        }
        return schemaBuilder.build();
    }




    /**
     * Shortcut method for returning a field
     * given an arrow type and name
     * with no sub fields
     * @param name the name of the field
     * @param arrowType the arrow type of the field
     * @return the resulting field
     */
    public static Field field(String name,ArrowType arrowType) {
        return new Field(name,FieldType.nullable(arrowType), new ArrayList<Field>());
    }



    /**
     * Create a field given the input {@link ColumnType}
     * and name
     * @param name the name of the field
     * @param columnType the column type to add
     * @return
     */
    public static Field getFieldForColumn(String name,ColumnType columnType) {
        switch(columnType) {
            case Long: return field(name,new ArrowType.Int(64,false));
            case Integer: return field(name,new ArrowType.Int(32,false));
            case Double: return field(name,new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE));
            case Float: return field(name,new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE));
            case Boolean: return field(name, new ArrowType.Bool());
            case Categorical: return field(name,new ArrowType.Utf8());
            case Time: return field(name,new ArrowType.Date(DateUnit.MILLISECOND));
            case Bytes: return field(name,new ArrowType.Binary());
            case NDArray: return field(name,new ArrowType.Binary());
            case String: return field(name,new ArrowType.Utf8());

            default: throw new IllegalArgumentException("Column type invalid " + columnType);
        }
    }

    /**
     * Shortcut method for creating a double field
     * with 64 bit floating point
     * @param name the name of the field
     * @return the created field
     */
    public static Field doubleField(String name) {
        return getFieldForColumn(name, ColumnType.Double);
    }

    /**
     * Shortcut method for creating a double field
     * with 32 bit floating point
     * @param name the name of the field
     * @return the created field
     */
    public static Field floatField(String name) {
        return getFieldForColumn(name,ColumnType.Float);
    }

    /**
     * Shortcut method for creating a double field
     * with 32 bit integer field
     * @param name the name of the field
     * @return the created field
     */
    public static Field intField(String name) {
        return getFieldForColumn(name,ColumnType.Integer);
    }

    /**
     * Shortcut method for creating a long field
     * with 64 bit long field
     * @param name the name of the field
     * @return the created field
     */
    public static Field longField(String name) {
        return getFieldForColumn(name,ColumnType.Long);
    }

    /**
     *
     * @param name
     * @return
     */
    public static Field stringField(String name) {
        return getFieldForColumn(name,ColumnType.String);
    }

    /**
     * Shortcut
     * @param name
     * @return
     */
    public static Field booleanField(String name) {
        return getFieldForColumn(name,ColumnType.Boolean);
    }


    /**
     * Provide a value look up dictionary based on the
     * given set of input {@link FieldVector} s for
     * reading and writing to arrow streams
     * @param vectors the vectors to use as a lookup
     * @return the associated {@link DictionaryProvider} for the given
     * input {@link FieldVector} list
     */
    public static DictionaryProvider providerForVectors(List<FieldVector> vectors,List<Field> fields) {
        Dictionary[] dictionaries = new Dictionary[vectors.size()];
        for(int i = 0; i < vectors.size(); i++) {
            DictionaryEncoding dictionary = fields.get(i).getDictionary();
            if(dictionary == null) {
                dictionary = new DictionaryEncoding(i,true,null);
            }
            dictionaries[i] = new Dictionary(vectors.get(i), dictionary);
        }
        return  new DictionaryProvider.MapDictionaryProvider(dictionaries);
    }


    /**
     * Given a buffer allocator and datavec schema,
     * convert the passed in batch of records
     * to a set of arrow columns
     * @param bufferAllocator the buffer allocator to use
     * @param schema the schema to convert
     * @param dataVecRecord the data vec record batch to convert
     * @return the converted list of {@link FieldVector}
     */
    public static List<FieldVector> toArrowColumns(final BufferAllocator bufferAllocator, final Schema schema, List<List<Writable>> dataVecRecord) {
        int numRows = dataVecRecord.size();

        List<FieldVector> ret = createFieldVectors(bufferAllocator,schema,numRows);

        for(int j = 0; j < schema.numColumns(); j++) {
            FieldVector fieldVector = ret.get(j);
            int row = 0;
            for(List<Writable> record : dataVecRecord) {
                Writable writable = record.get(j);
                setValue(schema.getType(j),fieldVector,writable,row);
                row++;
            }

        }

        return ret;
    }


    /**
     * Convert a set of input strings to arrow columns
     * for a time series.
     * @param bufferAllocator the buffer allocator to use
     * @param schema the schema to use
     * @param dataVecRecord the collection of input strings to process
     * @return the created vectors
     */
    public static  List<FieldVector> toArrowColumnsTimeSeries(final BufferAllocator bufferAllocator,
                                                              final Schema schema,
                                                              List<List<List<Writable>>> dataVecRecord) {
        return toArrowColumnsTimeSeriesHelper(bufferAllocator,schema,dataVecRecord);
    }


    /**
     * Convert a set of input strings to arrow columns
     * for a time series.
     * @param bufferAllocator the buffer allocator to use
     * @param schema the schema to use
     * @param dataVecRecord the collection of input strings to process
     * @return the created vectors
     */
    public static <T>  List<FieldVector> toArrowColumnsTimeSeriesHelper(final BufferAllocator bufferAllocator,
                                                                        final Schema schema,
                                                                        List<List<List<T>>> dataVecRecord) {
        //time series length * number of columns
        int numRows = 0;
        for(List<List<T>> timeStep : dataVecRecord) {
            numRows += timeStep.get(0).size() * timeStep.size();
        }

        numRows /= schema.numColumns();


        List<FieldVector> ret = createFieldVectors(bufferAllocator,schema,numRows);
        Map<Integer,Integer> currIndex = new HashMap<>(ret.size());
        for(int i = 0; i < ret.size(); i++) {
            currIndex.put(i,0);
        }
        for(int i = 0; i < dataVecRecord.size(); i++) {
            List<List<T>> record = dataVecRecord.get(i);
            for(int j = 0; j < record.size(); j++) {
                List<T> curr = record.get(j);
                for(int k = 0; k < curr.size(); k++) {
                    Integer idx = currIndex.get(k);
                    FieldVector fieldVector = ret.get(k);
                    T writable = curr.get(k);
                    setValue(schema.getType(k), fieldVector, writable, idx);
                    currIndex.put(k,idx + 1);
                }
            }
        }

        return ret;
    }



    /**
     * Convert a set of input strings to arrow columns
     * @param bufferAllocator the buffer allocator to use
     * @param schema the schema to use
     * @param dataVecRecord the collection of input strings to process
     * @return the created vectors
     */
    public static  List<FieldVector> toArrowColumnsStringSingle(final BufferAllocator bufferAllocator, final Schema schema, List<String> dataVecRecord) {
        return toArrowColumnsString(bufferAllocator,schema, Arrays.asList(dataVecRecord));
    }



    /**
     * Convert a set of input strings to arrow columns
     * for a time series.
     * @param bufferAllocator the buffer allocator to use
     * @param schema the schema to use
     * @param dataVecRecord the collection of input strings to process
     * @return the created vectors
     */
    public static  List<FieldVector> toArrowColumnsStringTimeSeries(final BufferAllocator bufferAllocator,
                                                                    final Schema schema,
                                                                    List<List<List<String>>> dataVecRecord) {
        return toArrowColumnsTimeSeriesHelper(bufferAllocator,schema,dataVecRecord);

    }


    /**
     * Convert a set of input strings to arrow columns
     * @param bufferAllocator the buffer allocator to use
     * @param schema the schema to use
     * @param dataVecRecord the collection of input strings to process
     * @return the created vectors
     */
    public static  List<FieldVector> toArrowColumnsString(final BufferAllocator bufferAllocator, final Schema schema, List<List<String>> dataVecRecord) {
        int numRows = dataVecRecord.size();

        List<FieldVector> ret = createFieldVectors(bufferAllocator,schema,numRows);
        /**
         * Need to change iteration scheme
         */

        for(int j = 0; j < schema.numColumns(); j++) {
            FieldVector fieldVector = ret.get(j);
            for(int row = 0; row < numRows; row++) {
                String writable = dataVecRecord.get(row).get(j);
                setValue(schema.getType(j),fieldVector,writable,row);
            }

        }

        return ret;
    }


    private static List<FieldVector> createFieldVectors(BufferAllocator bufferAllocator,Schema schema, int numRows) {
        List<FieldVector> ret = new ArrayList<>(schema.numColumns());

        for(int i = 0; i < schema.numColumns(); i++) {
            switch (schema.getType(i)) {
                case Integer: ret.add(intVectorOf(bufferAllocator,schema.getName(i),numRows)); break;
                case Long: ret.add(longVectorOf(bufferAllocator,schema.getName(i),numRows)); break;
                case Double: ret.add(doubleVectorOf(bufferAllocator,schema.getName(i),numRows)); break;
                case Float: ret.add(floatVectorOf(bufferAllocator,schema.getName(i),numRows)); break;
                case Boolean: ret.add(booleanVectorOf(bufferAllocator,schema.getName(i),numRows)); break;
                case String: ret.add(stringVectorOf(bufferAllocator,schema.getName(i),numRows)); break;
                case Categorical: ret.add(stringVectorOf(bufferAllocator,schema.getName(i),numRows)); break;
                case Time: ret.add(timeVectorOf(bufferAllocator,schema.getName(i),numRows)); break;
                case NDArray: ret.add(ndarrayVectorOf(bufferAllocator,schema.getName(i),numRows)); break;
                default: throw new IllegalArgumentException("Illegal type found " + schema.getType(i));

            }
        }

        return ret;
    }

    /**
     * Set the value of the specified column vector
     * at the specified row based on the given value.
     * The value will be converted relative to the specified column type.
     * Note that the passed in value may only be a {@link Writable}
     * or a {@link String}
     * @param columnType the column type of the value
     * @param fieldVector the field vector to set
     * @param value the value to set ({@link Writable} or {@link String} types)
     * @param row the row of the item
     */
    public static void setValue(ColumnType columnType,FieldVector fieldVector,Object value,int row) {
        if(value instanceof NullWritable) {
            return;
        }
        try {
            switch (columnType) {
                case Integer:
                    if (fieldVector instanceof IntVector) {
                        IntVector intVector = (IntVector) fieldVector;
                        int set = TypeConversion.getInstance().convertInt(value);
                        intVector.set(row, set);
                    } else if (fieldVector instanceof UInt4Vector) {
                        UInt4Vector uInt4Vector = (UInt4Vector) fieldVector;
                        int set = TypeConversion.getInstance().convertInt(value);
                        uInt4Vector.set(row, set);
                    } else {
                        throw new UnsupportedOperationException("Illegal type " + fieldVector.getClass() + " for int type");
                    }
                    break;
                case Float:
                    Float4Vector float4Vector = (Float4Vector) fieldVector;
                    float set2 = TypeConversion.getInstance().convertFloat(value);
                    float4Vector.set(row, set2);
                    break;
                case Double:
                    double set3 = TypeConversion.getInstance().convertDouble(value);
                    Float8Vector float8Vector = (Float8Vector) fieldVector;
                    float8Vector.set(row, set3);
                    break;
                case Long:
                    if (fieldVector instanceof BigIntVector) {
                        BigIntVector largeIntVector = (BigIntVector) fieldVector;
                        largeIntVector.set(row, TypeConversion.getInstance().convertLong(value));

                    } else if (fieldVector instanceof UInt8Vector) {
                        UInt8Vector uInt8Vector = (UInt8Vector) fieldVector;
                        uInt8Vector.set(row, TypeConversion.getInstance().convertLong(value));
                    } else {
                        throw new UnsupportedOperationException("Illegal type " + fieldVector.getClass() + " for long type");
                    }
                    break;
                case Categorical:
                case String:
                    String stringSet = TypeConversion.getInstance().convertString(value);
                    VarCharVector textVector = (VarCharVector) fieldVector;
                    textVector.set(row, stringSet.getBytes());
                    break;
                case Time:
                    //all timestamps are long based, just directly convert it to the super type
                    long timeSet = TypeConversion.getInstance().convertLong(value);
                    setLongInTime(fieldVector, row, timeSet);
                    break;
                case NDArray:
                    NDArrayWritable arr = (NDArrayWritable) value;
                    VarBinaryVector nd4jArrayVector = (VarBinaryVector) fieldVector;
                    //slice the databuffer to use only the needed portion of the buffer
                    //for proper offsets
                    ByteBuffer byteBuffer = ArrowSerde.toTensor(arr.get()).getByteBuffer().slice();
                    nd4jArrayVector.set(row,byteBuffer,0,byteBuffer.capacity());
                    break;

            }
        }catch(Exception e) {
            log.warn("Unable to set value at row " + row);
        }
    }


    private static void setLongInTime(FieldVector fieldVector,int index,long value) {
        if(fieldVector instanceof TimeStampMilliVector) {
            TimeStampMilliVector timeStampMilliVector = (TimeStampMilliVector) fieldVector;
            timeStampMilliVector.set(index,value);
        }
        else if(fieldVector instanceof TimeMilliVector) {
            TimeMilliVector timeMilliVector = (TimeMilliVector) fieldVector;
            timeMilliVector.set(index,(int) value);
        }
        else if(fieldVector instanceof TimeStampMicroVector) {
            TimeStampMicroVector timeStampMicroVector = (TimeStampMicroVector) fieldVector;
            timeStampMicroVector.set(index,value);
        }
        else if(fieldVector instanceof TimeSecVector) {
            TimeSecVector timeSecVector = (TimeSecVector) fieldVector;
            timeSecVector.set(index,(int) value);
        }
        else if(fieldVector instanceof TimeStampMilliVector) {
            TimeStampMilliVector timeStampMilliVector = (TimeStampMilliVector) fieldVector;
            timeStampMilliVector.set(index,value);
        }
        else if(fieldVector instanceof TimeStampMilliTZVector) {
            TimeStampMilliTZVector timeStampMilliTZVector = (TimeStampMilliTZVector) fieldVector;
            timeStampMilliTZVector.set(index, value);
        }
        else if(fieldVector instanceof TimeStampNanoTZVector) {
            TimeStampNanoTZVector timeStampNanoTZVector = (TimeStampNanoTZVector) fieldVector;
            timeStampNanoTZVector.set(index,value);
        }
        else if(fieldVector instanceof TimeStampMicroTZVector) {
            TimeStampMicroTZVector timeStampMicroTZVector = (TimeStampMicroTZVector) fieldVector;
            timeStampMicroTZVector.set(index,value);
        }
        else {
            throw new UnsupportedOperationException();
        }
    }


    /**
     *
     * @param allocator
     * @param name
     * @param data
     * @return
     */
    public static TimeStampMilliVector vectorFor(BufferAllocator allocator,String name,Date[] data) {
        TimeStampMilliVector float4Vector = new TimeStampMilliVector(name,allocator);
        float4Vector.allocateNew(data.length);
        for(int i = 0; i < data.length; i++) {
            float4Vector.setSafe(i,data[i].getTime());
        }

        float4Vector.setValueCount(data.length);

        return float4Vector;
    }


    /**
     *
     * @param allocator
     * @param name
     * @param length the length of the vector
     * @return
     */
    public static TimeStampMilliVector timeVectorOf(BufferAllocator allocator,String name,int length) {
        TimeStampMilliVector float4Vector = new TimeStampMilliVector(name,allocator);
        float4Vector.allocateNew(length);
        float4Vector.setValueCount(length);
        return float4Vector;
    }


    /**
     * Returns a vector representing a tensor view
     * of each ndarray.
     * Each ndarray will be a "row" represented as a tensor object
     * with in the return {@link VarBinaryVector}
     * @param bufferAllocator the buffer allocator to use
     * @param name the name of the column
     * @param data the input arrays
     * @return
     */
    public static VarBinaryVector vectorFor(BufferAllocator bufferAllocator,String name,INDArray[] data) {
        VarBinaryVector ret = new VarBinaryVector(name,bufferAllocator);
        ret.allocateNew();
        for(int i = 0; i < data.length; i++) {
            //slice the databuffer to use only the needed portion of the buffer
            //for proper offset
            ByteBuffer byteBuffer = ArrowSerde.toTensor(data[i]).getByteBuffer().slice();
            ret.set(i,byteBuffer,0,byteBuffer.capacity());
        }

        return ret;
    }



    /**
     *
     * @param allocator
     * @param name
     * @param data
     * @return
     */
    public static VarCharVector vectorFor(BufferAllocator allocator,String name,String[] data) {
        VarCharVector float4Vector = new VarCharVector(name,allocator);
        float4Vector.allocateNew();
        for(int i = 0; i < data.length; i++) {
            float4Vector.setSafe(i,data[i].getBytes());
        }

        float4Vector.setValueCount(data.length);

        return float4Vector;
    }


    /**
     * Create an ndarray vector that stores structs
     * of {@link INDArray}
     * based on the {@link org.apache.arrow.flatbuf.Tensor}
     * format
     * @param allocator the allocator to use
     * @param name the name of the vector
     * @param length the number of vectors to store
     * @return
     */
    public static VarBinaryVector ndarrayVectorOf(BufferAllocator allocator,String name,int length) {
        VarBinaryVector ret = new VarBinaryVector(name,allocator);
        ret.allocateNew();
        ret.setValueCount(length);
        return ret;
    }

    /**
     *
     * @param allocator
     * @param name
     * @param length the length of the vector
     * @return
     */
    public static VarCharVector stringVectorOf(BufferAllocator allocator,String name,int length) {
        VarCharVector float4Vector = new VarCharVector(name,allocator);
        float4Vector.allocateNew();
        float4Vector.setValueCount(length);
        return float4Vector;
    }



    /**
     *
     * @param allocator
     * @param name
     * @param data
     * @return
     */
    public static Float4Vector vectorFor(BufferAllocator allocator,String name,float[] data) {
        Float4Vector float4Vector = new Float4Vector(name,allocator);
        float4Vector.allocateNew(data.length);
        for(int i = 0; i < data.length; i++) {
            float4Vector.setSafe(i,data[i]);
        }

        float4Vector.setValueCount(data.length);

        return float4Vector;
    }


    /**
     *
     * @param allocator
     * @param name
     * @param length the length of the vector
     * @return
     */
    public static Float4Vector floatVectorOf(BufferAllocator allocator,String name,int length) {
        Float4Vector float4Vector = new Float4Vector(name,allocator);
        float4Vector.allocateNew(length);
        float4Vector.setValueCount(length);
        return float4Vector;
    }

    /**
     *
     * @param allocator
     * @param name
     * @param data
     * @return
     */
    public static Float8Vector vectorFor(BufferAllocator allocator,String name,double[] data) {
        Float8Vector float8Vector = new Float8Vector(name,allocator);
        float8Vector.allocateNew(data.length);
        for(int i = 0; i < data.length; i++) {
            float8Vector.setSafe(i,data[i]);
        }


        float8Vector.setValueCount(data.length);

        return float8Vector;
    }




    /**
     *
     * @param allocator
     * @param name
     * @param length the length of the vector
     * @return
     */
    public static Float8Vector doubleVectorOf(BufferAllocator allocator,String name,int length) {
        Float8Vector float8Vector = new Float8Vector(name,allocator);
        float8Vector.allocateNew();
        float8Vector.setValueCount(length);
        return float8Vector;
    }





    /**
     *
     * @param allocator
     * @param name
     * @param data
     * @return
     */
    public static BitVector vectorFor(BufferAllocator allocator,String name,boolean[] data) {
        BitVector float8Vector = new BitVector(name,allocator);
        float8Vector.allocateNew(data.length);
        for(int i = 0; i < data.length; i++) {
            float8Vector.setSafe(i,data[i] ? 1 : 0);
        }

        float8Vector.setValueCount(data.length);

        return float8Vector;
    }

    /**
     *
     * @param allocator
     * @param name
     * @return
     */
    public static BitVector booleanVectorOf(BufferAllocator allocator,String name,int length) {
        BitVector float8Vector = new BitVector(name,allocator);
        float8Vector.allocateNew(length);
        float8Vector.setValueCount(length);
        return float8Vector;
    }


    /**
     *
     * @param allocator
     * @param name
     * @param data
     * @return
     */
    public static IntVector vectorFor(BufferAllocator allocator,String name,int[] data) {
        IntVector float8Vector = new IntVector(name,FieldType.nullable(new ArrowType.Int(32,true)),allocator);
        float8Vector.allocateNew(data.length);
        for(int i = 0; i < data.length; i++) {
            float8Vector.setSafe(i,data[i]);
        }

        float8Vector.setValueCount(data.length);

        return float8Vector;
    }

    /**
     *
     * @param allocator
     * @param name
     * @return
     */
    public static IntVector intVectorOf(BufferAllocator allocator,String name,int length) {
        IntVector float8Vector = new IntVector(name,FieldType.nullable(new ArrowType.Int(32,true)),allocator);
        float8Vector.allocateNew(length);

        float8Vector.setValueCount(length);

        return float8Vector;
    }




    /**
     *
     * @param allocator
     * @param name
     * @param data
     * @return
     */
    public static BigIntVector vectorFor(BufferAllocator allocator,String name,long[] data) {
        BigIntVector float8Vector = new BigIntVector(name,FieldType.nullable(new ArrowType.Int(64,true)),allocator);
        float8Vector.allocateNew(data.length);
        for(int i = 0; i < data.length; i++) {
            float8Vector.setSafe(i,data[i]);
        }

        float8Vector.setValueCount(data.length);

        return float8Vector;
    }



    /**
     *
     * @param allocator
     * @param name
     * @param length the number of rows in the column vector
     * @return
     */
    public static BigIntVector longVectorOf(BufferAllocator allocator,String name,int length) {
        BigIntVector float8Vector = new BigIntVector(name,FieldType.nullable(new ArrowType.Int(64,true)),allocator);
        float8Vector.allocateNew(length);
        float8Vector.setValueCount(length);
        return float8Vector;
    }

    private static ColumnMetaData metaDataFromField(Field field) {
        ArrowType arrowType = field.getFieldType().getType();
        if(arrowType instanceof ArrowType.Int) {
            val intType = (ArrowType.Int) arrowType;
            if(intType.getBitWidth() == 32)
                return new IntegerMetaData(field.getName());
            else {
                return new LongMetaData(field.getName());
            }
        }
        else if(arrowType instanceof ArrowType.Bool) {
            return new BooleanMetaData(field.getName());
        }
        else if(arrowType  instanceof ArrowType.FloatingPoint) {
            val floatingPointType = (ArrowType.FloatingPoint) arrowType;
            if(floatingPointType.getPrecision() == FloatingPointPrecision.DOUBLE)
                return new DoubleMetaData(field.getName());
            else {
                return new FloatMetaData(field.getName());
            }
        }
        else if(arrowType instanceof  ArrowType.Binary) {
            return new BinaryMetaData(field.getName());
        }
        else if(arrowType instanceof ArrowType.Utf8) {
            return new StringMetaData(field.getName());

        }
        else if(arrowType instanceof ArrowType.Date) {
            return new TimeMetaData(field.getName());
        }
        else {
            throw new IllegalStateException("Illegal type " + field.getFieldType().getType());
        }

    }


    /**
     * Based on an input {@link ColumnType}
     * get an entry from a {@link FieldVector}
     *
     * @param item the row of the item to get from the column vector
     * @param from the column vector from
     * @param columnType the column type
     * @return the resulting writable
     */
    public static Writable fromEntry(int item,FieldVector from,ColumnType columnType) {
        if(from.getValueCount() < item) {
            throw new IllegalArgumentException("Index specified greater than the number of items in the vector with length " + from.getValueCount());
        }

        switch(columnType) {
            case Integer:
                return new IntWritable(getIntFromFieldVector(item,from));
            case Long:
                return new LongWritable(getLongFromFieldVector(item,from));
            case Float:
                return new FloatWritable(getFloatFromFieldVector(item,from));
            case Double:
                return new DoubleWritable(getDoubleFromFieldVector(item,from));
            case Boolean:
                BitVector bitVector = (BitVector) from;
                return new BooleanWritable(bitVector.get(item) > 0);
            case Categorical:
                VarCharVector varCharVector = (VarCharVector) from;
                return new Text(varCharVector.get(item));
            case String:
                VarCharVector varCharVector2 = (VarCharVector) from;
                return new Text(varCharVector2.get(item));
            case Time:
                //TODO: need to look at closer
                return new LongWritable(getLongFromFieldVector(item,from));
            case NDArray:
                VarBinaryVector valueVector = (VarBinaryVector) from;
                byte[] bytes = valueVector.get(item);
                Tensor tensor = Tensor.getRootAsTensor(ByteBuffer.wrap(bytes));
                INDArray fromTensor = ArrowSerde.fromTensor(tensor);
                return new NDArrayWritable(fromTensor);
            default:
                throw new IllegalArgumentException("Illegal type " + from.getClass().getName());
        }
    }


    private static int getIntFromFieldVector(int row,FieldVector fieldVector) {
        if(fieldVector instanceof UInt4Vector) {
            UInt4Vector uInt4Vector = (UInt4Vector) fieldVector;
            return uInt4Vector.get(row);
        }
        else if(fieldVector instanceof IntVector) {
            IntVector intVector = (IntVector) fieldVector;
            return intVector.get(row);
        }

        throw new IllegalArgumentException("Illegal vector type for int " + fieldVector.getClass().getName());
    }

    private static long getLongFromFieldVector(int row,FieldVector fieldVector) {
        if(fieldVector instanceof UInt8Vector) {
            UInt8Vector uInt4Vector = (UInt8Vector) fieldVector;
            return uInt4Vector.get(row);
        }
        else if(fieldVector instanceof IntVector) {
            BigIntVector intVector = (BigIntVector) fieldVector;
            return intVector.get(row);
        }
        else if(fieldVector instanceof TimeStampMilliVector) {
            TimeStampMilliVector timeStampMilliVector = (TimeStampMilliVector) fieldVector;
            return timeStampMilliVector.get(row);
        }
        else if(fieldVector instanceof BigIntVector) {
            BigIntVector bigIntVector = (BigIntVector) fieldVector;
            return bigIntVector.get(row);
        }
        else if (fieldVector instanceof DateMilliVector) {
            DateMilliVector dateMilliVector = (DateMilliVector) fieldVector;
            return dateMilliVector.get(row);

        }
        else if(fieldVector instanceof TimeStampMilliVector) {
            TimeStampMilliVector timeStampMilliVector = (TimeStampMilliVector) fieldVector;
            return timeStampMilliVector.get(row);
        }
        else if(fieldVector instanceof TimeMilliVector) {
            TimeMilliVector timeMilliVector = (TimeMilliVector) fieldVector;
            return timeMilliVector.get(row);
        }
        else if(fieldVector instanceof TimeStampMicroVector) {
            TimeStampMicroVector timeStampMicroVector = (TimeStampMicroVector) fieldVector;
            return timeStampMicroVector.get(row);
        }
        else if(fieldVector instanceof TimeSecVector) {
            TimeSecVector timeSecVector = (TimeSecVector) fieldVector;
            return timeSecVector.get(row);
        }
        else if(fieldVector instanceof TimeStampMilliVector) {
            TimeStampMilliVector timeStampMilliVector = (TimeStampMilliVector) fieldVector;
            return timeStampMilliVector.get(row);
        }
        else if(fieldVector instanceof TimeStampMilliTZVector) {
            TimeStampMilliTZVector timeStampMilliTZVector = (TimeStampMilliTZVector) fieldVector;
            return timeStampMilliTZVector.get(row);
        }
        else if(fieldVector instanceof TimeStampNanoTZVector) {
            TimeStampNanoTZVector timeStampNanoTZVector = (TimeStampNanoTZVector) fieldVector;
            return timeStampNanoTZVector.get(row);
        }
        else if(fieldVector instanceof TimeStampMicroTZVector) {
            TimeStampMicroTZVector timeStampMicroTZVector = (TimeStampMicroTZVector) fieldVector;
            return timeStampMicroTZVector.get(row);
        }
        else {
            throw new UnsupportedOperationException();
        }

    }

    private static double getDoubleFromFieldVector(int row,FieldVector fieldVector) {
        if(fieldVector instanceof Float8Vector) {
            Float8Vector uInt4Vector = (Float8Vector) fieldVector;
            return uInt4Vector.get(row);
        }


        throw new IllegalArgumentException("Illegal vector type for int " + fieldVector.getClass().getName());
    }


    private static float getFloatFromFieldVector(int row,FieldVector fieldVector) {
        if(fieldVector instanceof Float4Vector) {
            Float4Vector uInt4Vector = (Float4Vector) fieldVector;
            return uInt4Vector.get(row);
        }


        throw new IllegalArgumentException("Illegal vector type for int " + fieldVector.getClass().getName());
    }


    private static ArrowWritableRecordBatch asDataVecBatch(ArrowRecordBatch arrowRecordBatch, Schema schema, VectorSchemaRoot vectorLoader) {
        //iterate column wise over the feature vectors, returning entries
        List<FieldVector> fieldVectors = new ArrayList<>();
        for(int j = 0; j < schema.numColumns(); j++) {
            String name = schema.getName(j);
            FieldVector fieldVector = vectorLoader.getVector(name);
            fieldVectors.add(fieldVector);
        }

        ArrowWritableRecordBatch ret = new ArrowWritableRecordBatch(fieldVectors, schema);
        ret.setArrowRecordBatch(arrowRecordBatch);

        return ret;
    }



}
