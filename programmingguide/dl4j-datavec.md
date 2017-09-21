title: DeepLearning4j: DataVec
layout: default

------

 # DeepLearning4j: DataVec

 In this chapter, we will introduce DataVec for vectorization and ETL (extract, transform, load). DataVec is used primarily for getting raw data into a format that neural networks can read. DataVec can be used to convert all major types of data such as text, CSV, audio, images, and video and additionally apply scaling, normalization, or other transformations to the vectorized data. Furthermore, the functionality of DataVec can be extended for specialized inputs such as more exotic image formats. 

 ## Overview of DataVec Tools

 ### Schemas and TransformProcesses

 Schemas are used to define the layout of tabular data. The basic way to initialize a Schema is as follows:

      Schema inputDataSchema = new Schema.Builder()
        .addColumnString("DateTimeString")
 	      .addColumnsString("CustomerID", "MerchantID")
 	      .addColumnInteger("NumItemsInTransaction")
 	      .addColumnCategorical("MerchantCountryCode", Arrays.asList("USA","CAN","FR","MX"))
 	      .addColumnDouble("TransactionAmountUSD",0.0,null,false,false)
 	      .build();

 The columns should be added in the Schema in the order they appear in the data. In the TransactionAMountUSD column, we further specify that the amount should be non-negeative and have no maximum limit, NaN, or infinite values. 

 Once the Schema is defined, you can print it out to look at the details.

      System.out.println(inputDataSchema);

 To perform transformation processes on the data using the Schema we defined, a TransformProcess is needed. TransformProcesses can remove unnecesasry columns, filter out observations, make new variables, rename columns, and more. Below is code for defining a TransformProcess, which takes the inputDataSchema as input. 

      TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
	      .removeColumns("CustomerID","MerchantID")
	      .conditionalReplaceValueTransform(
                "TransactionAmountUSD",     //Column to operate on
                new DoubleWritable(0.0),    //New value to use, when the condition is satisfied
                new DoubleColumnCondition("TransactionAmountUSD",ConditionOp.LessThan, 0.0)) //Condition: amount < 0.0
	      .stringToTimeTransform("DateTimeString","YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)
	    .renameColumn("DateTimeString", "DateTime")

 The above TransformProcess first removes the unnecessary columns CustomerID and MerchantID of our Schema. The next transformation acts on the TransactionAmountUSD and replaces values less than 0 with 0. Lastly, the TransformProcess converts DateTimeString to a format like 2016/01/01 17:50.000 and renames the column as DateTime.

 A TransformProcess can then used to output a final Schema, which we can view. This is done as follows:

      Schema outputSchema = tp.getFinalSchema();
      System.out.println(outputSchema);

 To actually execute the transformations, Spark is needed. The first step is to set up a Spark Context which is done as follows:

      SparkConf conf = new SparkConf();
      conf.setMaster("local[*]");
      conf.setAppName("DataVec Example");

      JavaSparkContext sc = new JavaSparkContext(conf);

 We will assume that the data is contained in BasicDataVecExample/exampledata.csv, and we will create a JavaRDD from the raw data.

      String path =new ClassPathResource("BasicDataVecExample/exampledata.csv").getFile().getAbsolutePath();
      JavaRDD<String> stringData = sc.textFile(path);

 Lastly, a RecordReader is needed, which we will go into detail in the next section. They are needed to parse the data into record format.
        
       RecordReader rr = new CSVRecordReader();
       JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

 Finally to execute the transformation on the parsed data, a SparkTransformExecutor is used.

      JavaRDD<List<Writable>> processedData = SparkTransformExecutor.execute(parsedInputData, tp);

 ### Record Readers

 Record Readers are a class in the DataVec library used to serialize or parse data into records, i.e., a collection of elements indexed by a unique ID. They are the first step to getting the data into a format neural networks can understand. Depending on the data, different subclasses of Record Readers should be used. For example, if the data is in a CSV file, then CSVRecordReader should be  used, and if the data is contained in an image, then ImageRecordReader should be used (and etc). 

 To initialize a Record Reader, simply use code similar to the lines below.

      CSVRecordReader features = new CSVRecordReader();
      recordReader.initialize(new FileSplit( new File(path)));

 In the above example, the data is assuemd to be in CSV format, which is why CSVRecordReader is used. The CSVRecordReader can also take in optional parameters to skip a specified number of lines and specify a delimiter.

 For an image, ImageRecordReader should be used. We can see that there are only a few differences in parameters between different RecordReaders in the below example.

      ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
      ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
      recordReader.initialize(new FileSplit(parentDir));

 Here, labelMaker is a ParentPathLabelGenerator, which is used if the labels aren't manually specified. The ParentPathLabelGenerator will parse the parent directory and use the name of the subdirectories containing the data as the label and class names.  

 ### Dataset Iterators

 Once the raw data is converted into record format, they will need to be processed into DataSet objects, which can be fed directly into a neural network. To do this, DataSetIterators are used to traverse records sequentially using a RecordReader object. DataSetIterators iterate through the input datasets, fetch examples at each iteration, and load them in a DataSet object, which is a INDArray. The number of examples fetched at each iteration depends on the batch size of the intended neural network. Once the DataSet object is created, the data is ready for use.

 Below is an example of initializing a DataSetIterator. The parameters are the DataVec RecordReader, batch size, the offset of the label index, and the total number of label classes.

 DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum)

 Other types of DataSetIterators can also be used depending on the data and the neural network. For example, if the neural network has multiple inputs or outputs, a MultiDataSetIterator should be used. This is similar to a DataSetIterator but multiple inputs and outputs can be defined. These inputs and outputs need to be in Record Reader format like before. In the example below, there are one set of inputs and outputs each but more can be added if needed.

     MultiDataSetIterator trainData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                 .addReader("trainFeatures", trainFeatures)
                .addInput("trainFeatures")
                .addReader("trainLabels", trainLabels)
                .addOutput("trainLabels")
                .build();

 Once the DataSetIterator is initialized, further transformations can be applied. For example, you can scale the data as follows:

    DataNormalization scaler = new ImagePreProcessingScaler(0,1);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);

 Finally when we are content with the data, we can create DataSet objects by iterating through the data. The DataSetIterator will fetch a batch of data points in a DataSet format. 

      DataSet next = iterator.next();

 With these DataSet objects, the data is now ready to be read by a neural network.

 ### Spark

 If Spark is used for the job, a JavaRDD<DataSet> object must be created before passing the data into a neural network.  The following snippet of code is one way to obtain a JavaRDD<DataSet>. We will assume that trainData is a DataSetIterator and that sc is a JavaSparkContext.

      List<DataSet> trainDataList = new ArrayList<>();

      while (trainData.hasNext()) {
 	      trainDataList.add(trainData.next());
      }

      JavaRDD<DataSet> JtrainData = sc.parallelize(trainDataList);

 Like before, once a JavaRDD<DataSet> is created, the data is ready to be fed into a neural network using Spark. 
