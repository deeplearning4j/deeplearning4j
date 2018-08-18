from ..java_classes import SparkConf, SparkContext, SparkTransformExecutor
from ..java_classes import CSVRecordReader, WritablesToStringFunction, StringToWritablesFunction


class StringRDD(object):

    def __init__(self, java_rdd):
        self.java_rdd = java_rdd

    def __iter__(self):
        jlist = self.java_rdd.collect()
        size = jlist.size()
        return iter([jlist.get(i) for i in range(size)])

    def save(self, path):
        self.java_rdd.saveAsTextFile(path)

    def save_to_csv(self, path):
        l = list(self)
        with open(path, 'w') as f:
            for x in l:
                f.write(x + '\n')


class SparkExecutor(object):

    def __init__(self, master='local[*]', app_name='pydatavec'):
        spark_conf = SparkConf()
        spark_conf.setMaster(master)
        spark_conf.setAppName(app_name)
        self.spark_context = SparkContext(spark_conf)
        self.rr = CSVRecordReader()

    def __call__(self, tp, csv):
        string_data = self.spark_context.textFile(csv)  # JavaRDD<String>
        parsed_input_data = string_data.map(StringToWritablesFunction(self.rr))   # JavaRDD<List<Writable>>
        processed_data = SparkTransformExecutor.execute(parsed_input_data, tp.to_java())  # JavaRDD<List<Writable>>
        processed_as_string = processed_data.map(WritablesToStringFunction(","))  # JavaRDD<String>
        return StringRDD(processed_as_string)  # StringRDD
