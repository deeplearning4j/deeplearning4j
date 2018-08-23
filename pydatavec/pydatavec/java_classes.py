import jnius_config
import os


class_path = os.environ.get('DATAVEC_CLASS_PATH')

if class_path is None:
    raise Exception('Environment variable DATAVEC_CLASS_PATH not set.')

if class_path[-1] == '/':
    class_path += '*'
else:
    class_path += '/*'


jnius_config.set_classpath(class_path)

# -------------JVM starts here-------------
from jnius import autoclass


JSchema = autoclass('org.datavec.api.transform.schema.Schema')
SchemaBuilder = autoclass('org/datavec/api/transform/schema/Schema$Builder')

JTransformProcess = autoclass('org.datavec.api.transform.TransformProcess')
TransformProcessBuilder = autoclass('org/datavec/api/transform/TransformProcess$Builder')

ConditionOp = autoclass('org.datavec.api.transform.condition.ConditionOp')
ConditionFilter = autoclass('org.datavec.api.transform.filter.ConditionFilter')

BooleanColumnCondition = autoclass('org.datavec.api.transform.condition.column.BooleanColumnCondition')
CategoricalColumnCondition = autoclass('org.datavec.api.transform.condition.column.CategoricalColumnCondition')
DoubleColumnCondition = autoclass('org.datavec.api.transform.condition.column.DoubleColumnCondition')
#FloatColumnCondition = autoclass('org.datavec.api.transform.condition.column.FloatColumnCondition')
StringColumnCondition = autoclass('org.datavec.api.transform.condition.column.StringColumnCondition')


BooleanWritable = autoclass('org.datavec.api.writable.BooleanWritable')
IntegerWritable = autoclass('org.datavec.api.writable.IntWritable')
LongWritable = autoclass('org.datavec.api.writable.LongWritable')
FloatWritable = autoclass('org.datavec.api.writable.FloatWritable')
DoubleWritable = autoclass('org.datavec.api.writable.DoubleWritable')



DateTimeZone = autoclass('org.joda.time.DateTimeZone')
DateTimeFieldType = autoclass('org.joda.time.DateTimeFieldType')
DeriveColumnsFromTimeTransformBuilder = autoclass('org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform$Builder')


Arrays = autoclass('java.util.Arrays')
HashSet = autoclass('java.util.HashSet')


JDouble = autoclass('java.lang.Double')
JFloat = autoclass('java.lang.Float')

Arrays = autoclass('java.util.Arrays')
JMap = autoclass('java.util.HashMap')


SparkConf = autoclass('org.apache.spark.SparkConf')
SparkContext = autoclass('org.apache.spark.api.java.JavaSparkContext')
JavaRDD = autoclass('org.apache.spark.api.java.JavaRDD')
SparkTransformExecutor = autoclass('org.datavec.spark.transform.SparkTransformExecutor')

CSVRecordReader = autoclass('org.datavec.api.records.reader.impl.csv.CSVRecordReader')
StringToWritablesFunction = autoclass('org.datavec.spark.transform.misc.StringToWritablesFunction')
WritablesToStringFunction = autoclass('org.datavec.spark.transform.misc.WritablesToStringFunction')

#LocalTransformExecutor = autoclass('org.datavec.local.transforms.LocalTransformExecutor')

ChangeCaseStringTransform = autoclass('org.datavec.api.transform.transform.string.ChangeCaseStringTransform')
ChangeCaseStringTransformCaseType = autoclass('org.datavec.api.transform.transform.string.ChangeCaseStringTransform$CaseType')
ConcatenateStringColumns = autoclass('org.datavec.api.transform.transform.string.ConcatenateStringColumns')
RemoveWhiteSpaceTransform = autoclass('org.datavec.api.transform.transform.string.RemoveWhiteSpaceTransform')
ReplaceEmptyStringTransform = autoclass('org.datavec.api.transform.transform.string.ReplaceEmptyStringTransform')
ReplaceStringTransform = autoclass('org.datavec.api.transform.transform.string.ReplaceStringTransform')
StringMapTransform = autoclass('org.datavec.api.transform.transform.string.StringMapTransform')
