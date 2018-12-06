package org.deeplearning4j.spark.parameterserver.python;

import org.apache.spark.api.java.JavaRDD;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import javax.xml.crypto.Data;

public class Utils {

    private static  ArrayDescriptor getArrayDescriptor(INDArray arr) throws Exception{
        return new ArrayDescriptor(arr);
    }

    private static INDArray getArray(ArrayDescriptor arrDesc){
        return arrDesc.getArray();
    }

    private static DataSetDescriptor getDataSetDescriptor(DataSet ds)throws Exception{
        return new DataSetDescriptor(ds);
    }
    
    private static  DataSet getDataSet(DataSetDescriptor dsDesc){
        return dsDesc.getDataSet();
    }
    public static JavaRDD<ArrayDescriptor> getArrayDescriptorRDD(JavaRDD<INDArray> indarrayRDD){
        return indarrayRDD.map(Utils::getArrayDescriptor);
    }

    public static  JavaRDD<INDArray> getArrayRDD(JavaRDD<ArrayDescriptor> arrayDescriptorRDD){
        return arrayDescriptorRDD.map(ArrayDescriptor::getArray);
    }

    public static JavaRDD<DataSetDescriptor> getDatasetDescriptorRDD(JavaRDD<DataSet> dsRDD){
        return dsRDD.map(Utils::getDataSetDescriptor);
    }

    public static JavaRDD<DataSet> getDataSetRDD(JavaRDD<DataSetDescriptor> dsDescriptorRDD){
        return dsDescriptorRDD.map(Utils::getDataSet);
    }
}
