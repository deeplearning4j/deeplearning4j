import org.eclipse.python4j.NumpyArray;
import org.eclipse.python4j.Python;
import org.eclipse.python4j.PythonGC;
import org.eclipse.python4j.PythonObject;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class PythonNumpyImportTest {

    @Test
    public void testNumpyImport(){
        try(PythonGC gc = PythonGC.watch()){
            PythonObject np = Python.importModule("numpy");
            PythonObject zeros = np.attr("zeros").call(5);
            INDArray arr = NumpyArray.INSTANCE.toJava(zeros);
            Assert.assertEquals(arr, Nd4j.zeros(DataType.DOUBLE, 5));
        }
    }
}
