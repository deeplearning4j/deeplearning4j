public class TestNd4jCreation extends BaseNd4jTestWithBackends {

    @Test
    public void testZeros() {
        INDArray arr = Nd4j.zeros(3,3);
        assertEquals(0.0, arr.meanNumber().doubleValue(), 1e-6);
    }

    @Test
    public void testOnes() {
        INDArray arr = Nd4j.ones(2,2);
        assertEquals(1.0, arr.meanNumber().doubleValue(), 1e-6);
    }

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }
}
