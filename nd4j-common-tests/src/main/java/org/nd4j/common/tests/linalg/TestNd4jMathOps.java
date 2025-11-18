public class TestNd4jMathOps extends BaseNd4jTestWithBackends {

    @Test
    public void testAdd() {
        INDArray a = Nd4j.create(new float[]{1,2,3});
        INDArray b = Nd4j.create(new float[]{4,5,6});
        INDArray result = a.add(b);

        assertEquals(Nd4j.create(new float[]{5,7,9}), result);
    }

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }
}
