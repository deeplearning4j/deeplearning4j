@Test
public void testReshape() {
    INDArray arr = Nd4j.linspace(1,6,6);
    INDArray reshaped = arr.reshape(2,3);
    assertArrayEquals(new long[]{2,3}, reshaped.shape());
}
