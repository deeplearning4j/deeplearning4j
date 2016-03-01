package org.nd4j.linalg.util;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

public class TestArrayUtils extends BaseNd4jTest {

    public TestArrayUtils(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testFlattenDoubleArray(){
        assertArrayEquals(new double[0],ArrayUtil.flattenDoubleArray(new double[0]),0.0);
        Random r = new Random(12345L);

        double[] d1 = new double[10];
        for( int i=0; i<d1.length; i++ ) d1[i] = r.nextDouble();
        assertArrayEquals(d1,ArrayUtil.flattenDoubleArray(d1),0.0);

        double[][] d2 = new double[5][10];
        for( int i=0; i<5; i++) for( int j=0; j<10; j++ ) d2[i][j] = r.nextDouble();
        assertArrayEquals(ArrayUtil.flatten(d2),ArrayUtil.flattenDoubleArray(d2),0.0);

        double[][][] d3 = new double[5][10][15];
        double[] exp3 = new double[5*10*15];
        int c = 0;
        for( int i=0; i<5; i++ ){
            for( int j=0; j<10; j++ ){
                for( int k=0; k<15; k++ ){
                    double d = r.nextDouble();
                    exp3[c++] = d;
                    d3[i][j][k] = d;
                }
            }
        }
        assertArrayEquals(exp3,ArrayUtil.flattenDoubleArray(d3),0.0);


        double[][][][] d4 = new double[3][5][7][9];
        double[] exp4 = new double[3*5*7*9];
        c = 0;
        for( int i=0; i<3; i++ ){
            for( int j=0; j<5; j++ ){
                for( int k=0; k<7; k++ ){
                    for( int l=0; l<9; l++ ){
                        double d = r.nextDouble();
                        exp4[c++] = d;
                        d4[i][j][k][l] = d;
                    }
                }
            }
        }
        assertArrayEquals(exp4,ArrayUtil.flattenDoubleArray(d4),0.0);
    }

    @Test
    public void testFlattenFloatArray(){
        assertArrayEquals(new float[0],ArrayUtil.flattenFloatArray(new float[0]),0.0f);
        Random r = new Random(12345L);

        float[] f1 = new float[10];
        for( int i=0; i<f1.length; i++ ) f1[i] = r.nextFloat();
        assertArrayEquals(f1,ArrayUtil.flattenFloatArray(f1),0.0f);

        float[][] f2 = new float[5][10];
        for( int i=0; i<5; i++) for( int j=0; j<10; j++ ) f2[i][j] = r.nextFloat();
        assertArrayEquals(ArrayUtil.flatten(f2),ArrayUtil.flattenFloatArray(f2),0.0f);

        float[][][] f3 = new float[5][10][15];
        float[] exp3 = new float[5*10*15];
        int c = 0;
        for( int i=0; i<5; i++ ){
            for( int j=0; j<10; j++ ){
                for( int k=0; k<15; k++ ){
                    float d = r.nextFloat();
                    exp3[c++] = d;
                    f3[i][j][k] = d;
                }
            }
        }
        assertArrayEquals(exp3,ArrayUtil.flattenFloatArray(f3),0.0f);


        float[][][][] f4 = new float[3][5][7][9];
        float[] exp4 = new float[3*5*7*9];
        c = 0;
        for( int i=0; i<3; i++ ){
            for( int j=0; j<5; j++ ){
                for( int k=0; k<7; k++ ){
                    for( int l=0; l<9; l++ ){
                        float d = r.nextFloat();
                        exp4[c++] = d;
                        f4[i][j][k][l] = d;
                    }
                }
            }
        }
        assertArrayEquals(exp4,ArrayUtil.flattenFloatArray(f4),0.0f);
    }

    @Test
    public void testArrayShape(){
        assertArrayEquals(ArrayUtil.arrayShape(new int[0]),new int[]{0});
        assertArrayEquals(ArrayUtil.arrayShape(new int[5][7][9]),new int[]{5,7,9});
        assertArrayEquals(ArrayUtil.arrayShape(new Object[2][3][4][5][6]),new int[]{2,3,4,5,6});
        assertArrayEquals(ArrayUtil.arrayShape(new double[9][7][5][3]),new int[]{9,7,5,3});
        assertArrayEquals(ArrayUtil.arrayShape(new double[1][1][1][0]),new int[]{1,1,1,0});
        assertArrayEquals(ArrayUtil.arrayShape(new char[3][2][1]),new int[]{3,2,1});
        assertArrayEquals(ArrayUtil.arrayShape(new String[3][2][1]),new int[]{3,2,1});
    }

    @Test
    public void testArgMinOfMaxMethods(){
        int[] first = {1,5,2,4};
        int[] second = {4,6,3,2};

        assertEquals(2, ArrayUtil.argMinOfMax(first, second));

        int[] third = {7,3,8,10};
        assertEquals(1, ArrayUtil.argMinOfMax(first, second, third));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
