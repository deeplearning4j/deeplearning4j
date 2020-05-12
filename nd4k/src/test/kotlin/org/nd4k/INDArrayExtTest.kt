package org.nd4k

import org.junit.Assert.*
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms.exp
import org.junit.Test as test

class INDArrayExtTest {

    @test fun plusBetweenINDArray() {
        val a = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val b = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val expected = Nd4j.create(floatArrayOf(2f, 4f, 6f, 8f, 10f), intArrayOf(1, 5))
        assertEquals(expected, a + b)
    }

    @test fun plusBetweenNumberAndINDArray() {
        val a = 1
        val b = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val expected = Nd4j.create(floatArrayOf(2f, 3f, 4f, 5f, 6f), intArrayOf(1, 5))
        assertEquals(expected, a + b)
    }

    @test fun plusBetweenINDArrayAndNumber() {
        val a = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val b = 1
        val expected = Nd4j.create(floatArrayOf(2f, 3f, 4f, 5f, 6f), intArrayOf(1, 5))
        assertEquals(expected, a + b)
    }

    @test fun unaryMinusOnINDArray() {
        val a = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val expected = Nd4j.create(floatArrayOf(-1f, -2f, -3f, -4f, -5f), intArrayOf(1, 5))
        assertEquals(expected, -a)
    }

    @test fun minusBetweenINDArray() {
        val a = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val b = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val expected = Nd4j.create(floatArrayOf(0f, 0f, 0f, 0f, 0f), intArrayOf(1, 5))
        assertEquals(expected, a - b)
    }

    @test fun minusBetweenNumberAndINDArray() {
        val a = 1
        val b = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val expected = Nd4j.create(floatArrayOf(0f, -1f, -2f, -3f, -4f), intArrayOf(1, 5))
        assertEquals(expected, a - b)
    }

    @test fun minusBetweenINDArrayAndNumber() {
        val a = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val b = 1
        val expected = Nd4j.create(floatArrayOf(0f, 1f, 2f, 3f, 4f), intArrayOf(1, 5))
        assertEquals(expected, a - b)
    }

    @test fun timesBetweenINDArray() {
        val a = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2))
        val b = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2))
        val expected = Nd4j.create(floatArrayOf(1f, 4f, 9f, 16f), intArrayOf(2, 2))
        assertEquals(expected, a * b) // not dot() !!!
    }

    @test fun timesBetweenNumberAndINDArray() {
        val a = 2
        val b = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val expected = Nd4j.create(floatArrayOf(2f, 4f, 6f, 8f, 10f), intArrayOf(1, 5))
        assertEquals(expected, a * b)
    }

    @test fun timesBetweenINDArrayAndNumber() {
        val a = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f), intArrayOf(1, 5))
        val b = -3
        val expected = Nd4j.create(floatArrayOf(-3f, -6f, -9f, -12f, -15f), intArrayOf(1, 5))
        assertEquals(expected, a * b)
    }

    @test fun divBetweenINDArray() {
        val a = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2))
        val b = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2))
        val expected = Nd4j.create(floatArrayOf(1f, 1f, 1f, 1f), intArrayOf(2, 2))
        assertEquals(expected, a / b)
    }

    @test fun divBetweenNumberAndINDArray() {
        val a = 10
        val b = Nd4j.create(floatArrayOf(1f, 2f, 5f), intArrayOf(1, 3))
        val expected = Nd4j.create(floatArrayOf(10f, 5f, 2f), intArrayOf(1, 3))
        assertEquals(expected, a / b)
    }

    @test fun divBetweenINDArrayAndNumber() {
        val a = Nd4j.create(floatArrayOf(15f, 10f, 5f), intArrayOf(1, 3))
        val b = 5
        val expected = Nd4j.create(floatArrayOf(3f, 2f, 1f), intArrayOf(1, 3))
        assertEquals(expected, a / b)
    }

    @test fun sigmoidUsingOperator() {
        fun sigmoidUsingOperator(x: INDArray): INDArray { return 1.0 / (1.0 + exp(-x)) }
        val a = Nd4j.create(floatArrayOf(0.2f), intArrayOf(1, 1))
        assertEquals(Transforms.sigmoid(a), sigmoidUsingOperator(a))
    }

    @test fun sigmoidDerivativeUsingOperator() {
        fun s(x: INDArray): INDArray { return 1.0 / (1.0 + exp(-x)) }
        fun sigmoidDerivativeUsingOperator(x: INDArray): INDArray { return ( 1 - s(x) ) * s(x) }
        val a = Nd4j.create(floatArrayOf(0.2f), intArrayOf(1, 1))
        assertEquals(Transforms.sigmoidDerivative(a), sigmoidDerivativeUsingOperator(a))
    }

}

