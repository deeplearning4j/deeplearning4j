package org.nd4j.codegen.api

enum class Namespace {
    BASE, CNN, IMAGE, LOSS, MATH, NN, RANDOM, RNN;

    fun displayName() = name[0].toString() + name.substring(1).toLowerCase()
}