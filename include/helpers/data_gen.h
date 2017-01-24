//
// Created by agibsonccc on 1/7/17.
//

#ifndef LIBND4J_DATA_GEN_H_H
#define LIBND4J_DATA_GEN_H_H
/**
 * Returns a linear space array
 * @tparam T  the type
 * @param lower  the lower bound
 * @param upper  the upper bound
 * @param num  the number of elements to generate
 * @return the linear spaced array
 */
template <typename T>
T * linspace(int lower, int upper, int num) {
    T *data = new T[num];
    for (int i = 0; i < num; i++) {
        T t = (T) i / (num - 1);
        data[i] = lower * (1 - t) + t * upper;

    }

    return data;
}

#endif //LIBND4J_DATA_GEN_H_H
