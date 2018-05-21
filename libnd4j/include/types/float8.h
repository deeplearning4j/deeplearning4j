//
// Created by raver119 on 10.08.16.
//

#ifndef LIBND4J_FLOAT8_H
#define LIBND4J_FLOAT8_H

/*
#ifdef __CUDACC__
#define local_def __host__ __device__
#elif _MSC_VER
#define local_def
#elif __clang__
#define local_def
#elif __GNUC__
#define local_def
#endif
*/


namespace nd4j {

    typedef struct {
        unsigned char x;
    } __quarter;

    typedef __quarter quarter;

    quarter cpu_float2quarter_rn(float f);
    float cpu_quarter2float(quarter b);

    struct float8 {
        quarter data;

        float8();

        template <class T>
        float8(const T& rhs);

        template <class T>
        float8& operator=(const T& rhs);

        operator float() const;

        void assign(double rhs);

        void assign(float rhs);
    };
}

#endif //LIBND4J_FLOAT8_H
