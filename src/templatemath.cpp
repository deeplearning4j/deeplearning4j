//
// Created by agibsonccc on 1/3/16.
//

#include <templatemath.h>
#include <stdlib.h>

namespace nd4j {
    namespace math {

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_abs<float>(float value) {
            return fabsf(value);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_abs<double>(double value) {
            return abs(value);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_abs<int>(int value) {
            return abs(value);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_max<float>(float val1, float val2) {
            return fmaxf(val1,val2);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_max<double>(double val1, double val2) {
            return fmax(val1,val2);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_max<int>(int val1, int val2) {
            return fmax(val1,val2);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_min<float>(float val1, float val2) {
            return fminf(val1,val2);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_min<double>(double val1, double val2) {
            return fmin(val1,val2);
        }
        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_min<int>(int val1, int val2) {
            return fmin(val1,val2);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_ceil<float>(float val1) {
            return ceilf(val1);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_ceil<double>(double val) {
            return ceil(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_ceil<int>(int val) {
            return ceil(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_cos<float>(float val) {
            return cosf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_cos<double>(double val) {
            return cos(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_cos<int>(int val) {
            return cos(val);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_exp<float>(float val) {
            return expf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_exp<double>(double val) {
            return exp(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_exp<int>(int val) {
            return exp(val);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_floor<float>(float val) {
            return floorf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_floor<double>(double val) {
            return floor(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_floor<int>(int val) {
            return floor(val);
        }




        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_log<float>(float val) {
            return logf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_log<double>(double val) {
            return log(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_log<int>(int val) {
            return log(val);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_pow<float>(float val, float val2) {
            return powf(val,val2);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_pow<double>(double val, double val2) {
            return pow(val,val2);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_pow<int>(int val, int val2) {
            return pow(val,val2);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_round<float>(float val) {
            return roundf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_round<double>(double val) {
            return round(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_round<int>(int val) {
            return round(val);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_sin<float>(float val) {
            return sinf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_sin<double>(double val) {
            return sin(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_sin<int>(int val) {
            return sin(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_sqrt<float>(float val) {
            return sqrtf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_sqrt<double>(double val) {
            return sqrt(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_sqrt<int>(int val) {
            return sqrt(val);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_tanh<float>(float val) {
            return tanhf(val);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_tanh<double>(double val) {
            return tanh(val);
        }
        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_tanh<int>(int val) {
            return tanh(val);
        }
        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_acos<float>(float val) {
            return acosf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_acos<double>(double val) {
            return acos(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_acos<int>(int val) {
            return acos(val);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_asin<float>(float val) {
            return asinf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_asin<double>(double val) {
            return asin(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_asin<int>(int val) {
            return asin(val);
        }


        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        float nd4j_atan<float>(float val) {
            return atanf(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        double nd4j_atan<double>(double val) {
            return atan(val);
        }

        template <>
#ifdef __CUDACC__
        __host__ __device__
#endif
        int nd4j_atan<int>(int val) {
            return atan(val);
        }

    }
}