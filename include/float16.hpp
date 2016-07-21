/*

 Half-precision data type, based on NVIDIA code: https://github.com/NVIDIA/caffe/tree/experimental/fp16

 */

#ifndef CAFFE_UTIL_FP16_H_
#define CAFFE_UTIL_FP16_H_

#include <cfloat>
#include <iosfwd>
#include <fp16_emu.h>
#include <fp16_conversion.hpp>

#ifdef __CUDACC__
#define op_def inline __host__ __device__
#elif _MSC_VER
#define op_def inline
#elif __clang__
#define op_def inline
#elif __GNUC__
#define op_def inline
#endif

namespace nd4j
{

  struct float16
  {
    /* constexpr */ op_def float16() { data.x = 0; }
    
    template <class T>
    op_def /*explicit*/ float16(const T& rhs) {
      assign(rhs);
    }
      
//    op_def float16(float rhs) {
//      assign(rhs);
//    }
//
//    op_def float16(double rhs) {
//      assign(rhs);
//    }

    op_def operator float() const {
#ifdef __CUDA_ARCH__
      return __half2float(data);
#else
      return cpu_half2float(data);
#endif
    }

    //    op_def operator double() const { return (float)*this; } 
       
    op_def operator half() const { return data; }

    op_def unsigned short getx() const { return data.x; }
    op_def float16& setx(unsigned short x) { data.x = x; return *this; }

    template <class T>
    op_def float16& operator=(const T& rhs) { assign(rhs); return *this; }

    op_def void assign(unsigned int rhs) {
      // may be a better way ?
      assign((float)rhs);
    }

    op_def void assign(int rhs) {
      // may be a better way ?
      assign((float)rhs);
    }

    op_def void assign(double rhs) {
      assign((float)rhs);
    }
   
    op_def void assign(float rhs) {
#ifdef __CUDA_ARCH__
      data.x = __float2half_rn(rhs);
#else
  #if defined(DEBUG) && defined (CPU_ONLY)
      if (rhs > HLF_MAX || rhs < -HLF_MAX) {
        LOG(WARNING) << "Overflow: " << rhs;
      } else if (rhs != 0.F && rhs < HLF_MIN && rhs > -HLF_MIN) {
        LOG(WARNING) << "Underflow: " << rhs;
      }
  #endif
      data = cpu_float2half_rn(rhs);
#endif
    }

    op_def void assign(const half& rhs) {
      data = rhs;
    }

    op_def void assign(const float16& rhs) {
      data = rhs.data;
    }

    op_def float16& operator+=(float16 rhs) { assign((float)*this + rhs); return *this; }  

    op_def float16& operator-=(float16 rhs) { assign((float)*this - rhs); return *this; }  

    op_def float16& operator*=(float16 rhs) { assign((float)*this * rhs); return *this; }   

    op_def float16& operator/=(float16 rhs) { assign((float)*this / rhs); return *this; }  

    op_def float16& operator+=(float rhs) { assign((float)*this + rhs); return *this; }  

    op_def float16& operator-=(float rhs) { assign((float)*this - rhs); return *this; }  

    op_def float16& operator*=(float rhs) { assign((float)*this * rhs); return *this; }  

    op_def float16& operator/=(float rhs) { assign((float)*this / rhs); return *this; }  

    op_def float16& operator++() { assign(*this + 1.f); return *this; }  

    op_def float16& operator--() { assign(*this - 1.f); return *this; }  

    op_def float16 operator++(int i) { assign(*this + (float)i); return *this; }  

    op_def float16 operator--(int i) { assign(*this - (float)i); return *this; }  


    half data;

    // Utility contants
    static const float16 zero;
    static const float16 one;
    static const float16 minus_one;
  };

//  op_def bool  operator==(const float16& a, const float16& b) { return ishequ(a.data, b.data); }
//
//  op_def bool  operator!=(const float16& a, const float16& b) { return !(a == b); }
//
//  op_def bool  operator<(const float16& a, const float16& b) { return (float)a < (float)b; }
//
//  op_def bool  operator>(const float16& a, const float16& b) { return (float)a > (float)b; }
//
//  op_def bool  operator<=(const float16& a, const float16& b) { return (float)a <= (float)b; }
//
//  op_def bool  operator>=(const float16& a, const float16& b) { return (float)a >= (float)b; }
//
//  template <class T>
//  op_def float16 operator+(const float16& a, const T& b) { return float16((float)a + (float)b); }
//
//  template <class T>
//  op_def float16 operator-(const float16& a, const T& b) { return float16((float)a - (float)b); }
//
//  template <class T>
//  op_def float16 operator*(const float16& a, const T& b) { return float16((float)a * (float)b); }
//
//  template <class T>
//  op_def float16 operator/(const float16& a, const T& b) { return float16((float)a / (float)b); }
  

  op_def float16 /* constexpr */ operator+(const float16& h) { return h; }
  
  op_def float16 operator - (const float16& h) { return float16(hneg(h.data)); }
  
  op_def int isnan(const float16& h)  { return ishnan(h.data); }
  
  op_def int isinf(const float16& h) { return ishinf(h.data); }

  std::ostream& operator << (std::ostream& s, const float16&);

}   // namespace caffe

#endif
