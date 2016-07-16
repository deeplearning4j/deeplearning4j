#ifndef CAFFE_UTIL_FP16_H_
#define CAFFE_UTIL_FP16_H_

#include <cfloat>
#include <iosfwd>
#include <fp16_emu.h>
#include <fp16_conversion.hpp>

#ifdef CPU_ONLY
  #define CAFFE_UTIL_HD
  #define CAFFE_UTIL_IHD inline
#else
  #define CAFFE_UTIL_HD __host__ __device__
  #define CAFFE_UTIL_IHD __inline__ __host__ __device__
#endif

namespace nd4j
{

  struct float16
  {
    /* constexpr */ CAFFE_UTIL_IHD float16() { data.x = 0; }
    
    template <class T>
    CAFFE_UTIL_IHD /*explicit*/ float16(const T& rhs) {
      assign(rhs);
    }
      
//    CAFFE_UTIL_IHD float16(float rhs) {
//      assign(rhs);
//    }
//
//    CAFFE_UTIL_IHD float16(double rhs) {
//      assign(rhs);
//    }

    CAFFE_UTIL_IHD operator float() const {
#ifdef __CUDA_ARCH__
      return __half2float(data);
#else
      return cpu_half2float(data);
#endif
    }

    //    CAFFE_UTIL_IHD operator double() const { return (float)*this; } 
       
    CAFFE_UTIL_IHD operator half() const { return data; }

    CAFFE_UTIL_IHD unsigned short getx() const { return data.x; }
    CAFFE_UTIL_IHD float16& setx(unsigned short x) { data.x = x; return *this; }

    template <class T>
    CAFFE_UTIL_IHD float16& operator=(const T& rhs) { assign(rhs); return *this; }

    CAFFE_UTIL_IHD void assign(unsigned int rhs) {
      // may be a better way ?
      assign((float)rhs);
    }

    CAFFE_UTIL_IHD void assign(int rhs) {
      // may be a better way ?
      assign((float)rhs);
    }

    CAFFE_UTIL_IHD void assign(double rhs) {
      assign((float)rhs);
    }
   
    CAFFE_UTIL_IHD void assign(float rhs) {
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

    CAFFE_UTIL_IHD void assign(const half& rhs) {
      data = rhs;
    }

    CAFFE_UTIL_IHD void assign(const float16& rhs) {
      data = rhs.data;
    }

    CAFFE_UTIL_IHD float16& operator+=(float16 rhs) { assign((float)*this + rhs); return *this; }  

    CAFFE_UTIL_IHD float16& operator-=(float16 rhs) { assign((float)*this - rhs); return *this; }  

    CAFFE_UTIL_IHD float16& operator*=(float16 rhs) { assign((float)*this * rhs); return *this; }   

    CAFFE_UTIL_IHD float16& operator/=(float16 rhs) { assign((float)*this / rhs); return *this; }  

    CAFFE_UTIL_IHD float16& operator+=(float rhs) { assign((float)*this + rhs); return *this; }  

    CAFFE_UTIL_IHD float16& operator-=(float rhs) { assign((float)*this - rhs); return *this; }  

    CAFFE_UTIL_IHD float16& operator*=(float rhs) { assign((float)*this * rhs); return *this; }  

    CAFFE_UTIL_IHD float16& operator/=(float rhs) { assign((float)*this / rhs); return *this; }  

    CAFFE_UTIL_IHD float16& operator++() { assign(*this + 1.f); return *this; }  

    CAFFE_UTIL_IHD float16& operator--() { assign(*this - 1.f); return *this; }  

    CAFFE_UTIL_IHD float16 operator++(int i) { assign(*this + (float)i); return *this; }  

    CAFFE_UTIL_IHD float16 operator--(int i) { assign(*this - (float)i); return *this; }  


    half data;

    // Utility contants
    static const float16 zero;
    static const float16 one;
    static const float16 minus_one;
  };

//  CAFFE_UTIL_IHD bool  operator==(const float16& a, const float16& b) { return ishequ(a.data, b.data); }
//
//  CAFFE_UTIL_IHD bool  operator!=(const float16& a, const float16& b) { return !(a == b); }
//
//  CAFFE_UTIL_IHD bool  operator<(const float16& a, const float16& b) { return (float)a < (float)b; }
//
//  CAFFE_UTIL_IHD bool  operator>(const float16& a, const float16& b) { return (float)a > (float)b; }
//
//  CAFFE_UTIL_IHD bool  operator<=(const float16& a, const float16& b) { return (float)a <= (float)b; }
//
//  CAFFE_UTIL_IHD bool  operator>=(const float16& a, const float16& b) { return (float)a >= (float)b; }
//
//  template <class T>
//  CAFFE_UTIL_IHD float16 operator+(const float16& a, const T& b) { return float16((float)a + (float)b); }
//
//  template <class T>
//  CAFFE_UTIL_IHD float16 operator-(const float16& a, const T& b) { return float16((float)a - (float)b); }
//
//  template <class T>
//  CAFFE_UTIL_IHD float16 operator*(const float16& a, const T& b) { return float16((float)a * (float)b); }
//
//  template <class T>
//  CAFFE_UTIL_IHD float16 operator/(const float16& a, const T& b) { return float16((float)a / (float)b); }
  

  CAFFE_UTIL_IHD float16 /* constexpr */ operator+(const float16& h) { return h; }
  
  CAFFE_UTIL_IHD float16 operator - (const float16& h) { return float16(hneg(h.data)); }
  
  CAFFE_UTIL_IHD int isnan(const float16& h)  { return ishnan(h.data); }
  
  CAFFE_UTIL_IHD int isinf(const float16& h) { return ishinf(h.data); }

  std::ostream& operator << (std::ostream& s, const float16&);

}   // namespace caffe

#endif
