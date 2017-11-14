//
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <types/uint16.h>

namespace nd4j {
    float cpu_uint162float(int16_t data) {
        return (float) ((int) data);
    }
    
    uint8_t cpu_float2uint16(float data) {
        int t = (int) data;
        if (t > 65536 ) t = 65536;
        if (t < 0) t = 0;
    
        return (uint16_t) t;
    }

    uint16::uint16() { 
        data = cpu_float2uint16(0.0f); 
    }

    uint16::~uint16() { 
        //
    }

    template <class T>
    uint16::uint16(const T& rhs) {
        assign(rhs);
    }

    template <class T>
    uint16& uint16::operator=(const T& rhs) { 
        assign(rhs); 
        return *this; 
    }

    uint16::operator float() const {
        return cpu_uint162float(data);
    }

    void uint16::assign(float rhs) {
        data = cpu_float2uint16(rhs);
    }

    void uint16::assign(double rhs) {
        assign((float)rhs);
    }

    template uint16::uint16(const float& rhs);
    template uint16::uint16(const double& rhs);

    template uint16& uint16::operator=<double>(const double& rhs);
    template uint16& uint16::operator=<float>(const float& rhs);
}