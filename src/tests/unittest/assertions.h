/*
 * assertions.h
 *
 *  Created on: Dec 29, 2015
 *      Author: agibsonccc
 */

#ifndef ASSERTIONS_H_
#define ASSERTIONS_H_


#pragma once


#include "exceptions.h"
#include "util.h"

#define ASSERT_EQUAL_QUIET(X,Y)  unittest::assert_equal_quiet((X),(Y), __FILE__, __LINE__)
#define ASSERT_EQUAL(X,Y)        unittest::assert_equal((X),(Y), __FILE__,  __LINE__)
#define ASSERT_LEQUAL(X,Y)       unittest::assert_lequal((X),(Y), __FILE__,  __LINE__)
#define ASSERT_GEQUAL(X,Y)       unittest::assert_gequal((X),(Y), __FILE__,  __LINE__)
#define ASSERT_ALMOST_EQUAL(X,Y) unittest::assert_almost_equal((X),(Y), __FILE__, __LINE__)
#define KNOWN_FAILURE            { unittest::UnitTestKnownFailure f; f << "[" << __FILE__ ":" << __LINE__ << "]"; throw f;}

#define ASSERT_EQUAL_RANGES(X,Y,Z)  unittest::assert_equal((X),(Y),(Z), __FILE__,  __LINE__)

#define ASSERT_THROWS(X,Y)                                                         \
    {   bool thrown = false; try { X; } catch (Y) { thrown = true; }                  \
        if (!thrown) { unittest::UnitTestFailure f; f << "[" << __FILE__ << ":" << __LINE__ << "] did not throw " << #Y; throw f; } \
    }


namespace unittest
{

static size_t MAX_OUTPUT_LINES = 10;

static double DEFAULT_RELATIVE_TOL = 1e-4;
static double DEFAULT_ABSOLUTE_TOL = 1e-4;




// sometimes it's not possible to << a type
template <typename T1, typename T2>
void assert_equal_quiet(const T1& a, const T2& b,
                        const std::string& filename = "unknown", int lineno = -1)
{
    if(!(a == b)){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << "values are not equal.";
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}

template <typename T1, typename T2>
void assert_lequal(const T1& a, const T2& b,
                   const std::string& filename = "unknown", int lineno = -1)
{
    if(!(a <= b)){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << a << " is greater than " << b;
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}

template <typename T1, typename T2>
void assert_gequal(const T1& a, const T2& b,
                   const std::string& filename = "unknown", int lineno = -1)
{
    if(!(a >= T1(b))){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << a << " is less than " << b;
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}

// define our own abs() because std::abs() isn't portable for all types for some reason
template<typename T>
  T abs(const T &x)
{
  return x > 0 ? x : -x;
}


inline
bool almost_equal(const double& a, const double& b, const double& a_tol, const double& r_tol)
{
    if(abs(a - b) > r_tol * (abs(a) + abs(b)) + a_tol)
        return false;
    else
        return true;
}

template <typename T1, typename T2>
void assert_almost_equal(const T1& a, const T2& b,
                         const std::string& filename = "unknown", int lineno = -1,
                         double a_tol = DEFAULT_ABSOLUTE_TOL, double r_tol = DEFAULT_RELATIVE_TOL)

{
    if(!almost_equal(a, b, a_tol, r_tol)){
        unittest::UnitTestFailure f;
        f << "[" << filename << ":" << lineno << "] ";
        f << "values are not approximately equal: " << (double) a << " " << (double) b;
        f << " [type='" << type_name<T1>() << "']";
        throw f;
    }
}




template <typename T>
class almost_equal_to
{
    public:
        double a_tol, r_tol;
        almost_equal_to(double _a_tol = DEFAULT_ABSOLUTE_TOL, double _r_tol = DEFAULT_RELATIVE_TOL) : a_tol(_a_tol), r_tol(_r_tol) {}
        bool operator()(const T& a, const T& b) const {
            return almost_equal((double) a, (double) b, a_tol, r_tol);
        }
};


}


#endif /* ASSERTIONS_H_ */
