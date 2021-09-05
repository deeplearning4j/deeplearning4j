/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/


#ifndef LIBND4J_REQUIREMENTSHELPER_H
#define LIBND4J_REQUIREMENTSHELPER_H

#include <type_traits>
#include <exception>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <helpers/logger.h>
#include <ConstMessages.h>
#include <iostream>
#include <helpers/ShapeUtils.h>

//internal usage only
#define ENABLE_LOG_TO_TEST 1
namespace sd {

inline std::ostream& operator<<(std::ostream& o, const sd::DataType &type)
{ 
    o << DataTypeUtils::asString(type);
    return o;
}

template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    o<<'[';
    for(auto &x : arr){
        o<<x<<", ";
    }
    o<<']';
    return o;
}

template <class T>
std::ostream& operator<<(std::ostream& o, const std::initializer_list<T>& arr)
{
    for(auto &x : arr){
        o<<x<<", ";
    }
    return o;
}

template<typename T>
using remove_cvref_t = typename std::remove_cv< typename std::remove_reference<T>::type >::type;

template <class F>
struct Check_callable
{
  template <class... Args>
  static std::false_type try_call(Args&&...);
 
  template <class... Args>
  static auto try_call( F&& f, Args&&... args)
    -> decltype((void)f(std::forward<Args>(args)...),
                std::true_type{});
};
 
template <class F, class... Args>
using is_callable = decltype
  (
    Check_callable<F>::try_call(std::declval<F>(),
          std::forward<Args>(std::declval<Args>())...)
  );

template<typename S, typename T>
class is_streamable
{
    template<typename SS, typename TT>
    static auto test(int)
        -> decltype(std::declval<SS&>() << std::declval<TT>(), std::true_type());

    template<typename, typename>
    static auto test(...)->std::false_type;

public:
    static const bool value = decltype(test<S, T>(0))::value;
};

template<typename T>
class has_getMsg
{
    template<typename TT>
    static auto test(int)
        -> decltype(std::declval<TT&>().getMsg(), std::true_type());

    template<typename>
    static auto test(...)->std::false_type;

public:
    static const bool value = decltype(test<T>(0))::value;
};

template<typename T>
class has_getValue
{
    template<typename TT>
    static auto test(int)
        -> decltype(std::declval<TT&>().getValue(), std::true_type());

    template<typename>
    static auto test(...)->std::false_type;

public:
    static const bool value = decltype(test<T>(0))::value;
};

template<typename T>
class has_getValueStr
{
    template<typename TT>
    static auto test(int)
        -> decltype(std::declval<TT&>().getValueStr(), std::true_type());

    template<typename>
    static auto test(...)->std::false_type;

public:
    static const bool value = decltype(test<T>(0))::value;
};

template<typename T, typename Enable=void>
struct Underline{
   using type = T; 
};
template<typename T >
struct Underline<T,  typename std::enable_if<has_getValue<T>::value>::type>{
   using type = remove_cvref_t<decltype(std::declval<T&>().getValue())>; 
};

template<typename T, typename Enable=void>
struct UnderlineMsg{
   using type = T; 
};

template<typename T >
struct UnderlineMsg<T,  typename std::enable_if<has_getMsg<T>::value>::type>{
   using type = decltype(std::declval<T&>().getMsg()); 
};

template<typename T>
using UnderlineType=typename Underline<T>::type;
template<typename T>
using UnderlineMsgType=typename UnderlineMsg<T>::type;

template<typename T>
typename std::enable_if<(!has_getMsg<T>::value ||  (has_getMsg<T>::value && !is_streamable<std::stringstream, UnderlineMsgType<T>>::value)), const char*>::type 
getMsg(T x){
    return "";
}

template<typename T>
typename std::enable_if<(has_getMsg<T>::value && is_streamable<std::stringstream, UnderlineMsgType<T>>::value) , UnderlineMsgType<T>>::type 
getMsg(const T& x){
    return x.getMsg();
}

template<typename T>
typename std::enable_if<!has_getValue<T>::value,T>::type
getValue(const T& x){
    return x;
}

template<typename T>
typename std::enable_if<has_getValue<T>::value, decltype(std::declval<T&>().getValue())>::type
getValue(const T& x){
    return x.getValue();
}

template<typename T>
typename std::enable_if<!has_getValueStr<T>::value && !std::is_same<remove_cvref_t<UnderlineType<T>>, bool>::value
 && is_streamable<std::stringstream,UnderlineType<T>>::value,UnderlineType<T>>::type
getStreamValue(const T& x){
    return getValue(x);
}

template<typename T>
typename std::enable_if<!has_getValueStr<T>::value && !is_streamable<std::stringstream,UnderlineType<T>>::value,std::string>::type
getStreamValue(T x){
    return "{//system can not stringify the variable}";
}

template<typename T>
typename std::enable_if<has_getValueStr<T>::value,std::string>::type
getStreamValue(const T& x){
    return x.getValueStr();
}

template<typename T>
typename std::enable_if<!has_getValueStr<T>::value && std::is_same<remove_cvref_t<UnderlineType<T>>, bool>::value, const char*>::type
getStreamValue(const T& x){
    return getValue(x) ? "True" : "False";
}

class Requirements{
    public:

    Requirements(const char *prefix_msg=""):prefix(prefix_msg), ok(true){
#if defined(ENABLE_LOG_TO_TEST)
        sd::Environment::getInstance().setDebug(true);
        sd::Environment::getInstance().setVerbose(true);
#endif
    };

    //Requirements is implicitly converted to bool.
    //Note: this way you can use shortcircuit operators && to chain requirements
    //      req.expect && req.expect && ... 
    //      This way you will get the shortcircuit that will not evaluate the laters
    //      it is better than using calls chaining like: req.expect( ).expect
    operator bool() {
         return this->ok;
    }

    //sets the prefix message
    Requirements& setPrefix(const char *prefix_msg){
        prefix = prefix_msg;
        return *this;
    }

    //Compares two values with comparision
    //Note: to achive full lazy evaluation of the obtained values or messages
    //        you should wrap them and use getValue getMsg functions
    template<typename T, typename T1, typename Op >
    Requirements& expect(const T& expVar,const T1& reqVar, Op comparision, const char *first_half=""){
        if(!this->ok)  return *this; 
        bool cmp  = comparision(getValue(expVar), getValue(reqVar));
        if(!cmp && sd::Environment::getInstance().isDebug() && sd::Environment::getInstance().isVerbose()){
            std::stringstream stream;
            stream<<prefix<<": "<<getMsg(expVar)<<" {"<<getStreamValue(expVar)<<"} "<<first_half<<' '<<getMsg(reqVar)<<' '<<getStreamValue(reqVar);
            sd::Logger::info("%s\n", stream.str().c_str());
        }
        this->ok  = this->ok && cmp;
        return *this;
    }

   template<typename T, typename T1>
   Requirements& expectEq(const T& exp,const T1& req){
        return expect(exp, req, std::equal_to<UnderlineType<T1>>{}, EXPECTED_EQ_MSG);
   }

   template<typename T, typename T1>
   Requirements& expectNotEq(const T& exp,const T1& req){
        return expect(exp, req, std::not_equal_to<UnderlineType<T1>>{}, EXPECTED_NE_MSG);
   }

   template<typename T, typename T1>
   Requirements& expectLess(const T& exp,const T1& req){
        return expect(exp, req, std::less<UnderlineType<T1>>{}, EXPECTED_LT_MSG);
   }

   template<typename T, typename T1>
   Requirements& expectLessEq(const T& exp,const T1& req){
        return expect(exp, req, std::less_equal<UnderlineType<T1>>{}, EXPECTED_LE_MSG);
   }

   template<typename T, typename T1>
   Requirements& expectGreater(T exp, T1 req){
        return expect(exp, req, std::greater<UnderlineType<T1>>{}, EXPECTED_GT_MSG);
   }

   template<typename T, typename T1>
   Requirements& expectGreaterEq(const T& exp,const T1& req){
        return expect(exp, req, std::greater_equal<UnderlineType<T1>>{}, EXPECTED_GE_MSG);
   }

    //throws. use this if you want to throw an exception if there is any failure
    void throws(){
        if(!this->ok) throw std::invalid_argument(OP_VALIDATION_FAIL_MSG);
    }

    template<typename T>
    Requirements& expectTrue(const T& expVar, const char *msg=EXPECTED_TRUE){
        if(!this->ok)  return *this; 
        bool cmp  = static_cast<bool>(getValue(expVar));
        if(!cmp && sd::Environment::getInstance().isDebug() && sd::Environment::getInstance().isVerbose()){
            std::stringstream stream;
            stream<<prefix<<": "<<getMsg(expVar)<<" {"<<getStreamValue(expVar)<<"} "<<msg;
            sd::Logger::info("%s\n", stream.str().c_str());
        }
        this->ok  = this->ok && cmp;
        return *this;
    }

    template<typename T>
    Requirements& expectFalse(const T& expVar, const char *msg=EXPECTED_FALSE){
        if(!this->ok)  return *this; 
        bool cmp  = static_cast<bool>(getValue(expVar));
        if(cmp && sd::Environment::getInstance().isDebug() && sd::Environment::getInstance().isVerbose()){
            std::stringstream stream;
            stream<<prefix<<": "<<getMsg(expVar)<<" {"<<getStreamValue(expVar)<<"} "<<msg;
            sd::Logger::info("%s\n", stream.str().c_str());
        }
        this->ok  = this->ok && !cmp;
        return *this;
    }

    template<typename T, typename T2>
    Requirements expectIn( const T& expVar, std::initializer_list<T2> vals){
        if(!ok)  return *this;
        auto val=getValue(expVar);
        for(const auto& r_i : vals){
            if(val==r_i) return *this;
        }
        
        if(sd::Environment::getInstance().isDebug() && sd::Environment::getInstance().isVerbose()){
            std::stringstream stream;
            stream<<prefix<<": "<<getMsg(expVar)<<'{'<<getStreamValue(expVar)<<"} "<<EXPECTED_IN<<" {"<<getStreamValue(vals)<<"}\n";
            sd::Logger::info("%s", stream.str().c_str());
        }
        ok = false;
        return *this;
    }


    //Logs the Successfull cases to know if the requiremets met the conditions
    void logTheSuccess(){
        if(this->ok && sd::Environment::getInstance().isDebug() && sd::Environment::getInstance().isVerbose()){
            sd::Logger::info("%s: %s\n", prefix, REQUIREMENTS_MEETS_MSG);
        }
    }

    private:
    //the prefix used to add messages in each log
    const char *prefix;
    //bool value used for the Requirements to mimick bool
    bool ok;
};

//Generic wrapper where variable and info can be lazy evaluated using the lambda
template<typename T1, typename T2>
struct InfoVariable{ 

    T1 value_or_op;
    T2 message_or_op;

    InfoVariable(T1 valOrOp, T2 msgOrOp):value_or_op(valOrOp), message_or_op(msgOrOp){ 
    }

    template<typename U=T1, typename std::enable_if<!is_callable<U>::value, int>::type = 0>
    U getValue() const{
        return value_or_op;
    }

    template<typename U=T1, typename std::enable_if<is_callable<U>::value, int>::type = 0>
    auto getValue() const -> decltype(std::declval<U&>()()){
        return value_or_op();
    }

    template<typename U=T2,  typename std::enable_if<!is_callable<U>::value, int>::type = 0>
    U getMsg() const{
        return message_or_op;
    }

    template<typename U=T2, typename std::enable_if<is_callable<U>::value, int>::type = 0>
    auto getMsg() const -> decltype(std::declval<U&>()()) {
        return message_or_op();
    }
 

};

template<typename T1, typename T2>
InfoVariable<T1, T2> makeInfoVariable( T1&& v1,  T2&& v2){
    return InfoVariable<T1, T2>(std::forward<T1>(v1), std::forward<T2>(v2));
}

template<typename T>
struct ShapeInfoVariable{ 

    explicit ShapeInfoVariable(const T& val, const char *msg=""):value(val), message(msg){}  
    const T& value;
    const char* message; 

    const char* getMsg() const{
        return message;
    }
  
    const T& getValue() const { return value;}

    std::string getValueStr() const { return ShapeUtils::shapeAsString(value); }

};

template<typename T>
ShapeInfoVariable<T> makeShapeInfoVariable( T&& v, const char *msg){
    return ShapeInfoVariable<T>(std::forward<T>(v), msg);
}

}
#endif //LIBND4J_REQUIREMENTSHELPER_H
