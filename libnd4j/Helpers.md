## Helpers

### Requirements Helper

Requirements helper was introduced to replace plain checks for making them output informative messages (Debug and Verbose mode) and also replace macros REQUIRE_TRUE.

- it will lazily evaluate values and messages if the type wrapped and has` getValue` and `getMsg` methods
- it is implicit bool. this makes it usable with logical operators and also inside if conditions. Besides it will benefit from shortcircuit nature of those operators.
- it has the following check methods
```cpp
Requirements& expect(const T& expVar,const T1& reqVar, Op comparision, const char *first_half="")
Requirements& expectEq(const T& exp,const T1& req)
Requirements& expectNotEq(const T& exp,const T1& req)
Requirements& expectLess(const T& exp,const T1& req)
Requirements& expectLessEq(const T& exp,const T1& req)
Requirements& expectGreater(T exp, T1 req)
Requirements& expectGreaterEq(const T& exp,const T1& req
Requirements& expectTrue(const T& expVar, const char *msg=)
Requirements& expectFalse(const T& expVar, const char *msg=)
```
- you can either log the success case or throw error on the failure
- it can use plain types for checks. 
- if value has stream operator it will be used to output it's value. for custom types you may need add that by yourself
`ostream& operator<<(ostream& os, const CustomUserType& dt)`
- there is generic template `InfoVariable` wrapper for types to make it informative. you can use lambda operators with them as well to make it lazily evaluated
- we added custom `ShapeInfoVariable` wrapper for the NDArray and vector<> shapes to make them informative
- one can use `expect` to add its own proper comparision. simple lambda for that will be like this:
```cpp
[](const decltype(expType)& l, const decltype(reqType)& r){
            //compare and return
            return  ....;
        }
```

#### Examples:

firstly, we should enable logging
```cpp
    sd::Environment::getInstance().setDebug(true);
    sd::Environment::getInstance().setVerbose(true); 
```

1. simple case

```cpp    
Requirements req1("Requirement Helper Example#1");
int x = 20;
req1.expectLess(x, 22);
req1.expectEq(x, 21); //should fail
```
    
    
Output:
```  Requirement Helper Example#1:  {20} expected to be equal to  21```

2. using InfoVariable wrapper 
```cpp
int age = 15;
Requirements req2("Requirement Helper Example#2");
req2.expectGreaterEq(makeInfoVariable(age, "the user's age"), 18);
```
Output:
```
Requirement Helper Example#2: the user's age {15} expected to be greater than or equal  18
```

3. helper behavior while using many checks in one block
```cpp
int getAge(){
    std::cout<<"getAge() was called"<<std::endl;
    return 15;
}
....
Requirements req3("Requirement Helper Example#3");
int z = 20;
req3.expectEq(z, 21); 
req3.expectGreaterEq(makeInfoVariable(getAge(), "the user's age"), 18);
```
Output:
```
Requirement Helper Example#3:  {20} expected to be equal to  21
getAge() was called
```

As it is seen the second check did not happen as the previous failed. But still ```getAge()``` method was called as its function argument.

4. using **shortcircuit** to avoid Requirement call at all if the previous one was failed
```cpp
Requirements req4("Requirement Helper Example#4");
int zz = 20;
req4.expectEq(zz, 21) &&  //shortcicuit And
req4.expectGreaterEq(makeInfoVariable(getAge(), "the user's age"), 18);
```
Output:
```
Requirement Helper Example#4:  {20} expected to be equal to  21
```
5. using lambdas with InfoVariable. it will make it lazily evaluated 
```cpp
Requirements req5("Requirement Helper Example#5"); 
req5.expectEq( 21, 
            makeInfoVariable(21, []{
                   std::cout<<"lambda call#1"<<std::endl;
                   return "twenty one";
               }));
req5.expectEq(makeInfoVariable([]{ return 20;}, []{return "twenty";}), 
               makeInfoVariable(21, []{
                   std::cout<<"lambda call#2"<<std::endl;
                   return "twenty one";
               }));
req5.expectGreaterEq(makeInfoVariable([]{
                        std::cout<<"lambda call#3" <<std::endl;
                        return 15;
                        }, 
                        []{ return "the user's age";}), 
                     makeInfoVariable([]{return 18;}, []{return "the allowed age";})
      );
```
Output:
```
lambda call#2
Requirement Helper Example#5: twenty {20} expected to be equal to twenty one 21

```

6. use bool nature and also log the success case
```cpp
Requirements req6("Requirement Helper Example#6");
NDArray * arr= nullptr;
arr !=nullptr && req6.expectEq(arr->rankOf(), 3) ;
req6.logTheSuccess();
```
Output:
```
Requirement Helper Example#6: meets the requirements
```

7. custom comparision lambda and also another usage of the custom wrapper written by us ```ShapeInfoVariable```. Note: we will use ```std::vector<int>```. this wrapper can be used with ```NDArray``` as well.
```cpp
Requirements req7("Requirement Helper Example#7");
req7.expect(makeShapeInfoVariable(std::vector<sd::LongType>{2,3,4,5}, SHAPE_MSG_INPUT0), makeShapeInfoVariable(std::vector<sd::LongType>{2,3,4,7}, SHAPE_MSG_INPUT1),
                    [](const std::vector<sd::LongType>& l, const std::vector<sd::LongType>& r){
                        return l == r;
                    }
                 , EXPECTED_EQ_MSG);
}
```

Output:
```
Requirement Helper Example#7: the Shape of the Input NDArray#0 {[2, 3, 4, 5]} expected to be equal to the Shape of the Input NDArray#1 [2, 3, 4, 7]
```

8. throw error when there is failure
```cpp
Requirements req8("Requirement Helper Example#8");
req8.expectEq(6,6) &&
req8.expectIn(6, {1,2,3,7,8,9});
req8.throws();
```
Output:
```
terminate called after throwing an instance of 'std::invalid_argument'
  what():  Op validation failed
...
Requirement Helper Example#8: {6} expected to be one of these {[1, 2, 3, 7, 8, 9, ]}
```


##### Here is live example:
**Note:** some classes were mocked  there and do not represent the exact implementations in libnd4j. 
https://godbolt.org/z/sq98vchs5
