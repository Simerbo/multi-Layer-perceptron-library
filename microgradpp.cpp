#include<iostream>
#include<cmath>
#include<memory>
#include<functional>
#include<string>

using ValuePtr = std::shared_ptr<Value>; // adding shared ptr 

class Value : public std::shared_from_this<Value>{

	private:
		float data;
		float grad;
		std::string op; // this is the primitive operation that is used to run on the primitive binaries
		size_t id;	
		std::vector<Value> prev;
	
	private:
		Value(float data, const std::string &op) : data(data), op(op) {};

	public ValuePtr create(float data, const std::string &op ="")
	{
		return std::shared_ptr<Value>(new Value(data,op)); // this calls the construct value - we need to expose an API for the client to construct
	}
};

int main(){
	
	Value a;
	Value b;
	auto c = add(a+b);

	//  the variables a and b are the two operants that are used in a binary primitive operation of +,-,/,*
	
	// a -->
	//	  + ---> c for bp we need to know the child of a + b 
	// b -->
	//derivative of dl/da

	auto a = std::shared_ptr<Value>();

}
