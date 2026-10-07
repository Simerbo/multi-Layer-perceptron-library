#include<iostream>
#include<cmath>
#include<memory>
#include<functional>
#include<string>
#include<vector>
#include<cassert>

class Value;

using ValuePtr = std::shared_ptr<Value>; // adding shared ptr

class Value : public std::enable_shared_from_this<Value>{


public:
	inline static size_t currentID = 0; // current id would start from zero and when we add a value or remove one, it will load or remove one.
	float data;
	float grad;
	std::string op; // this is the primitive operation that is used to run on the primitive binaries
	size_t id;
	std::vector<ValuePtr> prev;

private:
	Value(float data, const std::string &op, size_t id)
	: data(data), op(op), id(id) {};

public:
	static ValuePtr create(float data, const std::string &op ="") // this function is used for the client to call by passing their data and an operator for their binary operation
	{
		return std::shared_ptr<Value>(new Value(data,op, Value::currentID++)); // this calls the construct value - we need to expose an API for the client to construct
	}

	~Value(){
		--Value::currentID;
		// decrement currentID because it is being used to generate unique IDs.
	}
	
	// this function for Forward Propagation
	static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs){
		// return of the addition of a and b
		auto out = Value::create(lhs->data+rhs->data, "+");
		// saving the child value pointer (lhs,rhs)
		out->prev = {lhs, rhs};
	
		return out;	
	}	

	// this function for Forward Propagation
	static ValuePtr multiply(const ValuePtr& lhs, const ValuePtr& rhs){
		// return of the addition of a and b
		auto out = Value::create(lhs->data*rhs->data, "*");
		// saving the child value pointer (lhs,rhs)
		out->prev = {lhs, rhs};
		
		return out;		
	}	
	
	// this function for Forward Propagation
	static ValuePtr subtract(const ValuePtr& lhs, const ValuePtr& rhs){
		// return of the addition of a and b
		auto out = Value::create(lhs->data-rhs->data, "-");
		// saving the child value pointer (lhs,rhs)
		out->prev = {lhs, rhs};
		
		return out;		
	}	
	
	void print(){
		std::cout<<"[data=" <<this->data << ", grad="<<this->grad <<"]\n";
	}

};

int main(){
	auto a = Value::create(1.0,"+");
	auto b = Value::create(2.0,"+");
	//auto b;
	auto c = Value::add(a,b);
	auto d = Value::multiply(c,c);


	assert(c->data ==3.0);
	assert(c->op == "+");

	assert(d->data ==9.0);
	assert(d->op == "*");


	//  the variables a and b are the two operants that are used in a binary primitive operation of +,-,/,*

	// a -->
	//	  + ---> c for bp we need to know the child of a + b
	// b -->
	//derivative of dl/da

//	auto b = std::shared_ptr<Value>();

}
