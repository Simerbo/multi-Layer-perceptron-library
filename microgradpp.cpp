#include<iostream>
#include<cmath>
#include <locale>
#include<memory>
#include<functional>
#include<string>
#include<vector>
#include<cassert>
#include<unordered_set>


class Value;

using ValuePtr = std::shared_ptr<Value>; // adding shared ptr

// hash function
struct Hash{
	size_t operator()(const ValuePtr value) const;
};

class Value : public std::enable_shared_from_this<Value>{

public:
	inline static size_t currentID = 0; // current id would start from zero and when we add a value or remove one, it will load or remove one.
	float data;
	float grad;
	std::string op; // this is the primitive operation that is used to run on the primitive binaries
	size_t id;
	std::vector<ValuePtr> prev;

	std::function<void()> backward;

private:
	Value(float data, const std::string &op, size_t id)
	: data(data), grad(0.0f), op(op), id(id) {};

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

		// in order to calculate the derivative or da in respect to dc we need to create a backward function which used to update the gradient of a and b
		out->backward = [lhs_weak = std::weak_ptr<Value>(lhs),
		rhs_weak = std::weak_ptr<Value>(rhs),
		out_weak = std::weak_ptr<Value>(out)](){

			auto lhs_ptr = lhs_weak.lock();
			auto rhs_ptr = rhs_weak.lock();
			auto out_ptr = out_weak.lock();

			lhs_ptr->grad += out_ptr->grad;
			rhs_ptr->grad += out_ptr->grad;
		};

		return out;
	}

	// this function for Forward Propagation
	static ValuePtr multiply(const ValuePtr& lhs, const ValuePtr& rhs){
		// return of the multiplication of a and b
		auto out = Value::create(lhs->data*rhs->data, "*");

		// saving the child value pointer (lhs,rhs)
		out->prev = {lhs, rhs};

		out->backward = [lhs_weak = std::weak_ptr<Value>(lhs),
		rhs_weak = std::weak_ptr<Value>(rhs),
		out_weak = std::weak_ptr<Value>(out)](){

			auto lhs_ptr = lhs_weak.lock();
			auto rhs_ptr = rhs_weak.lock();
			auto out_ptr = out_weak.lock();

			lhs_ptr->grad += rhs_ptr->data * out_ptr->grad;
			rhs_ptr->grad += lhs_ptr->data * out_ptr->grad;
		};

		return out;
	}

	// this function for Forward Propagation
	static ValuePtr subtract(const ValuePtr& lhs, const ValuePtr& rhs){
		// return of the subtraction of a and b
		auto out = Value::create(lhs->data-rhs->data, "-");

		// saving the child value pointer (lhs,rhs)
		out->prev = {lhs, rhs};

		out->backward = [lhs_weak = std::weak_ptr<Value>(lhs),
		rhs_weak = std::weak_ptr<Value>(rhs),
		out_weak = std::weak_ptr<Value>(out)](){

			auto lhs_ptr = lhs_weak.lock();
			auto rhs_ptr = rhs_weak.lock();
			auto out_ptr = out_weak.lock();

			lhs_ptr->grad += out_ptr->grad;
			rhs_ptr->grad -= out_ptr->grad;
		};

		return out;
	}


	void buildTopo(
		std::shared_ptr<Value> v,
		std::unordered_set<std::shared_ptr<Value>, Hash>& visited,
		std::vector<std::shared_ptr<Value>>& topo)
	{
		if(visited.find(v) == visited.end()){
			visited.insert(v);

			for(const auto& child : v->prev){
				buildTopo(child, visited, topo);
			}

			topo.push_back(v);
		}
	}

	void backProp(){

		std::vector<std::shared_ptr<Value>> topo;

		std::unordered_set<std::shared_ptr<Value>, Hash> visited;

		buildTopo(shared_from_this(), visited, topo);

		this->grad = 1.0f;

		for(auto it = topo.rbegin(); it != topo.rend(); ++it ){
			if((*it)->backward){
				(*it)->backward();
			}
		}

		for(auto it = topo.begin(); it != topo.end(); ++it ){
			(*it)->print();
		}
	}


	void print(){
		std::cout<<"[data=" <<this->data << ", grad="<<this->grad <<"]\n";
	}

};

size_t Hash::operator()(const ValuePtr value) const{
	return std::hash<std::string>()(value->op) ^
	std::hash<float>()(value->data);
}


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


	auto loss = Value::add(d,d);
	loss->backProp();



	//  the variables a and b are the two operants that are used in a binary primitive operation of +,-,/,*

	// a -->
	//	  + ---> c for bp we need to know the child of a + b
	// b -->
	//derivative of dl/da
	//      dl/da =  dc/dl

	//	auto b = std::shared_ptr<Value>();

}
