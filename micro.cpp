
#include<iostream>
#include<cmath>
#include <locale>
#include<memory>
#include<functional>
#include<string>
#include<vector>
#include<cassert>
#include<unordered_set>
#include <unordered_map>
#include <random>
#include <stdexcept>
#include <algorithm>


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


	static ValuePtr pow(const ValuePtr base, float exponent){
		float newValue = std::pow(base->data, exponent);

		auto out = Value::create(newValue, "^");

		out->prev = {base};

		out->backward = [
			base_weak = std::weak_ptr<Value>(base),
			out_weak = std::weak_ptr<Value>(out),
			exponent
		](){

			auto base_ptr = base_weak.lock();
			auto out_ptr = out_weak.lock();

			if(base_ptr && out_ptr){
				base_ptr->grad +=
					exponent *
					std::pow(base_ptr->data, exponent - 1) *
					out_ptr->grad;
			}
		};

		return out;
	}


	static ValuePtr divide(const ValuePtr& lhs, const ValuePtr& rhs){
		// return of the multiplication of a and b

		auto reciprocal = pow(rhs, -1);

		return multiply(lhs, reciprocal);
	}


	static ValuePtr relu(const ValuePtr& input){
		float val = std::max(0.0f,input->data);

		auto out = Value::create(val,"ReLU");

		out->prev={input};

		out->backward = [input_weak = std::weak_ptr<Value>(input),
		out_weak = std::weak_ptr<Value>(out)](){

			auto input_ptr = input_weak.lock();
			auto out_ptr = out_weak.lock();

			if(input_ptr && out_ptr){
				input_ptr->grad +=
					(input_ptr->data > 0.0f ? 1.0f : 0.0f) *
					out_ptr->grad;
			}
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


	static ValuePtr sigmoid(const ValuePtr& input) {
		float x = input->data;
		float t = std::exp(x) / (1 + std::exp(x));

		auto out = Value::create(t, "Sigmoid");
		out->prev = {input};

		out->backward = [
			input_weak = std::weak_ptr<Value>(input),
			out_weak = std::weak_ptr<Value>(out),
			t
		]() {
			auto input_ptr = input_weak.lock();
			auto out_ptr = out_weak.lock();

			if(input_ptr && out_ptr){
				input_ptr->grad += t * (1 - t) * out_ptr->grad;
			}
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

		buildTopo(shared_from_this(), visited, topo); // this will create a graph for us

		this->grad = 1.0f;

		for(auto it = topo.rbegin(); it != topo.rend(); ++it ){
			if((*it)->backward){
				(*it)->backward();
			}

			(*it)->print();
		}
	}


	void print(){
		std::cout<<"[data=" <<this->data << ", grad="<<this->grad<<"]\n";
	}

};


size_t Hash::operator()(const ValuePtr value) const{
	return std::hash<std::string>()(value->op) ^
	std::hash<float>()(value->data);
}


// Part 4

enum ActivationType{
	RELU,
	SIGMOID
};

class Activation{
	static std::shared_ptr<Value> Relu(const std::shared_ptr<Value>& val){
		return Value::relu(val);
	}

	static std::shared_ptr<Value> Sigmoid(const std::shared_ptr<Value>& val){
		return Value::sigmoid(val);
	}

public:
	static inline std::unordered_map<ActivationType, std::function<std::shared_ptr<Value>(std::shared_ptr<Value>&)>> mActivationFcn = {
			{ActivationType::RELU, Relu},
			{ActivationType::SIGMOID, Sigmoid}
	};
};













// Function to generate a random float between -1 and 1
float getRandomFloat() {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_real_distribution<> dis(-1, 1);
	return dis(gen);
}

class Neuron {
private:
	std::vector<ValuePtr> weights;
	ValuePtr bias = Value::create(0.0);
	const ActivationType activation_t;

public:
	Neuron(size_t nin, const ActivationType& activation_t) : activation_t(activation_t) {
		for (size_t idx = 0; idx < nin; ++idx) {
			weights.emplace_back(Value::create(getRandomFloat()));
		}
	}

//    // For testing
//    Neuron(size_t nin, float val, const ActivationType& activation_t = ActivationType::SIGMOID)
//            : activation_t(activation_t) {
//        for (size_t idx = 0; idx < nin; ++idx) {
//            weights.emplace_back(Value::create(getRandomFloat()));
//        }
//    }

	void zeroGrad() {
		for (auto& weight : weights) {
			weight->grad = 0;
		}
		bias->grad = 0;
	}

	// Dot product of a Neuron's weights with the input
	ValuePtr operator()(const std::vector<ValuePtr>& x) {
		if (x.size() != weights.size()) {
			throw std::invalid_argument("Vectors must be of the same length");
		}

		ValuePtr sum = Value::create(0.0);

		for (size_t idx = 0; idx < weights.size(); ++idx) {

			ValuePtr intermediateVal = Value::multiply(x[idx], weights[idx]);
			sum = Value::add(sum, intermediateVal);
		}

		// Add bias
		//sum->add_inplace(bias);
		sum = Value::add(sum, bias);

		// Apply activation function
		const auto& activationFcn = Activation::mActivationFcn.at(activation_t);
		return activationFcn(sum);
	}

	std::vector<ValuePtr> parameters() const {
		std::vector<ValuePtr> out;
		out.reserve(weights.size() + 1);

		out.insert(out.end(), weights.begin(), weights.end());
		out.push_back(bias);

		return out;
	}

	void printParameters() const {
		printf("Number of Parameters: %zu\n", weights.size() + 1);
		for (const auto& param : weights) {
			printf("%f, %f\n", param->data, param->grad);
		}
		printf("%f, %f\n", bias->data, bias->grad);
		printf("\n");
	}

	size_t getParametersSize() const {
		return weights.size() + 1;
	}
};




















int main()
{
	auto a = Value::create(1.0, "");
	auto b = Value::create(2.0, "");
	//Value b;
	auto c = Value::add(a, b); // 3
	
	auto d = Value::multiply(c,c); // 9
	
	assert(c->data == 3.0);
	assert(c->op == "+");
	
	assert(d->data == 9.0);
	assert(d->op == "*");
	
	auto loss = Value::add(d, d); // 9+9 = 18
	
	loss->backProp();
	
	
	
	//  the variables a and b are the two operants that are used in a binary primitive operation of +,-,/,*

	// a -->
	//	  + ---> c for bp we need to know the child of a + b
	// b -->
	//derivative of dl/da
	//      dl/da =  dc/dl

	//	auto b = std::shared_ptr<Value>();


	// auto d = subtract(a-b);

	// auto e = multiply(c*d);

	// a -->
	//        +    -->   c ->           L   ->  [a,b,c,d,L]
	
	// b -->             +
                        
                        // d
	
	//auto a = std::shared_ptr<Value>();
}
