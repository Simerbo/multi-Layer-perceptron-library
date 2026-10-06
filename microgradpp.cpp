#include<iostream>
#include<cmath>
#include<memory>
#include<functional>
#include<string>

using namespace std;
using ValuePtr = std::shared_ptr<Value>; // adding shared ptr 

class Value : public std::shared_from_this<Value>{

	private:
		float data;
		float grad;
		string op; // this is the primitive operation that is used to run on the primitive binaries
		size_t id;	


}





int main(){


}

