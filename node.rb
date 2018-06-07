# Constants:
# HIDDEN_NODE_TYPES - an array of node types that hidden nodes can be. Mutation will allow the node types to change inside this array
# INPUT_NODE_TYPES - a single element array of input node types. In function this type is equivelent to :none.
# OUTPUT_NODE_TYPES - a single element array of output node types. In function this type is equivelent to :none. Neural network classes return arrays of output node results

HIDDEN_NODE_TYPES = [
	:sigmoid, 
	:threshold,
	:tanh, 
	:none,
	:rectified_liniar_unit,
	:leaky_rectified_liniar_unit,
	:radial_basis
]

INPUT_NODE_TYPES  = [:input]
OUTPUT_NODE_TYPES = [:output]

# Node class wraps functions for initialization, activation, mutation and cloning
# 
# == READERS: 
# type, id, action
#
# == FUNCTIONS
# 
# initialize (id, type = :sigmoid, bias = 0, action = :+ ) - creates a node
# value (sum)
# activate (x)
# sigmoid(x)
# tanh(x)
# sinh(x)
# threshold(x)
# rectified_liniar_unit(x)
# leaky_rectified_liniar_unit(x)
# none(x)
# mutate!
# clone
# mutate?
# mutation_offset

class Node

	#Node readable values:
	attr_reader :type, :id, :action

	#Creates a new node
	# === Attributes
	# * +id+ - used to keep track of nodes, not used in node operation
	# * +type+ - the activation function to use for return values
	# * +bias+ - a permanent node input in addition to the node inputs of the value function
	# * +action+ - the base operation the node uses (addition or multiplication)
	def initialize(id, type = :sigmoid, bias = 0, action = :+ )
		@id = id
		@type = type
		@bias = bias
		@action = action
	end

	#Returns the activation value after applying bias
	# === Paramater
	# * +sum+ = +nil+ - the sum or product of all the node inputs. They are added or multiplied with the bias depending on action type
	def value(sum = nil)
		if sum.nil?
			sum = @bias
		else
			case @action
			when :+
				sum += @bias
			when :*
				sum *= @bias
			end
		end

		activate sum
	end

	#Returns the result of applying the activation function according to node +type+
	# === Paramater
	# *+x+ - the x value of the function input
	def activate x
		case @type

		#perceptron types
		when :sigmoid
			sigmoid(x)
		when :threshold
			threshold(x)
		when :tanh
			tanh(x)
		when :none
			none(x)
		when :rectified_liniar_unit
			rectified_liniar_unit(x)
		when :leaky_rectified_liniar_unit
			leaky_rectified_liniar_unit(x)
		when :radial_basis
			radial_basis(x)

		#I/O
		when :input #no params
			none(x)
		when :output #no params			
			none(x)
		else
			raise "unknown node type: #{@type}"
		end
	end

	#ACTIVATION FUNCTIONS

	#Returns the result of applying a sigmoid function
	def sigmoid(x)
		e = Math::E
		1.0 / (1 + e**(-x) )
	end

	#Applies the tanh function
	def tanh(x)
		Math.tanh(x)
	end

	#returns 1 if x is positive, 0 otherwise. Works in tangent with node bias
	def threshold(x)
		x > 0 ? 1.0 : 0.0
	end

	#ReLu - returns X if it is positive, 0 if not
	def rectified_liniar_unit(x)
		x > 0 ? x : 0.0
	end

	#Divides X by 5 if it's negative, returns X if it's positive
	def leaky_rectified_liniar_unit(x)
		x > 0 ? x : (x / 5.0)
	end

	#Returns X
	def none(x)
		x
	end

	#Returns the Y value of a bell curve function. Maxes out at 1 if X is 0
	def radial_basis(x)
		Math::E ** ( - (x**2) )		
	end


	#Mutates the node in place
	#
	# === Paramaters
	# *+rate+ - base chance check - used for bias change
	# *+severity+ - change rate for bias
	# *+heavy_mutation_cap+ - mutation limit after which node can change action and type
	def mutate!(rate, severity, heavy_mutation_cap)

		if rate > heavy_mutation_cap and mutate?(rate)
			hidden_nodes = HIDDEN_NODE_TYPES
			if hidden_nodes.include? @type
				@type = hidden_nodes.shuffle.first
			end			
		end

		#mutate what the node does with the synapses
		if rate > heavy_mutation_cap and mutate?(rate)
			action_type = rand(2)
			case action_type
			when 0
				@action = :+
			when 1
				@action = :*
			end				
		end

		@bias = mutation_offset(@bias, severity) if mutate?(rate)

		return self
	end

	#Returns a new node with identical paramaters
	def clone
		new_node = Node.new(@id, @type, @bias, @action)
	end

	#Checks if mutation should occur
	#
	# === Params
	#
	# *+rate+ - value from 0 to 1 for mutation chance
	def mutate? (rate)
		rand() < rate
	end

	#Returns a new value near the base value shifted in either direction
	#
	# === Params
	#
	# *+x+ - base value, new value will be close to this param
	# *+severity+ - max distance new value can be placed away form base value
	def mutation_offset x, severity
		min = (x-severity)
		max = (x+severity)
		rand() * (max - min) + min
	end
end