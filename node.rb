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

class Node

	attr_reader :type, :id, :action

	def initialize(id, type = :sigmoid, bias = 0, action = :+ )
		@id = id
		@type = type
		@bias = bias
		@action = action
	end

	def value sum = nil
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
	def sigmoid(x)
		e = Math::E
		1.0 / (1 + e**(-x) )
	end

	def tanh(x)
		Math.tanh(x)
	end

	def sinh(x)
		Math.sinh(x)
	end

	def threshold(x)
		x > 0 ? 1.0 : 0.0
	end

	def rectified_liniar_unit(x)
		x > 0 ? x : 0.0
	end

	def leaky_rectified_liniar_unit(x)
		x > 0 ? x : (x / 5.0)
	end

	def none(x)
		x
	end

	def radial_basis(x)
		Math::E ** ( - (x**2) )		
	end



	def mutate!(rate, severity, heavy_mutation_cap)

		if mutate?(heavy_mutation_cap)
			hidden_nodes = HIDDEN_NODE_TYPES
			if hidden_nodes.include? @type
				@type = hidden_nodes.shuffle.first
			end			
		end

		#mutate what the node does with the synapses
		if mutate? (heavy_mutation_cap)
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

	def clone
		new_node = Node.new(@id, @type, @bias, @action)
	end

	def mutate? (rate)
		rand() < rate
	end

	def mutation_offset x, severity
		min = (x-severity)
		max = (x+severity)
		rand() * (max - min) + min
	end

end
