# HIDDEN_NODE_TYPES = [:sigmoid, :flat, :tanh, :positive, :none] #, :delay, :average] #TODO: disabled stacked param node types

HIDDEN_NODE_TYPES = [
	:sigmoid, 
	:threshold,
	:tanh, 
	# :sinh,
	:none,
	:rectified_liniar_unit,
	:leaky_rectified_liniar_unit,
	# :softplus,
	:radial_basis
]

INPUT_NODE_TYPES  = [:input] #, :fuzzy_input, :flat_random_input] #TODO - add genetic controller options to enable fuzzy inputs
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
		when :sigmoid #params show how rough the curve is
			sigmoid(x)
		when :threshold #no params are used. Bias allows to move the merge point left and right
			threshold(x)
		when :tanh #same as sigmoid
			tanh(x)
		when :sinh #same as sigmoid
			sinh(x)
		when :none #take no action
			none(x)
		when :rectified_liniar_unit
			rectified_liniar_unit(x)
		when :leaky_rectified_liniar_unit
			leaky_rectified_liniar_unit(x)
		when :softplus
			softplus(x)
		when :radial_basis
			radial_basis(x)

		# #memory / delay circuts
		# when :delay #params show how deep the rabbit hole goes
		# 	a = @params[:a]
		# 	a.push(x).shift(1)
		# when :average #params show how deep the rabbit hole goes
		# 	a = @params[:a]
		# 	a.push(x).shift(1)

		# 	a.reduce(:+) / a.size

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

	# def softplus(x)
	# 	Math.log(x)
	# end

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
			if mutate? ( rate )
				action_type = rand(2)
				case action_type
				when 0
					@action = :+
				when 1
					@action = :*
				end				
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
		# rand ( (x-severity)..(x+severity)  )
		rand() * (max - min) + min
	end

end
