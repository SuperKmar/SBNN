require_relative "node"

#NeuralNetwork class encapsulates all methods needed to run a feed foreward neural network, including input-out mechanics, cloning and mutation
#
# == Readers:
#
# id, mutation_rate, severity, heavy_mutation_rate
#
# == Accessors:
#
# nodes, synapses
#
# == Functions:
# 
# initialize
# set_inputs
# get_outputs
# get_cache
# set_cache
# clear_cache
# mutate
# mutate?
# mutation_offset
class NeuralNetwork
	
	attr_accessor :nodes, :synapses
	attr_reader :id, :mutation_rate, :severity, :heavy_mutation_rate

	#Creates a new neural network
	# === Attributes
	# *+id+ - id for tracking the neural network. Uses a random number if nil/ommited
	# *+nodes+ - array of nodes (see Node class)
	# *+synapses+ - array of synapses
	# *+mutation_rate+ - base chance for adjusting values when calling the mutate function, this value mutates as well
	# *+severity+ - amount of adjustment to be done when mutating, this value mutates as well
	# *+heavy_mutation_rate - base chance for adding and deleting nodes and synapses at random
	# === Synapses
	# a synapse is represented as a hash with the following keys:
	# *+from+ - node id from which a value is extracted to be used as input
	# *+to+ - node id for which etracted value is used as input
	# *+weight+ - value multiplier for adjusting importance of a given input, value is any float value
	def initialize id = nil, nodes = nil, synapses = nil, mutation_rate = 0.01, severity = 0.1, heavy_mutation_rate = 0.1
		if id.nil? 
			@id = rand(0..1000000)
		else
			@id = id #really not sure why i need id
		end

		#node: { id. value = nil, a, type}
		#synapses: { from: id, to: id, weight}
		
		# @nodes = nodes #old nodes creation		
		@nodes = []
		nodes.each do |node|
			@nodes << node.clone
		end		

		@synapses = []
		synapses.each do |synapse|
			@synapses << {from: synapse[:from], to: synapse[:to], weight: synapse[:weight]} #
		end

		#mutation rate and severity is automatic
		#heavy mutation cap determins when adding / removing nodes and synapsis can happen in mutation
		@mutation_rate = mutation_rate
		@severity = severity
		@heavy_mutation_rate = heavy_mutation_rate

		@e = Math::E
		@cache = Hash.new(nil)

		return self
	end

	#network-wide node operations

	#Sets neural network input. Must be called before attempting to retreve an output
	#
	# === Paramaters
	# 
	# *+inputs+ - an array of floats representing the starting values of the input nodes. Array length must be identical to neural network input node count. Input node ID corresponds to array element index, which is why input nodes must have continuous ID's starting from 0.
	def set_inputs inputs = []
		#sets the input node values to the ones in the method paramater, one by one depending on the order they are returned from the select function
		#the order is sorted by id to keep consistancy
		# input_nodes = @nodes.select { |node| INPUT_NODE_TYPES.include? node[:type]}
		inputs.each_index do |id|
			node = @nodes.select{ |node| node.id == id }.first
			set_cache(id, node.value( inputs[id] ) ) 
		end
	end

	#Returns an array of node outputs. Checks only for nodes with the :output type
	def get_outputs
		#returns an array of the end values of all the output nodes. sorted by id for consistancy
		output_nodes = @nodes.select { |node| node.type == :output } #there might be more output types later on though
		outputs = output_nodes.map { |node| get_cache(node.id) } #get_cache will calculate the outputs if the cache's are empty. outputs can be reused. Manyally call clear_cache when done and want to reset inputs
		raise "wtf: #{outputs}, cache dump: #{@cache.inspect}" if outputs.nil? or outputs.any? { |output| output.nil? }
		return outputs
	end

	#cache speed ups 

	#Returns the cached value of the specified node. If no cache value exists, it will be looked up and saved
	#
	# === Paramaters
	#
	# *+id+ - the ID of the node
	# *+stack+ - an array of ID's of nodes that should not be checked in case a value must be calculated (used to ignore loops and infinite recursion)
	def get_cache id, stack = []
		#gets the node value, sets it if was not found in the cache

		if @cache[id] 
			@cache[id] #all input nodes are auto-cached
		else
			correct_nodes = @nodes.select{ |node| node.id == id}
			if correct_nodes.count == 0 then
				raise "missing node #{id}"
			else
				correct_node = correct_nodes.first
			end

			input_vals = @synapses.select{ |synapse| synapse[:to] == id}.reject do |synapse| 
				stack.include? synapse[:from]
			end.map do |synapse| 
				(get_cache(synapse[:from], stack + [id] ) * synapse[:weight])
			end

			input_sum = input_vals.reduce(correct_node.action)

			node_value = correct_node.value input_sum
			set_cache(id, node_value)
			raise "bad node return value: #{node_value} for node #{correct_node.inspect}, inputs: #{input_vals.inspect}" if node_value.nil? or node_value.is_a? Integer or node_value.nan?
			return node_value
		end
	end

	#Saves the output value of a node
	#
	# === Paramaters
	# 
	# *+id+ - the node ID that will be cached. Overwrites existing caches
	# *+value+ - the value that is being saved
	def set_cache id, value
		raise "attempting to set nil cache at id = #{id}: #{value}" if value.nil?
		@cache[id] = value
	end

	#Clears the current cache
	def clear_cache
		@cache.clear
	end

	#GA functions. things like fitness should be in the GA/NN controller and exist outside the NN itself

	#Creates a new neural network based on self. ID is preserved
	def clone		
		NeuralNetwork.new(
			self.id, 
			self.nodes, 
			self.synapses, 
			self.mutation_rate, 
			self.severity, 
			self.heavy_mutation_rate
			)
	end

	#Mutates self based on mutation rate, severity and heavy mutation rate.
	# mutations include adjusting synapse weights, adding and destroying nodes, as well as calling node.mutate on all nodes
	def mutate		
		#first mutate global NN values
		#then add or delete nodes if the sevetiry is high enough
		#then mutate each node (call node mutate and pass the rate and severity)

		@mutation_rate 		= mutation_offset(@mutation_rate, @severity			/ 10.0).abs if mutate? @mutation_rate
		@severity 			= mutation_offset(@severity, @severity 				/ 10.0).abs if mutate? @mutation_rate
		@heavy_mutation_rate= mutation_offset(@heavy_mutation_rate, @severity 	/ 10.0).abs if mutate? @mutation_rate

		#ADD OR DELETE A NODE
		# morph_times = @nodes.select{ |node| INPUT_NODE_TYPES.include? node.type or OUTPUT_NODE_TYPES.include? node.type}.count
		# morph_times.times do
		# if mutate? @heavy_mutation_rate

		#add a random node
		if mutate? @heavy_mutation_rate
			#TODO adding a node - adjust for 
			new_id = rand(0..1000000)
			bias = 0

			@nodes << Node.new(new_id, HIDDEN_NODE_TYPES.shuffle.first, bias)


			id_from = @nodes.reject{ |node| OUTPUT_NODE_TYPES.include? node.type }.map { |node| node.id }
			id_to	= @nodes.reject{ |node| INPUT_NODE_TYPES.include?  node.type }.map { |node| node.id }

			id_from.each do |id_f|
				weight = 0
				@synapses << {from: id_f, to: new_id, weight: weight}
			end
			id_to.each do |id_t|
				weight = 0
				@synapses << {from: new_id, to: id_t, weight: weight}
			end
		end

		#delete a random node (this would work better with lower influence nodes)
		if mutate? @heavy_mutation_rate
			hidden_nodes = HIDDEN_NODE_TYPES
			node_for_deletion = @nodes.select{ |node| hidden_nodes.include? node.type}.shuffle.first
			unless node_for_deletion.nil?
				id_for_deletion = node_for_deletion.id

				@nodes.reject! { |node| node.id == id_for_deletion }
				@synapses.reject! { |synapse| (synapse[:to] == id_for_deletion) or (synapse[:from] == id_for_deletion)}
			end
		end

		#add a random synapse
		if mutate? @heavy_mutation_rate
			from = @nodes.reject{ |node| OUTPUT_NODE_TYPES.include? node.type}.shuffle.first.id
			to 	 = @nodes.reject{ |node| INPUT_NODE_TYPES.include?  node.type}.shuffle.first.id

			weight = mutation_offset(0, 1)
			@synapses << {from: from, to: to, weight: weight}
		end				

		#delete a random synapse
		#TODO add a bias to synapses with weights close to 0 / 1 (depends on the reading node)
		if mutate? @heavy_mutation_rate
			chosen_synapse = @synapses.shuffle.first
			@synapses.reject! { |synapse| synapse == chosen_synapse }
		end
		@nodes.map! { |node| node.mutate!(@mutation_rate, @severity, @heavy_mutation_rate) }
		return self
	end

	#Checks if a mutation should occour
	#
	# === Paramaters
	# 
	# *+rate+ - the mutation chance between 0.0 and 1.0
	def mutate? rate
		rand() < rate
	end

	#creates a new value near the initial value and mutation severity
	#
	# === Paramaters
	#
	# *+x+ - the base value. New value will be close to this paramater
	# *+severity+ - the max deviation from the oridinal value.
	def mutation_offset x, severity
		min = (x-severity)
		max = (x+severity)
		# rand ( (x-severity)..(x+severity)  )
		rand() * (max - min) + min
	end
end

