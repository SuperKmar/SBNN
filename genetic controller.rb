#genetic controller
require_relative "neuralNetwork"

# CONSTANTS
# STRATAGIES - an array of depopulation and repopulation stratagies. Used for comparison and for the user to choose from
# :flat - will kill the N% worst or use the N% best for cloning
# :random - unbiased killing of N random solutions, selection of N random parents for cloning
# :score_bias - lower score increases chance of being deleted, higher score increases chance of being cloned
# :tournament - selects two random solutions. Worst score is killed if used for depopulation, best score is cloned if used for repopulation
STRATAGIES = [:flat, :random, :score_bias, :tournament]

#EvolutionController class provides the interface for setting up a population, scoring and untilizing genetic algorithms for improving the results
# == READERS:
# population - an array of hashes, where the hash has a neural network class object and a score
#
# == ACCESSORS:
# pop_cap - current population cap for genetic algorithm
# min_pop - minimum population cap for genetic algorithm (values under 20 not reccomended)
# max_pop - maximum population cap for genetic algorithm
# survival_rate - amount of population that gets killed each cycle.
# depopulation_stratagy - the method used to kill off population. Must be an element of the STRATAGIES constant: [:flat, :random, :score_bias, :tournament]
# repopulation_stratagy - the method used select population members for cloning. Must be an element of the STRATAGIES constant: [:flat, :random, :score_bias, :tournament]
# record_evolution - if true, will record evolution history such as population counts, fitness scores and generations. Outputs to a single table.
class EvolutionController

	#EvolutionController readable and modable values:
	attr_reader :population
	attr_accessor :pop_cap, 
		:min_pop, 
		:max_pop, 
		:survival_rate, 
		:depopulation_stratagy, 
		:repopulation_stratagy,
		:record_evolution

	#Creates a new controller
	# === Attributes
	# * +inputs+ - int; Amount of input nodes in the neural network
	# * +outputs+ - int; Amount of output nodes in the neural network
	# * +hidden+ - int; Amount of hidden nodes in the initial neural network. This amount can change. Initial NN is fully connected
	# * +pop_cap+ = 20; Initial population cap. This value can be changed each generation if needed.
	# * +mutation_rate+ = 0.01; Initial mutation rate for a neural network. This value can mutate as well and is likely to drop to very low values
	# * +severity+ = 0.1; How drastic the mutation is. This value can mutate as well and is likely to drop.
	# * +heavy_mutation_rate+ = 0.1; The rate at which nodes and synapses can be added / deleted. This value can mutate and is likely to drop to very low values.
	# * +depopulation_stratagy+ = :random; The method by which population members are destroyed.
	# * +repopulation_stratagy+ = :score_bias; The method by which parents are selected from the population for cloning and mutation
	# * +survival_rate+ = 0.5; Amount of population (%) subject to depopulation when creating next generation
	# * +threads+ = false; Use multithreadding or not. True means each NN will get it's own thread. An integer will use that many threads (i.e. threads = 3 means only 3 extra threads will be used). False uses single threaded mutations
	# * +record_evolution+ = false; Record the evolution proccess (generations, scores, populations). Can be turned off and on if only several points are needed.
	def initialize (
			inputs, 
			outputs, 
			hidden, 
			pop_cap = 20,
			mutation_rate = 0.01, 
			severity = 0.1, 
			heavy_mutation_rate = 0.1,					
			depopulation_stratagy = :random, 
			repopulation_stratagy = :score_bias, 
			survival_rate = 0.5,
			threads = false,
			record_evolution = false)

		@pop_cap = pop_cap

		@depopulation_stratagy 	= depopulation_stratagy
		@repopulation_stratagy 	= repopulation_stratagy
		@survival_rate 			= survival_rate

		@threads = threads # threads can be:
		# false - use classic single thread approach
		# true - use as many threads as possible (might be a bad idea)
		# a number - use this many extra threads
		@record_evolution = record_evolution

		@population = []
		@pop_cap.times do 

			input_nodes		= (0...inputs ).map { |i| Node.new( i , :input, rand(-1.0..1.0) ) }
			output_nodes 	= (0...outputs).map { |i| Node.new(rand(0..1000000), :output, rand(-1.0..1.0) ) }

			hidden_nodes	= (0...(hidden)).map { |i| Node.new(rand(0..1000000), :sigmoid, rand(-1.0..1.0)) }

			synapses = []

			hidden_nodes.each do |hidden_node|
				input_nodes.each do |input_node|
					synapses << {from: input_node.id, to: hidden_node.id, weight: rand(-1.0..1.0)}
				end
				output_nodes.each do |output_node|
					synapses << {from: hidden_node.id, to: output_node.id, weight: rand(-1.0..1.0)}
				end
			end
			hidden_nodes.combination(2) do |node_from, node_to|
				synapses << {from: node_from.id, to: node_to.id, weight: rand(-1.0..1.0)}
			end


			baseNN = NeuralNetwork.new( rand(1000000), 
				input_nodes + output_nodes + hidden_nodes, 
				synapses, 
				mutation_rate, 
				severity, 
				heavy_mutation_rate)

			@population << {nn: baseNN, score: nil}#no idea about this
		end
		return self
	end

	#REPOPULATE SECTION
	
	#Repopulates up to @pop_cap.	
	def repopulate
		parents = case @repopulation_stratagy			
		when :flat
			flat_repopulate()
		when :tournament
			tournament_repopulate()
		when :score_bias
			score_bias_repopulate()
		when :random
			random_repopulate()
		end
		
		breed( parents )
	end

	#Selects the best population members rated by their score. Returns the selected parents
	def flat_repopulate()
		# @population.reject{ |pop| pop.nil? }
		parents = []
		@population = @population.sort_by{|pop| - pop[:score]} #best first
		pop_count = @population.count
		i = 0
		while (pop_count + parents.count) < @pop_cap
			parents << @population[i]
			i = (i + 1) % pop_count
		end
		return parents
	end

	#Selects random population members. Returns the selected parents
	def random_repopulate()
		scored_pop = @population.reject{ |pop| pop[:score].nil? }
		parents = []
		while (@population.count + parents.count) < @pop_cap
			parents << scored_pop.shuffle.first
		end	
	end

	#Selects parents based on their score. Higher score gives a higher chance to be selected. Returns the selected parents
	def score_bias_repopulate()
		parents = []
		max_score = @population.max_by { |pop| pop[:score]}[:score]
		min_score = @population.min_by { |pop| pop[:score]}[:score]
		scored_pop = @population.reject{ |pop| pop[:score].nil? }
		while (@population.count + parents.count) < @pop_cap
			parent = scored_pop.shuffle.first
			if max_score == min_score
				chance = 1
			else
				parent_score = parent[:score]
				chance = (parent_score - min_score) / (max_score - min_score)
			end

			if rand() < chance
				parents << parent
			end
		end
		return parents
	end

	#Selects the best of 2 random population members based on their score. Returns the selected parents
	def tournament_repopulate()
		scored_pop = @population.reject{ |pop| pop[:score].nil? }
		parents = []
		while (@population.count + parents.count) < @pop_cap
			parent1, parent2 = scored_pop.shuffle.first(2)

			if parent2.nil? or (parent1[:score] > parent2[:score])
				parents << parent1
			else
				parents << parent2
			end
		end
		return parents
	end

	#Clones the parents and adds them to the population. This is the only multithreaded part.
	# === Paramaters
	# * +parents+ - an array of parents to be clones and mutated.
	def breed ( parents )
		case @threads 
		when false
			@population += parents.map { |parent| {nn: parent[:nn].clone.mutate, score: nil} }
		when true 
			#unlimited threads (bad idea)
			thread_results = []
			parents.each do |parent|  
				thread_results << Thread.new do
			      	Thread.current["child"] = {nn: parent[:nn].clone.mutate, score: nil}
			   	end
			end

			@population += thread_results.map do |thread| 
				thread.join 
				thread["child"]
			end
		else
			i = 0
			thread_results = []
			parents.group_by{(i+=1) % @threads }.each do |k, parent_array|
				thread_results << Thread.new do
					Thread.current["children"] = parent_array.map { |parent| {nn: parent[:nn].clone.mutate, score:nil} }
				end
			end
			@population += thread_results.map do |thread|
				thread.join
				thread["children"]
			end.reduce(:+)
		end
	end

	#DEPOPULATE SECTION
	
	#Kills off N% of the population, based on the survivability paramater
	def depopulate
		case @depopulation_stratagy
		when :flat
			flat_depopulate()			
		when :tournament			
			tournament_depopulate()
		when :random			
			random_depopulate()
		when :score_bias
			score_bias_depopulate()
		end			
	end

	#Kills off the worst N% based on score
	def flat_depopulate()
		#take the worst N% and kill them
		@population = @population.sort_by{ |pop| - pop[:score] }.first((@population.count * @survival_rate).round)
	end

	#Kills off the worst out of 2 members N% times
	def tournament_depopulate()
		to_keep = ( @population.count * (@survival_rate)).floor
		scored_pop = @population.reject { |pop| pop[:score].nil? }
		while @population.count > to_keep
			pop1, pop2 = scored_pop.shuffle.first(2)
			@population.delete( (pop1[:score] < pop2[:score]) ? pop1 : pop2 ) 
		end
	end

	#Kills off N% random members
	def random_depopulate()
		#kill random examples until N% are dead
		#@population.reject!{ |pop| rand() > @survival_rate }
		to_keep = ( @population.count * (@survival_rate)).floor
		@population = @population.shuffle.first( to_keep )
	end

	#Kills off members based on their score. Lower score increases the chance to be selected for elimination
	def score_bias_depopulate()
		#use reverse score bias: worst score = 100% death rate, best score = 0% death rate
		starting_count = @population.count
		while @population.count > starting_count * @survival_rate

			max_score = @population.max_by { |pop| pop[:score]}[:score]
			min_score = @population.min_by { |pop| pop[:score]}[:score]
			
			pop = @population.shuffle.first
			
			if max_score == min_score
				chance = 0
			else
				score = pop[:score]
				chance = (score - min_score) / (max_score - min_score)
			end

			if rand() > chance
				@population.delete( pop )
			end
		end
	end

	#############during training###########
	#######################################
	
	#Runs the neural network with the provided inputs
	# === Paramaters
	# * +nn+ - the neural network. This is the :nn part of the population member hash ( :score is ommited )
	# * +inputs+ - an array of floats or integers. The order if the inputs shouldn't change as array index is tied to input node id's
	def test nn, inputs
		nn.set_inputs inputs
		result = nn.get_outputs
		nn.clear_cache
		return result
	end

	#Sets the score for selected neural network
	# === Paramaters
	# * +nn+ - the neural network (:nn) part of a population member ( :score is ommited )
	# * +score+ - the score to be set for the neural network
	def set_score nn, score
		selected_pops = @population.select { |pop| pop[:nn] == nn }
		selected_pops.each { |pop| pop[:score] = score }
		
		if @max_score.nil?
			@max_score = score
		else
			@max_score = [@max_score, score].max
		end		
	end

	#Depopulates and repopulated the neural networks. Scores should be set beforhand unless using :random depop and repop stratagies. If recording is set then the recording happens at this point
	def evolve
		record_evolution() if @record_evolution
		depopulate() 
		repopulate()
	end	
	
	#Records the populations, their scores and generations
	def record_evolution
		@evolution_record = [] if @evolution_record.nil?
		@evolution_record << @population.map{|pop| pop[:score]}.sort.reverse
	end

	########################################
	############getting results#############
	
	#Dumps a YAML representation of the :nn portion of the population members. The YAML is returned to be saved manually where needed
	# === Paramaters
	# * +best+ = true; If true, only the best neural network will be offloaded. Otherwise the population is sorted by score and offloaded in an array of neural networks
	def offload best = true
		if best
			pop = @population.max_by{ |pop| pop[:score]}
			YAML.dump(pop[:nn])
		else
			YAML.dump(
				@population
					.sort_by{ |pop| - pop[:score] }
					.map { |pop| pop[:nn] }
				)
		end
	end

	#Saves a visual representation of a neural network to a graph. Offloads a PDF and a JPG format. Large graphs (50+ nodes) cantake about 15 minutes to save to JPG. Use of this function requires the GraphViz gem to be installed
	# === Paramaters
	# * +nn+ - the neural network to be saved
	# * +filename+ - the path to save the file
	def save_graph nn, filename
		#this requires GraphViz to be installed, so "meh" at best?
		
		require "graphviz"

		g = GraphViz.new( :G , "rankdir" => "LR")
		gnodes = {}

		nn.nodes.each do |node|
			label = "#{node.id}: #{node.type} #{node.action}"
			shape = nil
			if node.type == :input
				shape = :invhouse
			elsif node.type == :output
				shape = :house
			end

			if shape.nil?
				gnode = g.add_nodes( node.id.to_s, :label => label ) #
			else
				gnode = g.add_nodes( node.id.to_s, :label => label, shape: shape ) #
			end
			gnodes[ node.id ] = gnode
		end

		nn.synapses.each do |synapse|
			g.add_edges( 
				gnodes[synapse[:from]], 
				gnodes[synapse[:to]], 
				:label => synapse[:weight].round(4)
				)
		end

		g.output( :pdf => "#{filename}.pdf" )
		g.output( :jpg => "#{filename}.jpg" )
	end

	#If recording evolution was enabled, this function will offload the statistics to a csv file
	# === Paramaters
	# * +filename+ - the filename to offload the csv file to
	def save_charts filename
		#offload happens in .csv to avoid adding gems
		File.open(filename, 'w') do |file| 
			file.puts("generation;#{(1..@population.count).to_a.join(";")};" ) 
			@evolution_record.each.with_index(1) do |record_line, line_number|
				file.puts("#{line_number};#{record_line.join(";")};")
			end			
		end
	end
end
