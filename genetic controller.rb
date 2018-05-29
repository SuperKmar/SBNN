#genetic controller
require_relative "neuralNetwork"

STRATAGIES = [:flat, :random, :score_bias, :tournament]

class EvolutionController

	attr_reader :population
	attr_accessor :pop_cap, 
		:min_pop, 
		:max_pop, 
		:survival_rate, 
		:depopulation_stratagy, 
		:repopulation_stratagy,
		:record_evolution

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

	def random_repopulate()
		scored_pop = @population.reject{ |pop| pop[:score].nil? }
		parents = []
		while (@population.count + parents.count) < @pop_cap
			parents << scored_pop.shuffle.first
		end	
	end

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

	def flat_depopulate()
		#take the worst N% and kill them
		@population = @population.sort_by{ |pop| - pop[:score] }.first((@population.count * @survival_rate).round)
	end

	def tournament_depopulate()
		to_keep = ( @population.count * (@survival_rate)).floor
		scored_pop = @population.reject { |pop| pop[:score].nil? }
		while @population.count > to_keep
			pop1, pop2 = scored_pop.shuffle.first(2)
			@population.delete( (pop1[:score] < pop2[:score]) ? pop1 : pop2 ) 
		end
	end

	def random_depopulate()
		#kill random examples until N% are dead
		#@population.reject!{ |pop| rand() > @survival_rate }
		to_keep = ( @population.count * (@survival_rate)).floor
		@population = @population.shuffle.first( to_keep )
	end

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

	#########################################
	#TODO: this should be an optional part of the controller, possibly taken outside
	#the 1.1 and 0.9 mods should be paramaters
	def adapt_population_size
		max_score = @population.max_by{ |pop| pop[:score]}[:score]
		if @best_score.nil?
			@best_score = max_score
		else
			if @best_score >= max_score
				#stagnation
				@pop_cap = [@pop_cap * 1.1, @max_pop].min
				return max_score
			else
				#progress
				@pop_cap = [@pop_cap * 0.9, @min_pop].max
				@best_score = max_score
			end
		end
	end

	#############during training###########
	#######################################
	def test nn, inputs
		nn.set_inputs inputs
		result = nn.get_outputs
		nn.clear_cache
		return result
	end

	def set_score nn, score
		selected_pops = @population.select { |pop| pop[:nn] == nn }
		selected_pops.each { |pop| pop[:score] = score }
		
		if @max_score.nil?
			@max_score = score
		else
			@max_score = [@max_score, score].max
		end		
	end

	def evolve
		record_evolution() if @record_evolution
		depopulate() 
		repopulate()
	end	

	def record_evolution
		@evolution_record = [] if @evolution_record.nil?
		@evolution_record << @population.map{|pop| pop[:score]}.sort.reverse
	end

	########################################
	############getting results#############
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