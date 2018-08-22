# SBNN
general self building neural network in ruby

To create a neural network controller:
require "genetic controller"
controller = EvolutionController.new(
    3, #Amont of input nodes
    2, #Amount of output nodes
    10, #Initial amount of hidden nodes
    
    100, #Population cap. This value can be manually changed later
    0.5, #Initial mutation rate
    0.5, #Initial severity of mutation
    0.5, #Initial chance of adding and removing nodes (heavy mutation rate)
    
    :flat, #depopulation stratagy
    :score_bias, #repopulation stratagy
    0.5, #Survival rate after depopulation
    
    false, #use multithreadding ('true' for max threads, int to set amount of extra threads when cloning population)
    false) #record statistics of evolution
    
# "controller" now has 100 similar fully connected neural networks and the methods to test them and score them

#testing every neural network on our test data:
test_input_array = [0.1, 0.5, 0.9] 

# test data in the array is sorted. 
# Input nodes have incremental id's starting from 0 and are assigned the corresponding array element. 
# This example uses only one test input, but you can score neural networks across multiple test cases 
# and rate their significance as you wish

controller.population.each do |pop|
  nn = pop[:nn]
  outputs = controller.test( nn, test_input_array )

  #rating the output with the desired value
  #...
  #...
  #setting the score of the nn based on our expectations. Score can include a penalty for having too many or too few nodes / synapses

  size_penalty = -nn.nodes.count / 50.0
  size_penalty -= nn.synapses.count / 50.0

  controller.set_score( nn, base_score + size_penalty)
end

# after scores have been set launch an evolution cycle

controller.evolve

# after multiple traning cycles and evolution cycles you may stop when the accuracy of responce is within acceptable borders
# get the best neural network:

nn = controller.population.reject{ |pop| pop[:score].nil? }.max_by{ |pop| pop[:score]}[:nn]

# displaying the best neural network based on an array of test inputs (which are also arrays)

control_inputs.each_index do |i|
    outputs = controller.test(nn, control_inputs[i] )
    puts "============"
    puts "test input: #{control_inputs[i].join("; ")}"
    puts "expected:"
    puts control_outputs[i].join("; ")
    puts "got (rounded):"
    puts "#{outputs.map{ |out| out.round(3) }.join("; ")} (#{outputs.map { |out| out.round }.join("; ")})"
    puts
end

# saving the best neural network to a graph:

controller.save_graph(nn, "best nn #{nn.id}" )

# saving evolution statistics

controller.save_charts("my_evolution_statistics.csv")
