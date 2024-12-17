#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include "GANeuralNetwork.h"

size_t parent_index(const std::vector<NeuralNetwork>& generation, Random& random)
{
	size_t index = 0;
	float max_fitness = -FLT_MAX;
	for (int i = 0; i < 10; i++) {
		size_t test_index = random.next_uint() % generation.size();
		if (generation[test_index].fitness > max_fitness) {
			index = test_index;
			max_fitness = generation[test_index].fitness;
		}
	}
	return index;
}

int main()
{
	constexpr size_t GENERATION_SIZE = 100;
	constexpr size_t NUM_GENERATIONS = 2000;
	constexpr float MUTATION_RATE = 0.95f;
	constexpr size_t ELITISM_SIZE = 10;
	std::vector<size_t> layer_sizes = { 4,2,1 };
	Random random;

	std::vector<NeuralNetwork> generation;
	generation.reserve(GENERATION_SIZE);
	for (int i = 0; i < GENERATION_SIZE; i++){
		std::string filename = "ga_neural_network_" + std::to_string(i);
		generation.emplace_back(layer_sizes, filename, random);
	}

	for (int generation_step = 0; generation_step < NUM_GENERATIONS; generation_step++)
	{
		//SORT THE GENERATION
		for (NeuralNetwork& nn : generation) nn.get_fitness();
		std::sort(generation.begin(),
				  generation.end(), 
				  [](const NeuralNetwork& nn1, const NeuralNetwork& nn2) {
			return nn1.fitness > nn2.fitness;
		});

		std::cout << "GENERATION = " << generation_step 
				  << ", MAX FITNESS = " << generation[0].fitness << "\n";

		//TAKE ELITE OVER TO THE NEW GENERATION
		std::vector<NeuralNetwork> new_generation;
		new_generation.reserve(GENERATION_SIZE);
		for (int i = 0; i < ELITISM_SIZE; i++) {
			new_generation.push_back(generation[i]);
		}

		//MUTATION AND CROSS OVER
		for (int i = 0; i < GENERATION_SIZE - ELITISM_SIZE; i++) {
			size_t parent1_index = parent_index(generation, random);
			size_t parent2_index = parent_index(generation, random);
			NeuralNetwork child = NeuralNetwork::crossover(generation[parent1_index],
			generation[parent2_index], random);
			if (random.next_float() < MUTATION_RATE)
				child.mutate(random);
			new_generation.push_back(child);
		}

		generation = new_generation;
	}

	generation[0].save_psi("best_psi.py");
}

