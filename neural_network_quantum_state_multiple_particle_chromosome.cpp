#pragma once
#include "Utils.h"
#include <string>
#include <vector>
#include <fstream>

//Only the important functions: mutate, crossover and 
//get_fitness are shown here for brevity
//The full code can be found on GitHub/BrandonWillnecker

void mutate(Random& random)
{
	size_t layer_index =  random.next_uint() % m_weights.size();
	size_t weights_cols = m_layer_sizes[layer_index];
	size_t weights_col = random.next_uint() % m_layer_sizes[layer_index];
	size_t weights_row = random.next_uint() % m_layer_sizes[layer_index + 1];
	size_t biases_row = random.next_uint() % m_layer_sizes[layer_index + 1];

	m_weights[layer_index][weights_row * weights_cols + weights_col] +=
	0.5f * (random.next_float() - 0.5f);
	m_biases[layer_index][biases_row] +=
	0.5f * (random.next_float() - 0.5f);
}

static NeuralNetwork crossover(const NeuralNetwork& nn1, const NeuralNetwork& nn2, Random& random)
{
	NeuralNetwork child(nn1.m_layer_sizes, "fn", random);
	for (size_t layer_index = 0; layer_index < nn1.m_layer_sizes.size()-1; layer_index++)
	{
		if (layer_index % 2 == 0){
			child.m_weights[layer_index] = nn1.m_weights[layer_index];
			child.m_biases[layer_index] = nn1.m_biases[layer_index];
		}else {
			child.m_weights[layer_index] = nn2.m_weights[layer_index];
			child.m_biases[layer_index] = nn2.m_biases[layer_index];
		}
	}
	return child;
}

float get_fitness()
{
	float energy = 0.0f;
	float norm_sqr = 0.0f;

	//Set all psi(x1,x2,x3) values
	for (int i = 0; i < N; i++) {
		for (int j = i+1; j < N; j++) {
			for (int k = j+1; k < N; k++) {
				float x1 = L_min + i * dx;
				float x2 = L_min + j * dx;
				float x3 = L_min + k * dx;
				psi[i][j][k] = output(x1, x2, x3) *modulation_function(x1, x2, x3, 0.2f);
			}
		}
	}

	//Get the energy
	for (int i = 1; i < N-1; i++) {
		for (int j = i+1; j < N-1; j++) {
			for (int k = j+1; k < N-1; k++) {
				norm_sqr += psi[i][j][k] * psi[i][j][k];

				//Kinetric for x1
				energy += psi[i][j][k] * (-m) * (psi[i + 1][j][k] 
				- 2.0f * psi[i][j][k] + psi[i - 1][j][k]) / (dx * dx);
				//Kinetric for x2
				energy += psi[i][j][k] * (-m) * (psi[i][j + 1][k]
				- 2.0f * psi[i][j][k] + psi[i][j - 1][k]) / (dx * dx);
				//Kinetric for x3
				energy += psi[i][j][k] * (-m) * (psi[i][j][k + 1]
				- 2.0f * psi[i][j][k] + psi[i][j][k - 1]) / (dx * dx);

				//Background potential
				energy += psi[i][j][k] * (V_atomic_chain[i]
				+ V_atomic_chain[j] + V_atomic_chain[k]) * psi[i][j][k];

				float x1 = i * dx;
				float x2 = j * dx;
				float x3 = k * dx;

				//Interaction x1-x2
				float x12 = abs(x1 - x2);
				energy += psi[i][j][k] * (1.0f / x12) * psi[i][j][k];

				//Interaction x2-x3
				float x23 = abs(x2 - x3);
				energy += psi[i][j][k] * (1.0f / x23) * psi[i][j][k];

				//Interaction x1-x3
				float x13 = abs(x1 - x3);
				energy += psi[i][j][k] * (1.0f / x13) * psi[i][j][k];
				
			}
		}
	}

	fitness = -energy / norm_sqr;
	return fitness;
}
