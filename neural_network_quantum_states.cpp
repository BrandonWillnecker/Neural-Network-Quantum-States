#include <vector>
#include "NeuralNetwork.h"
#include <mutex>
#include "Eigen/Dense"

constexpr float L = 1.0f;  //Length of the box
constexpr uint32_t N = 100;//Number of discretized psi values
constexpr float dx = 1.0f/(float)(N);

float psi_values[N] = {};

float V(float x)
{
	//Particle in a box potential
	//The psi values at the boundaries are clamped to zero since V->+inf
	return 0.0f; 
	//return 0.5*x*x; for QHO
}

float modulation(float x)
{
	//Infinite potential well with a=0 and b=1
	const float a = 0.0f;
	const float b = 1.0f;
	const float w = 0.01f;
	float x1 = (x-a)/w;
	float x2 = (x-b)/w;
	
	if(x<=a)
		return 0.0f;
	else if(a<x && x<=a+w)
		return x1*x1*(3.0f - 2.0f*x1);
	else if(a+w<x && x<=b-w)
		return 1.0f;
	else if(b-w< x && x <= b)
		return x2*x2*(3.0f - 2.0f*x2);
	else
		return 0.0f;
}

void set_psi_values(NeuralNetwork& psi)
{
	for(uint32_t i=0;i<N;i++){
		float x = (float)i*dx;
		Eigen::VectorXf inputs(1);
		inputs << x;
		psi_values[i] = psi.calculate_outputs(inputs)(0)*modulation(x);
	}
}

float get_energy(NeuralNetwork& psi) {
	set_psi_values(psi);	
    float E = 0.0f;
    float psi_norm_sqr = 0.0f; 
    for (uint32_t i = 1; i < N-1; i++) {
		float x = (float)i*dx;
		float d2fdx2 = (psi_values[i+1] - 2.0f*psi_values[i] + psi_values[i-1])/(dx*dx);
		E += -psi_values[i]*(d2fdx2 + V(x)*psi_values[i]);
		psi_norm_sqr += psi_values[i]*psi_values[i];
    }
    return E;
}

uint32_t tournament_selection(const uint32_t& seed, std::vector<NeuralNetwork>& generation)
{
    uint32_t best_index = 0;
    uint32_t tournament_size = 10;
    float lowest_cost = FLT_MAX;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (uint32_t i = 0; i < tournament_size; i++) {
        uint32_t index = (uint32_t)(generation.size() * dis(gen));
        float cost = get_energy(generation[index]);
        if (cost < lowest_cost) {
            lowest_cost = cost;
            best_index = index;
        }
    }
    return best_index;
}

void mutate(const uint32_t& seed, uint32_t generation_step, NeuralNetwork& psi) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> p_dis(0.0f, 1.0f);

	//Used to specify the min and max amount of variation in the weights adjustment
	float f = 0.01f; 
    uint32_t i = 0, j = 0, k = 0;
    float prev_value = 0.0f;

	//Choose a random layer
    i = (uint32_t)(p_dis(gen)* psi.get_num_layers());
	//Choose a random row from the chosen weight matrix
    j = (uint32_t)(p_dis(gen)* psi.weights(i).rows());
	//Choose a random col from the chosen weight matrix
    k = (uint32_t)(p_dis(gen)*psi.weights(i).cols());

    prev_value = abs(psi.weight(i, j, k));
    std::uniform_real_distribution<float> weight_shift_dis(-f*prev_value, f*prev_value);

    psi.weight(i,j,k) += weight_shift_dis(gen);

    i = (uint32_t)(p_dis(gen)* psi.get_num_layers());
    j = (uint32_t)(p_dis(gen)* psi.biases(i).rows());

    prev_value = abs(psi.bias(i,j));
    std::uniform_real_distribution<float> bias_shift_dis(-f * prev_value, f * prev_value);

    psi.bias_vectors[i](j) += bias_shift_dis(gen);
}

void crossover(const uint32_t& seed,
              int generation_step, 
			  std::vector<NeuralNetwork>& generation,
			  std::vector<NeuralNetwork>& new_generation)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::stringstream thread_id;
    thread_id << std::this_thread::get_id();

    const float mutation_rate = 0.9f;

    uint32_t parent_1_index = tournament_selection(seed, generation);
    uint32_t parent_2_index = tournament_selection(seed, generation);

    NeuralNetwork child_1 = generation[parent_1_index];
    NeuralNetwork child_2 = generation[parent_2_index];

    uint32_t i= 0, j= 0, k= 0;
    float tmp;

    //Pick a random weight and swop
    i = (uint32_t)(dis(gen)*child_1.get_num_layers());
    j = (uint32_t)(dis(gen)*child_1.weights(i).rows());
    k = (uint32_t)(dis(gen)*child_1.weights(i).cols());

    tmp = child_1.weight(i,j, k);
    child_1.weight(i,j, k) = child_2.weight(i, j, k);
    child_2.weight(i,j, k) = tmp;

    //Pick a random bias and swap
    i = (uint32_t)(dis(gen)*child_1.get_num_layers();
    j = (uint32_t)(dis(gen)*child_1.biases(i).rows());

    tmp = child_1.bias(i,j);
    child_1.bias(i,j) = child_2.bias(i,j);
    child_2.bias(i,j) = tmp;

    //mutation
    if (dis(gen) < mutation_rate) mutate(seed, generation_step,child_1);
    if (dis(gen) < mutation_rate) mutate(seed, generation_step,child_2);


    //take the better child
    std::lock_guard<std::mutex> lock(new_generation_mutex);
    new_generation.push_back(child_1);
    new_generation.push_back(child_2);
}

int main()
{

	srand((unsigned int)time(NULL));
	const uint32_t GENERATION_SIZE = 50;
	const uint32_t ELITISM_SIZE = 5;
	const uint32_t MAX_GENERATIONS = 10000;
	double duration = 0;
	float lowest_energy = FLT_MAX;

	std::vector<uint32_t> layer_sizes = {1,10,10,10,10,1};
	
	std::cout << "Creating initial generation...\n";

	std::vector<NeuralNetwork> generation;
	generation.reserve(GENERATION_SIZE);
	while (generation.size()<GENERATION_SIZE)
        generation.emplace_back(layer_sizes);

	std::cout << "Starting...\n";

	for (uint32_t step = 0; step < MAX_GENERATIONS; step++) {
		{
			Timer timer(&duration);

			//Sort the generation
			std::partial_sort(generation.begin(),
							generation.begin() + ELITISM_SIZE,
							generation.end(), 
							[](NeuralNetwork& psi1, NeuralNetwork& psi2) 
							{return psi1.get_energy() < psi2.get_energy();}
							);
			
			std::vector<NeuralNetwork> new_generation;
			new_generation.reserve(GENERATION_SIZE);

			for (uint32_t i = 0; i < ELITISM_SIZE; i++)
				new_generation.push_back(generation[i]);

			for (uint32_t i = 0; i < GENERATION_SIZE-ELITISM_SIZE; i++) {
				uint32_t seed = rand();
				std::thread t(crossover, 
				std::ref(seed), 
				step,
				std::ref(generation),
				std::ref(new_generation));
				add_thread(std::move(t));
			}	
			join_threads();
			clear_threads();

			//generation = new_generation;
			std::swap(generation, new_generation);
		}

	}
	generation[0].save_to_file("neuro_wave_function.txt");
}