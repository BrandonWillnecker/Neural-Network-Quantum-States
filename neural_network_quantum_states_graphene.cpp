//The GA structure is exactly the same as the previous two examples 
// but here we need to include two inputs, one for x and another for y

//The other changes are to the potential function V(x), we now use 
constexpr float x_min = -2.0f;
constexpr float x_max = 2.0f;
constexpr float y_min = -2.0f;
constexpr float y_max = 2.0f;
constexpr uint32_t N = 100;
constexpr float dx = (x_max-x_min)/(float)N;
constexpr float dy = (y_max-y_min)/(float)N;
constexpr float mu_inv = 0.0045129785968f;

float nuclei_pos[6*2] = pos = { 1.000f, 0.000f, //(x1,y1)
							    0.500f, 0.866f, //(x2,y2)
							   -0.500f, 0.866f, //(x3,y3)
							   -1.000f, 0.000f, //(x4,y4)
							   -0.500f,-0.866f, //(x5,y5)
							    0.500f,-0.866f  //(x6,y6)
							};

float V(float x, float y, float E_field_strength)
{
	float E = 0.0f;
	//Contribution from nuclei_pos
	for(uint32_t i=0;i<6;i++){
		float dx = x - nuclei_pos[2*i];
		float dy = y - nuclei_pos[2*i+1];
		float r = sqrt(dx*dx + dy*dy);
		E += -1.0f/r;
	}
	//Contribution from the external electric field
	E += E_field_strength*x;
	return E;
}

// and the energy function
float get_energy(NeuralNetwork& psi)
{
	set_psi_values(psi);	
    float E = 0.0f;
    float psi_norm_sqr = 0.0f; 
    for (uint32_t i = 1; i < N-1; i++){
		for(uint32_t j=1;j<N-1;j++){
			float x = x_min + (float)i*dx;
			float y = y_min + (float)j*dy;
			float d2fdx2 = (psi_values[i+1][j]
			- 2.0f*psi_values[i][j] + psi_values[i-1][j])/(dx*dx);
			float d2fdy2 = (psi_values[i][j+1] 
			- 2.0f*psi_values[i][j] + psi_values[i][j-1])/(dy*dy);
			E += psi_values[i][j]*(-mu_inv*d2fdx2 - mu_inv*d2fdy2 + V(x,y)*psi_values[i][j]);
			psi_norm_sqr += psi_values[i][j]*psi_values[i][j];
		}
    }
    return E;
}

