/*
 * This is the simulation model for paper 'Infectivity enhances prediction
 *  of viral cascades in Twitter', by Weihua Li, Skyler J. Cranmer,
 *  Zhiming Zheng & Peter J. Mucha. 
 *  
 *  It requires two input files:
 *  (1) The 'network.dat' is the network data. The origianl Twitter follower
 *   network can be downloaded from http://carl.cs.indiana.edu/data/#virality2013. 
 *  (2) The 'infectivity_mu0.0012_sigma2.4_maxRate0.017.dat' is the cumulative 
 *  probability distribution of infectivity_{null}, and should be used when decay
 *  factor alpha is set to 0.01. 
 *  
 * Given the amount of computation for the simulation, parallel computing is
 *  implemented with mpi.h. There will be multiple output files generated 
 *  by this program.
 *  
 * An output file stores the retweet data in panel data format: 
 *  retweet time, cascade id, cascade initial time, cascade infectivity,
 *  retweeted user id, originated user id.
 *  
 */


#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <queue>
#include <vector>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <mpi.h>

using namespace std;

#define N 595460 // Number of Twitter users
int T = 133; // Total time-steps; (T - 33) burn-in steps
#define M 43 // Memory length
#define numFiles 10 // Number of simulations per core; total simulations = numFiles * number of cores
#define N_Hash 80000000 // Allocate memory for tweets
double beta = 0.528; // Innovation rate
double alpha = 0.01; // Decay factor on infectivity

int t = 0; // Current time-step
int numHash = 0; // Current hashtag number
double attRateDistrib[501] = {0.0}; // Infectivity distribution
double lam; // Infectivity

struct Hash{
  int t0;               // When the cascade started
  double lam0;              // Infectivity null
  int node0;	// User who posted the tweet
};

Hash allHash[N_Hash];
vector <int> retweetedPost[N];
vector <int> point[N];	    // Twitter follower network
vector <int> memory[N];		// Memory to store tweets
ofstream Fout1, Fout2;

void broadcast(int id, int hashId); // Tweet cascade hashId to all followers
int check_ifTweeted(int id, int tweet); // Eliminate repeated retweets
void init();            // Initialize
void innovate(int id); // User id generates a new cascade
void input();           // Read input files
int my_rand(); // Generate large random integers
void ReRankMeme(int i, int hashId); // Put new come cascade first
void simu(vector<string> outF); // Run simulation
void simuOne();
void sumHash();

int main ( int argc, char *argv[] ){
  int rank, size;
  MPI_Init (&argc, &argv); /* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);	/* get number of processes */
    
  srand (( time(NULL) + 1) * (rank + 1 ) ); /* initialize random seed: */
  vector<string> outFiles;
  char tmpS[100];
  for(int i = 0; i < numFiles; i++){
	  sprintf(tmpS, "%s%d%s%d%s", "Out_cl", rank, "_", i, ".txt");
	  outFiles.push_back(tmpS);
  }
  simu(outFiles);
  
  int finalize_retcode = MPI_Finalize();
  if(0 == rank) fprintf(stderr, "Process, return_code\n");
  fprintf(stderr, "%i, %i\n", rank, finalize_retcode);
  return 0;
}

void input(){
  int i;
  ifstream fin, fin1;
  int id1,id2;
  double pro; 
  fin1.open("infectivity_mu0.0012_sigma2.4_maxRate0.017.dat"); // Import attRate distribution for hashtags
  
  if (fin1.is_open()){
    cout << "Successfully open attRateDistrib.dat\n";
  }
  i = 0;
  while(fin1 >> pro){
    attRateDistrib[i] = pro;
    i++;
  }
  fin1.close();
   
  // Read reciprocal network
  fin.open("network.dat");
  while(fin >> id1 >> id2){
    point[id1].push_back(id2);
    point[id2].push_back(id1);  
  }
  fin.close();
  cout << "Network ready for simualtion." << endl;
}

// Put the cascade hashId on the screens of user id's followers
void broadcast(int id, int hashId){ 
  int tail;
  for(int i = 0; i < point[id].size(); i++){
   tail = point[id][i];
   ReRankMeme(tail, hashId);
  }
}

int check_ifTweeted(int id, int tweet){
	int len;
	len = retweetedPost[id].end() - retweetedPost[id].begin();
	if(len == 0) return 0;
	if(len > 0){
		for(int i = 0; i < len; i++){
			if(retweetedPost[id][i] == tweet) return 1;
		}
		return 0;
	}
}

void init(){
	int i, j;
	// clear memory
	for(i = 0; i < N; i++)	memory[i].clear();	

	// clear retweetedPost[N]
	for(i = 0; i < N; i++)	retweetedPost[i].clear();

	// clear allHash
	for(i = 0; i < N_Hash; i++){
		allHash[i].lam0 = 0.0;
		allHash[i].t0 = 0;
	}

	// set up memory to 0
	for(i = 0; i < N; i++){
		for(j = 0; j < M; j++)	memory[i].push_back(0);
	}
	numHash = 1; 
}

void innovate(int id){  
  // With probability mu, user id innovates and tweets a new cascade.
  double rnd, rndHash;
  int ino = 0;
  Hash hash = {0,0};

  // Innovate. At most one cascade will be created.
  // The infectivity will be sampled from a distribution
  rnd = (double) rand() / RAND_MAX;
  if(rnd < beta){
	  rndHash = (double) rand() / RAND_MAX;
	  for(int i = 0; i < 500; i++){ 
		  if(rndHash >= attRateDistrib[i] && rndHash <= attRateDistrib[i + 1]){
			  hash.t0 = t;
			  hash.lam0 = (i + 1) * 0.0001; 
			  hash.node0 = id;
			  allHash[numHash] = hash;
			  retweetedPost[id].push_back(numHash);
			  broadcast(id, numHash);
			  numHash++; 
			  if(numHash >= N_Hash){
				  cout << "too many hashtags!";
				  exit(1);
			  }
			  break;
		  }
	  }
  }

  // Choose some cascades in his/her memory and retweet
  for(int i = 0; i < M; i++){
	  rnd = (double) rand() / RAND_MAX;
	  int theHash = memory[id][i]; 
	  if(theHash != 0 && rnd < allHash[theHash].lam0 * exp( - alpha * (t - allHash[theHash].t0))){
		  if(check_ifTweeted(id, theHash) == 0){ // No repeated retweets of the same cascade
			  retweetedPost[id].push_back(theHash);				       
		    broadcast(id, theHash);
			if(t > (T - 33)){
			  Fout1 << t << " " << theHash << " " << allHash[theHash].t0 << " " << allHash[theHash].lam0 << " ";
			  Fout1 << id << " " << allHash[theHash].node0 << " " << endl;  // retweeted node; retweeted from node (source node)
			 }		  
		  }
	  }
  }
  return;
}

int my_rand(){
	int a, b, c;
	a = rand() % 4096;
	b = rand() % 4096;
	c = a * 4096 + b;
	return c;
}

void ReRankMeme(int i, int hashId){  
	memory[i].insert(memory[i].begin(), hashId);
	memory[i].erase(memory[i].begin() + M); 
	return;
}

void simu(vector<string> outF){
  input();
  for(int i = 0; i < numFiles; i++){
	  Fout1.open(outF[i].c_str());
	  cout << "Start to simulate" << " " << i <<endl;
	  simuOne();
	  Fout1.close();
	  Fout1.clear();
  }
  return;
}

void simuOne(){
  int i;
  init();
  for(t = 1; t <= T; t++){
	cout << t << endl;
    for(i = 0; i < N; i++){	
	innovate(i);
    }
  }
  return;
}






