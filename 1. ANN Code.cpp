
#include<iostream>
#include<fstream>
#include<cmath>
#include<stdlib.h>
#include<time.h>
  
using namespace std;

#define tolerance 0.0001
#define P_Testing 5
#define learning_rate 0.25


int main()
{
 
// Declaration of the variables used

   int p,P,L,M,N,i,j,k,iteration=0;
    double I[100][100],IA[100][100],V[100][100],W[100][100],IH[100][100],OH[100][100],IO[100][100],OO[100][100],TO[100][100],TA[100][100];
	double delW[100][100],delV[100][100],error,MSE,MSE_prd=0,err_prd=0;
	double maxI[100],minI[100],maxTO[100],minTO[100];


    ifstream input;                          
      input.open("inputdata.txt");
      ifstream target;
      target.open("targetdata.txt");
      input>>P>>L>>M>>N;                                  // ANN parameter taken from file
	  ofstream output("solution.txt");
	  ofstream mse("MSE_Iteration.txt");
      output<<"No of pattern="<<P<<endl<<"No of Input="<<L<<endl<<"No of Hidden Neuron="<<M<<endl<<"No of Output neuron="<<N<<endl;
      for(p=1;p<=P;p++)
	{
		for(i=1;i<=L;i++)
		{
			input>>IA[i][p];                           // Input Data taken from file  (IA: Actual Input)
		}
	}
	for(i=1;i<=L;i++)
	{
		maxI[i]=-500;minI[i]=500;                       // Maximum and Minimum Value finding from the data 
		for(p=1;p<=P;p++)
		{
			if(I[i][p]>maxI[i])
			maxI[i]=IA[i][p];
			if(I[i][p]<minI[i])
			minI[i]=IA[i][p];
		}
	}

    // Normalization of the input data

	for(p=1;p<=P;p++)
	{
		for(i=1;i<=L;i++)
		{
			I[i][p]=((IA[i][p]-minI[i])/(maxI[i]-minI[i]));
		}
	}
	for(p=1;p<=P;p++)
	{
		for(k=1;k<N+1;k++)
		{
			target>>TA[k][p];              // Target data taken from another file
		}
	}
	for(k=1;k<N+1;k++)
	{
		maxTO[k]=1000;minTO[k]=550;
		for(p=1;p<=P;p++)
		{
			if(TA[k][p]>maxTO[k])         // Maximum and Minimum data finding for normalization
			maxTO[k]=TA[k][p];
			if(TA[k][p]<minTO[k])
			minTO[k]=TA[k][p];
		}
	}

    // Normalization of the target data

	for(p=1;p<=P;p++)
	{
		for(k=1;k<N+1;k++)
		{
			TO[k][p]=((TA[k][p]-minTO[k])/(maxTO[k]-minTO[k]));
		}
	} 

    // Randomly assign value to the weight terms

	srand(time(0));
	for(i=0;i<L+1;i++)
	{
		for(j=1;j<=M;j++)
		{
			V[i][j]=cos(rand());       // sine and cosine functions used to have the values between -1 to 1 
		}
	}
	for(j=0;j<M+1;j++)
	{
		for(k=1;k<=N;k++)
		{
			W[j][k]=sin(rand());
		}
	}


     // Starting the loop to iterate the calculations

	do     
	{
		iteration++;


		// forward pass calculation

		for(p=1;p<=P-P_Testing;p++)
		{
			IH[j][p]=0;
			for(j=1;j<M+1;j++)
			{
				for(i=1;i<L+1;i++)
				{
					IH[j][p]=IH[j][p]+(I[i][p]*V[i][j]);
				}
				IH[j][p]=IH[j][p]+(1*V[0][j]);
				OH[j][p]=1.0/(1.0+exp(-IH[j][p]));
				IH[j][p]=0;
			}
			
		}

		for(p=1;p<=P-P_Testing;p++)
		{
			IO[k][p]=0;
			for(k=1;k<N+1;k++)
			{
				for(j=1;j<M+1;j++)
				{
					IO[k][p]+=OH[j][p]*W[j][k];
				}
				IO[k][p]+=(1*W[0][k]);
				OO[k][p]=(exp(IO[k][p])-exp(-1*IO[k][p]))/(exp(IO[k][p])+exp(-1*IO[k][p]));
				IO[k][p]=0;
			}
		}


		//delWjk calculations


		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				delW[j][k]=0;
				for(p=1;p<=P-P_Testing;p++)
				{
					OH[0][p]=1.0;
					delW[j][k]=delW[j][k]+((learning_rate/P)*(TO[k][p]-OO[k][p])*(1-(OO[k][p]*OO[k][p]))*OH[j][p]);
				}
			}
		}


		//delVij calculations

		for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				delV[i][j]=0;
				for(p=1;p<=P-P_Testing;p++)
				{
					for(k=1;k<=N;k++)
					{
						I[0][p]=1.0;
						delV[i][j]=delV[i][j]+((learning_rate/(P*N))*((TO[k][p]-OO[k][p])*(1-(OO[k][p]*OO[k][p]))*W[j][k]*OH[j][p]*(1-OH[j][p])*I[i][p]));
					}
				}
			}
		}

		//error calculations

		MSE=0;
		for(p=1;p<=P-P_Testing;p++)
		{
			for(k=1;k<=N;k++)
			{
				error=pow((TO[k][p]-OO[k][p]),2)/2;
				MSE=MSE+error;
			}
		}
		MSE=MSE/(P-P_Testing);
		if(iteration>95000||iteration<5001)
            mse<<"Iteration="<<iteration<<"\t"<<"MSE="<<MSE<<endl;

		//updating Vij values

		for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				V[i][j]=V[i][j]+delV[i][j];
			}
		}

		//updating Wjk values

		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				W[j][k]=W[j][k]+delW[j][k];
			}
		}
	}while(MSE>0.0001);                                      //limiting condition applied
	



	output<<endl<<"Printing Wjk updated values"<<endl;
	
		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				output<<"W["<<j<<"]["<<k<<"]= "<<W[j][k]<<endl;
			}
		}
	output<<endl<<"Printing Vij updated values"<<endl;
	
		for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				output<<"V["<<j<<"]["<<k<<"]= "<<V[i][j]<<endl;
			}
		}
	
	
        output<<endl<<endl;     
        
        output<<"\t"<<"            Training Data "<<endl<<endl;

        output<<"Pattrern"<<"\t"<<"Input 1"<<"\t"<<"Input 2"<</*"\t\t"<<"Input 3"<<"\t\t"<<"Input 4"<<"\t\t"<<"Input 5"<<*/"\t\t"<<"Target 1"<<"\t"<<"Output 1"<<"\t"<<"Target 2"<<"\t"<<"Output 2"<<endl;
		for(p=(1);p<=P-P_Testing;p++)
		{ 
		        output<<p<<"\t\t\t";
				for(i=1;i<L+1;i++)
				{
                   output<<IA[i][p]<<"\t\t";
                }
                for(k=1;k<N+1;k++)
			    {
                   OO[k][p]=(maxTO[k]-minTO[k])*OO[k][p]+minTO[k];
                   output<<TA[k][p]<<"\t\t"<<OO[k][p]<<"\t\t";
                }
            output<<endl;
        }


	//TESTING


	// forward pass calculation
		for(p=(P-P_Testing);p<=P;p++)
		{
			IH[j][p]=0;
			for(j=1;j<M+1;j++)
			{
				for(i=1;i<L+1;i++)
				{
					IH[j][p]=IH[j][p]+(I[i][p]*V[i][j]);
				}
				IH[j][p]=IH[j][p]+(1*V[0][j]);
				OH[j][p]=1/(1+exp(-IH[j][p]));
			}
			
		}
		//output of output layer
	    err_prd=0;
        MSE_prd=0;
		for(p=(P-P_Testing);p<=P;p++)
		{
			IO[k][p]=0;
			for(k=1;k<N+1;k++)
			{
				for(j=1;j<M+1;j++)
				{
					IO[k][p]+=OH[j][p]*W[j][k];
				}
				IO[k][p]+=1.0*W[0][k];
//				OO[k][p]=1.0/(1.0+exp(-IO[k][p]));
				OO[k][p]=(exp(IO[k][p])-exp(-1*IO[k][p]))/(exp(IO[k][p])+exp(-1*IO[k][p]));
				err_prd=pow((TO[k][p]-OO[k][p]),2)/2;
				MSE_prd=MSE_prd+err_prd;
			}
		}
		MSE_prd=MSE_prd/(P_Testing);
		
		output<<endl<<"\t\t"<<"        TESTING  DATA"<<endl<<endl;
		
        output<<"Pattrern"<<"\t"<<"Input 1"<<"\t"<<"Input 2"<</*"\t\t"<<"Input 3"<<"\t\t"<<"Input 4"<<"\t\t"<<"Input 5"<<*/"\t\t"<<"Target 1"<<"\t"<<"Output 1"<<"\t"<<"Target 2"<<"\t"<<"Output 2"<<endl;
		for(p=(P-P_Testing);p<=P;p++)
		{
		        output<<p<<"\t\t\t";
				for(i=1;i<L+1;i++)
				{
                   output<<IA[i][p]<<"\t\t";
                }
                for(k=1;k<N+1;k++)
			    {
                   OO[k][p]=(maxTO[k]-minTO[k])*OO[k][p]+minTO[k];
                   output<<TA[k][p]<<"\t\t"<<OO[k][p]<<"\t\t";
                }
            output<<endl;
        }

       
        output<<endl<<"Error in prediction for learning rate="<<learning_rate<<"and number of hidden neuron="<<M<<" is "<<MSE_prd<<endl;

			 return 0;
}

	