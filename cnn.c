//**************************************************************************//
// Copyright (c) 2023, Duan Jingkun.
//**************************************************************************//
// FILE NAME : cnn.c
// FILE FUNCTION : Function for DSCNN
// DEPARTMENT : School of Microelectronics, SJTU
// AUTHOR : Duan Jingkun
// AUTHOR¡¯S EMAIL : djk355@sjtu.edu.cn
//**************************************************************************//
// Release history
// Time           Version          Content
// 2023.08.07     Version 1.0      Build Network
// 2023.08.10     Version 1.1      Correct BatchNormlization
// 2023.08.15     Version 1.2      Updata BatchNormlization and Linear
// 2023.08.15     Version 2.0      Load Network Coefficient
//**************************************************************************//

#include "cnn.h"

//ReLU
float ReLU(float input)
{
	float tmp;
	if(input > 0)
	{
	    tmp = input;
	}
	else
	{
	    tmp = 0;
	}
	return tmp;
}

//Convolution
void Conv(float*** img, float**** conv_w, float*** out, int IMG_SIZE_W, int IMG_SIZE_H, int IMG_SIZE_CH, int CONV_W_W, int CONV_W_H, int CONV_O_CH,int STEP)
{
	//printf("start conv\n");
	int i,j,k,r,m,n,o=0;
	float tmp1,tmp2;
	//if(IMG_SIZE_W == 96)
    /*for(i = 0; i < 32; i++)
    {
        for(j = 0; j < 1; j++)
        {
            for(k = 0; k < 10; k++)
            {
                for(m = 0; m < 4; m++)
                {
                    printf("No.%d WEIGHT = %f\n",o++,conv_w[i][j][k][m]);
                }
            }
        }
    }*/

	for(r = 0; r < CONV_O_CH; r++)//Output channel
	{
		//printf("r = %d\n",r);
		for(m = 0; m <= (IMG_SIZE_W-CONV_W_W)/STEP; m++)//Input row
		{
			for(n = 0; n <= (IMG_SIZE_H-CONV_W_H)/STEP; n++)//Input column
			{
				tmp2 = 0.0;//initial
				for(k = 0; k < IMG_SIZE_CH; k++)//Input channel
				{
					tmp1 = 0.0;//initial
					for(i = 0; i < CONV_W_W; i++)//kernel row
					{
						for(j = 0; j < CONV_W_H; j++)//kernel column
						{
							//printf("j = %d\n",j);
							tmp1 = tmp1 + img[k][i+m*STEP][j+n*STEP]*conv_w[r][k][i][j];
							//if(i == 1 && k == 0 && n == 0 && m == 0)
                            //printf("No.%f ReLU\n",conv_w[r][k][i][j]);
						}
						//if(r == 0 && m == 0 && n == 0 && IMG_SIZE_CH == 1)
                        //printf("No.%f ReLU\n",tmp1);
					}
					tmp2 = tmp2 + tmp1;
					//if(n == 0)
				}
				//Activation Function
				tmp2 = ReLU(tmp2);
                //printf("No.%f ReLU\n",tmp2);
				out[r][m][n] = tmp2;
			}
		}
	}
}

//Deepwise Convolution
void DPConv(float*** img, float*** conv_w, float*** out, int IMG_SIZE_W, int IMG_SIZE_H, int IMG_SIZE_CH, int CONV_W_W, int CONV_W_H, int STEP)
{
	//printf("start conv\n");
	int i,j,r,m,n;
	float tmp1;

	for(r = 0; r < IMG_SIZE_CH; r++)//Output channel
	{
		for(m = 0; m <= (IMG_SIZE_W-CONV_W_W)/STEP; m++)//Input row
		{
			for(n = 0; n <= (IMG_SIZE_H-CONV_W_H)/STEP; n++)//Input column
			{
                tmp1 = 0.0;//initial
                for(i = 0; i < CONV_W_W; i++)//kernel row
                {
                    for(j = 0; j < CONV_W_H; j++)//kernel row
                    {
                        //printf("j = %d\n",j);
                        tmp1 = tmp1 + img[r][i+m*STEP][j+n*STEP]*conv_w[r][i][j];
                        //if(i == 1 && k == 0 && n == 0 && m == 0)
                        //printf("No.%f ReLU\n",conv_w[r][k][i][j]);
                    }
                }
				//Activation Function
				tmp1 = ReLU(tmp1);
                //printf("No.%f ReLU\n",tmp2);
				out[r][m][n] = tmp1;
			}
		}
	}
}

//Batch Normalization
void BatchNorm(float*** in, float* weight, float*** out,int IN_W, int IN_H, int CH)
{
	int i,j,k;

	for(k = 0; k < CH; k+=1)
	{
		/*mean = 0;
		var = 0;
		for(i = 0; i < IN_W; i+=1)//row
		{
			for(j = 0; j < IN_H; j+=1)//column
			{
				mean = mean + in[k][i][j];
			}
		}
		mean = mean/(IN_W*IN_H);
		for(i = 0; i < IN_W; i+=1)//row
		{
			for(j = 0; j < IN_H; j+=1)//column
			{
				var = var + (in[k][i][j] - mean) * (in[k][i][j] - mean);
			}
		}
		var = sqrt(var/(IN_W*IN_H));*/
		for(i = 0; i < IN_W; i+=1)//row
		{
			for(j = 0; j < IN_H; j+=1)//column
			{
				out[k][i][j] = in[k][i][j] * weight[k] + weight[CH + k];
				//if(i == 0 && j == 0 && k == 0)
				//printf("in = %f, w = %f, bia = %f\n",in[k][i][j],weight[k], weight[CH + k]);
			}
		}
	}
}

//Max Pooling
void Maxpool(float*** in, float*** out,int IN_W, int IN_H, int CH)
{
	//printf("start pooling\n");
	int i,j,k;
	float tmp1,tmp2,tmp3;

	//Pooling
	for(k = 0; k < CH; k+=1)
	{
		for(i = 0; i < IN_W; i+=2)//row
		{
			for(j = 0; j < IN_H; j+=2)//column
			{
                //Max
				tmp1 = in[k][i][j] > in[k][i][j+1] ? in[k][i][j] : in[k][i][j+1];
				tmp2 = in[k][i+1][j] > in[k][i+1][j+1] ? in[k][i+1][j] : in[k][i+1][j+1];
				tmp3 = tmp1 > tmp2 ? tmp1 : tmp2;

				out[k][i/2][j/2] = tmp3;
				//printf("tmp1 = %f\n",tmp3);
			}
		}
	}
}

//Flatten
void Flatten(float*** in, float* out, int IN_W, int IN_H, int IN_CH)
{
	int i,j,k;
	float tmp;

	for(i = 0; i < IN_CH; i++)
	{
		for(j = 0; j < IN_W; j++)
		{
			for(k = 0; k < IN_H; k++)
			{
				tmp = in[i][j][k];
				out[i * IN_H * IN_W + j * IN_H + k] = tmp;
			}
		}
	}
}

//Linear
void Linear(float* in, float** w, float* bia, float* out, int IN_LN, int OUT_LN)
{
	int i,j;
	float tmp;

	for(i = 0; i < OUT_LN; i++)
	{
		tmp = 0;
		for(j = 0; j < IN_LN; j++)
		{
			tmp = tmp + in[j]*w[j][i];
			//printf("in[%d] = %f,w[%d][%d] = %f,tmp[%d] = %f,mux = %f\n",j,in[j],j,i,w[j][i],i,tmp,in[j]*w[j][i]);
		}
		tmp += bia[i];
		out[i] = tmp;
	}
}
