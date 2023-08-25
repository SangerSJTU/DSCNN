//**************************************************************************//
// Copyright (c) 2023, Duan Jingkun.
//**************************************************************************//
// FILE NAME : main.c
// FILE FUNCTION : A Depthwise Separable Convolutional Neural Network
// DEPARTMENT : School of Microelectronics, SJTU
// AUTHOR : Duan Jingkun
// AUTHOR��S EMAIL : djk355@sjtu.edu.cn
//**************************************************************************//
// Release history
// Time           Version          Content
// 2023.08.07     Version 1.0      Build Network
// 2023.08.10     Version 1.1      Correct BatchNormlization
// 2023.08.15     Version 1.2      Updata BatchNormlization and Linear
// 2023.08.15     Version 2.0      Load Network Coefficient
//**************************************************************************//

#include "cnn.h"

int main()
{
    float** img;//Input data
    float*** tmp1;//Template variable
    float*** tmp2;//Template variable
    float**** conv_w1;//First Conv Layer kernal
    float*** conv_w2;//Second Conv Layer kernal
    float**** conv_w3;//Third Conv Layer kernal
    float*** conv_w4;//Forth Conv Layer kernal
    float**** conv_w5;//Fifth Conv Layer kernal
    float** w;//Linear Layer kernal
    float* bia;//Linear Layer bias
    float* tmp3;//Template variable
    float* output;//Output result
    float* in = (float*)malloc(IMG_CH*IMG_H*IMG_W * sizeof(float));//Input file data
    float* co = (float*)malloc(K2_NUM*K2_H*K2_W * sizeof(float));//Conv coefficient file data
    float* bn = (float*)malloc(2*K1_NUM * sizeof(float));//BatchNormalization coefficient file data
    float* fcn = (float*)malloc(609*2 * sizeof(float));//Linear coefficient file data
    float max = 0;
    int H,W;
    int i,j,k,m;
    FILE *input = NULL;//input file
    FILE *coeff = NULL;//Conv coefficient file
    FILE *bneff = NULL;//BatchNormalization coefficient file
    FILE *fcneff = NULL;//Linear coefficient file

//***************************************************************//
//***************************************************************//
//***********************Input Layer*****************************//
//***************************************************************//
//***************************************************************//
    //input
    FILE* out = fopen("11.txt", "w");
    input = fopen("C:/Users/admin/Desktop/NN/DSCNN/7.txt", "r");
    for(i = 0; i < IMG_CH*IMG_H*IMG_W; i++)
    {
        fscanf(input, "%f", &in[i]);
        //printf("img[%d] = %f\n",i,in[i]);
    }
    img = (float **)malloc(IMG_H * sizeof(float *));
	for (i = 0; i < IMG_H; i++)
	{
		img[i] = (float *)malloc(IMG_W * sizeof(float));
		for(j = 0; j < IMG_W; j++)
		{
		    //img[i][j] = i + j;
		    //fscanf(input, "%f", &img[i][j]);
		    img[i][j] = in[i*IMG_W + j];
		    //printf("img[%d][%d] = %f\n",i,j,img[i][j]);
		}
	}
	free(in);
	in = NULL;
    tmp1 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp1[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp1[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    tmp2 = (float ***)malloc(IMG_CH * sizeof(float **));
    for(i = 0; i < IMG_CH; i++)
    {
        tmp2[i] = (float **)malloc(IMG_H * sizeof(float *));
        for(j = 0; j < IMG_H; j++)
        {
            tmp2[i][j] = (float *)malloc(IMG_W * sizeof(float));
        }
    }

    for(i = 0; i < IMG_H; i++)
    {
        for(j = 0; j < IMG_W; j++)
        {
            tmp2[0][i][j] = img[i][j];
            //printf("img[%d][%d] = %f\n",i,j,tmp2[0][i][j]);
        }
    }
    printf("Img done\n");

//***************************************************************//
//***************************************************************//
//*******************First Conv Layer****************************//
//***************************************************************//
//***************************************************************//
    //conv1
    coeff = fopen("conv1_para.txt", "r");
    for(i = 0; i < K2_NUM*K2_H*K2_W; i++)
    {
        fscanf(coeff, "%f", &co[i]);
        //printf("coeff[%d] = %f\n",i,co[i]);
    }
    bneff = fopen("bn1_para.txt", "r");
    for(i = 0; i < 2*K1_NUM; i++)
    {
        fscanf(bneff, "%f", &bn[i]);
        //printf("coeff[%d] = %f\n",i,bn[i]);
    }
    conv_w1 = (float ****)malloc(K1_NUM * sizeof(float ***));
    for(i = 0; i < K1_NUM; i++)
    {
        conv_w1[i] = (float ***)malloc(IMG_CH * sizeof(float **));
        for(j = 0; j < IMG_CH; j++)
        {
            conv_w1[i][j] = (float **)malloc(K1_H * sizeof(float *));
            for(k = 0; k < K1_H; k++)
            {
                conv_w1[i][j][k] = (float *)malloc(K1_W * sizeof(float));
                for(m = 0; m < K1_W; m++)
                {
                    conv_w1[i][j][k][m] = co[i*IMG_CH*K1_W*K1_H + j*K1_H*K1_W + k*K1_W + m];
                    //printf("WEIGHT = %f\n",conv_w1[i][j][k][m]);
                }
            }
        }
    }
    Conv(tmp2, conv_w1, tmp1, IMG_H, IMG_W, IMG_CH, K1_H, K1_W, K1_NUM, K1_STEP);
    for (i = 0; i < IMG_CH; i++)
    {
		for(j = 0; j < IMG_H; j++)
		{
		    free(tmp2[i][j]);
		}
		free(tmp2[i]);
    }
	free(tmp2);
	tmp2 = NULL;
    tmp2 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp2[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp2[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < IMG_CH; j++)
		{
		    for(k = 0; k < K1_H; k ++)
		    {
		        free(conv_w1[i][j][k]);
		    }
		    free(conv_w1[i][j]);
		}
		free(conv_w1[i]);
    }
	free(conv_w1);
	conv_w1 = NULL;
    for(i = 0; i < K1_NUM; i++)
    {
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            for(k = 0; k < ((IMG_W - K1_W)/K1_STEP + 1); k++)
            {
                //fprintf(out, "%f\n",tmp2[i][j][k]);
                //tmp2[i][j][k] /= pow(2,K1);
                //tmp2[i][j][k] = floor(tmp2[i][j][k] + 0.5);
                if(tmp2[i][j][k] > max)
                max = tmp2[i][j][k];
                //printf("tmp1[%d][%d][%d] = %f\n",i,j,k,tmp1[i][j][k]);
                //fprintf(out, "%f\n",tmp1[i][j][k]);
            }
        }
    }
	//printf("data = %d\n",done);
    BatchNorm(tmp1, bn, tmp2, ((IMG_H - K1_H)/K1_STEP + 1), ((IMG_W - K1_W)/K1_STEP + 1), K1_NUM);
    for(i = 0; i < K1_NUM; i++)
    {
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            for(k = 0; k < ((IMG_W - K1_W)/K1_STEP + 1); k++)
            {
                fprintf(out, "%f\n",tmp2[i][j][k]);
                //printf("tmp1[%d][%d][%d] = %f\n",i,j,k,tmp2[i][j][k]);
                //tmp2[i][j][k] /= pow(2,K1);
                //tmp2[i][j][k] = floor(tmp2[i][j][k] + 0.5);
                if(tmp2[i][j][k] > max)
                max = tmp2[i][j][k];
                //printf("tmp1[%d][%d][%d] = %f\n",i,j,k,tmp2[i][j][k]);
                //fprintf(out, "%f\n",tmp2[i][j][k]);
            }
        }
    }
    printf("max = %f\n",max);
    printf("Conv1 done\n");

//***************************************************************//
//***************************************************************//
//*******************Second Conv Layer***************************//
//***************************************************************//
//***************************************************************//
    //conv2
    coeff = fopen("conv2_para.txt", "r");
    for(i = 0; i < K2_NUM*K2_H*K2_W; i++)
    {
        fscanf(coeff, "%f", &co[i]);
        //printf("coeff[%d] = %f\n",i,co[i]);
    }
    bneff = fopen("bn2_para.txt", "r");
    for(i = 0; i < 2*K1_NUM; i++)
    {
        fscanf(bneff, "%f", &bn[i]);
        //printf("coeff[%d] = %f\n",i,bn[i]);
    }
    conv_w2 = (float ***)malloc(K2_NUM * sizeof(float **));
    for(i = 0; i < K2_NUM; i++)
    {
        conv_w2[i] = (float **)malloc(K2_H * sizeof(float *));
        for(j = 0; j < K2_H; j++)
        {
            conv_w2[i][j] = (float *)malloc(K2_W * sizeof(float));
            for(k = 0; k < K2_W; k++)
            {
                conv_w2[i][j][k] = co[i*K2_H*K2_W + j*K2_W +k];
                //printf("No.%d WEIGHT = %f\n",i,conv_w2[i][j][k]);
            }
        }
    }
    DPConv(tmp2, conv_w2, tmp1, ((IMG_H - K1_H)/K1_STEP + 1), ((IMG_W - K1_W)/K1_STEP + 1), K1_NUM, K2_H, K2_W, K2_STEP);
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    free(tmp2[i][j]);
		}
		free(tmp2[i]);
    }
	free(tmp2);
	tmp2 = NULL;
    tmp2 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp2[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp2[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    H = ((IMG_H - K1_H)/K1_STEP + 1 - K2_H)/K2_STEP + 1;
    W = ((IMG_W - K1_W)/K1_STEP + 1 - K2_W)/K2_STEP + 1;
    BatchNorm(tmp1, bn, tmp2, H, W, K2_NUM);
    for (i = 0; i < K2_NUM; i++)
    {
		for(j = 0; j < K2_H; j++)
		{
		    free(conv_w2[i][j]);
		}
		free(conv_w2[i]);
    }
	free(conv_w2);
	conv_w2 = NULL;
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    free(tmp1[i][j]);
		}
		free(tmp1[i]);
    }
	free(tmp1);
    tmp1 = NULL;
    max = 0;
    for(i = 0; i < K2_NUM; i++)
    {
        for(j = 0; j < H; j++)
        {
            for(k = 0; k < W; k++)
            {
                //tmp2[i][j][k] /= pow(2,K2);
                //tmp2[i][j][k] = floor(tmp2[i][j][k] + 0.5);
                if(tmp2[i][j][k] > max)
                max = tmp2[i][j][k];
                //printf("tmp1[%d][%d][%d] = %f\n",i,j,k,tmp2[i][j][k]);
            }
        }
    }
    printf("max = %f\n",max);
    printf("Conv2 done\n");

//***************************************************************//
//***************************************************************//
//*******************Third Conv Layer****************************//
//***************************************************************//
//***************************************************************//
    //conv3
    coeff = fopen("conv3_para.txt", "r");
    for(i = 0; i < K2_NUM*K2_H*K2_W; i++)
    {
        fscanf(coeff, "%f", &co[i]);
        //printf("coeff[%d] = %f\n",i,co[i]);
    }
    bneff = fopen("bn3_para.txt", "r");
    for(i = 0; i < 2*K1_NUM; i++)
    {
        fscanf(bneff, "%f", &bn[i]);
        //printf("coeff[%d] = %f\n",i,bn[i]);
    }
    conv_w3 = (float ****)malloc(K3_NUM * sizeof(float ***));
    for(i = 0; i < K3_NUM; i++)
    {
        conv_w3[i] = (float ***)malloc(K2_NUM * sizeof(float **));
        for(j = 0; j < K2_NUM; j++)
        {
            conv_w3[i][j] = (float **)malloc(K3_H * sizeof(float *));
            for(k = 0; k < K3_H; k++)
            {
                conv_w3[i][j][k] = (float *)malloc(K3_W * sizeof(float));
                for(m = 0; m < K3_W; m++)
                {
                    conv_w3[i][j][k][m] = co[i*K2_NUM*K3_H*K3_W + j*K3_H*K3_W + k*K3_W + m];
                    //printf("No.%d WEIGHT = %f\n",i,conv_w3[i][j][k][m]);
                }
            }
        }
    }
    tmp1 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp1[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp1[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    Conv(tmp2, conv_w3, tmp1, H, W, K2_NUM, K3_H, K3_W, K3_NUM, K3_STEP);
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    free(tmp2[i][j]);
		}
		free(tmp2[i]);
    }
	free(tmp2);
	tmp2 = NULL;
    tmp2 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp2[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp2[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    H = (H - K3_H)/K3_STEP + 1;
    W = (W - K3_W)/K3_STEP + 1;
    BatchNorm(tmp1, bn, tmp2, H, W, K3_NUM);
    for (i = 0; i < K3_NUM; i++)
    {
		for(j = 0; j < K2_NUM; j++)
		{
		    for(k = 0; k < K3_H; k ++)
		    {
		        free(conv_w3[i][j][k]);
		    }
		    free(conv_w3[i][j]);
		}
		free(conv_w3[i]);
    }
	free(conv_w3);
	conv_w3 = NULL;
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    free(tmp1[i][j]);
		}
		free(tmp1[i]);
    }
	free(tmp1);
	tmp1 = NULL;
    max = 0;
    for(i = 0; i < K3_NUM; i++)
    {
        for(j = 0; j < H; j++)
        {
            for(k = 0; k < W; k++)
            {
                //tmp2[i][j][k] /= pow(2,K3);
                //tmp2[i][j][k] = floor(tmp2[i][j][k] + 0.5);
                if(tmp2[i][j][k] > max)
                max = tmp2[i][j][k];
                //printf("tmp1[%d][%d][%d] = %f\n",i,j,k,tmp2[i][j][k]);
            }
        }
    }
    printf("max = %f\n",max);
    printf("Conv3 done\n");

//***************************************************************//
//***************************************************************//
//*******************Forth Conv Layer****************************//
//***************************************************************//
//***************************************************************//
    //conv4
    coeff = fopen("conv4_para.txt", "r");
    for(i = 0; i < K2_NUM*K2_H*K2_W; i++)
    {
        fscanf(coeff, "%f", &co[i]);
        //printf("coeff[%d] = %f\n",i,co[i]);
    }
    bneff = fopen("bn4_para.txt", "r");
    for(i = 0; i < 2*K1_NUM; i++)
    {
        fscanf(bneff, "%f", &bn[i]);
        //printf("coeff[%d] = %f\n",i,bn[i]);
    }
    conv_w4 = (float ***)malloc(K4_NUM * sizeof(float **));
    for(i = 0; i < K4_NUM; i++)
    {
        conv_w4[i] = (float **)malloc(K4_H * sizeof(float *));
        for(j = 0; j < K4_H; j++)
        {
            conv_w4[i][j] = (float *)malloc(K4_W * sizeof(float));
            for(k = 0; k < K4_W; k++)
            {
                conv_w4[i][j][k] = co[i*K4_H*K4_W + j*K4_W + k];
                //printf("No.%d WEIGHT = %f\n",i,conv_w4[i][j][k]);
            }
        }
    }
    tmp1 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp1[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp1[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    DPConv(tmp2, conv_w4, tmp1, H, W, K3_NUM, K4_H, K4_W, K4_STEP);
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    free(tmp2[i][j]);
		}
		free(tmp2[i]);
    }
	free(tmp2);
	tmp2 = NULL;
    tmp2 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp2[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp2[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    H = (H - K4_H)/K4_STEP + 1;
    W = (W - K4_W)/K4_STEP + 1;
    BatchNorm(tmp1, bn, tmp2, H, W, K4_NUM);
    for (i = 0; i < K4_NUM; i++)
    {
		for(j = 0; j < K4_H; j++)
		{
		    free(conv_w4[i][j]);
		}
		free(conv_w4[i]);
    }
	free(conv_w4);
	conv_w4 = NULL;
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    free(tmp1[i][j]);
		}
		free(tmp1[i]);
    }
	free(tmp1);
	tmp1 = NULL;
    max = 0;
    for(i = 0; i < K4_NUM; i++)
    {
        for(j = 0; j < H; j++)
        {
            for(k = 0; k < W; k++)
            {
                //tmp2[i][j][k] /= pow(2,K4);
                //tmp2[i][j][k] = floor(tmp2[i][j][k] + 0.5);
                if(tmp2[i][j][k] > max)
                max = tmp2[i][j][k];
                //printf("tmp1[%d][%d][%d] = %f\n",i,j,k,tmp2[i][j][k]);
            }
        }
    }
    printf("max = %f\n",max);
    printf("Conv4 done\n");

//***************************************************************//
//***************************************************************//
//*******************Fifth Conv Layer****************************//
//***************************************************************//
//***************************************************************//
    //conv5
    coeff = fopen("conv5_para.txt", "r");
    for(i = 0; i < K2_NUM*K2_H*K2_W; i++)
    {
        fscanf(coeff, "%f", &co[i]);
        //printf("coeff[%d] = %f\n",i,co[i]);
    }
    bneff = fopen("bn5_para.txt", "r");
    for(i = 0; i < 2*K1_NUM; i++)
    {
        fscanf(bneff, "%f", &bn[i]);
        //printf("coeff[%d] = %f\n",i,bn[i]);
    }
    conv_w5 = (float ****)malloc(K5_NUM * sizeof(float ***));
    for(i = 0; i < K5_NUM; i++)
    {
        conv_w5[i] = (float ***)malloc(K4_NUM * sizeof(float **));
        for(j = 0; j < K4_NUM; j++)
        {
            conv_w5[i][j] = (float **)malloc(K5_H * sizeof(float *));
            for(k = 0; k < K5_H; k++)
            {
                conv_w5[i][j][k] = (float *)malloc(K5_W * sizeof(float));
                for(m = 0; m < K5_W; m++)
                {
                    conv_w5[i][j][k][m] = co[i*K4_NUM*K5_H*K5_W + j*K5_H*K5_W + k*K5_W + m];
                    //printf("No.%d WEIGHT = %f\n",i,conv_w5[i][j][k][m]);
                }
            }
        }
    }
    tmp1 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp1[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp1[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    Conv(tmp2, conv_w5, tmp1, H, W, K4_NUM, K5_H, K5_W, K5_NUM, K5_STEP);
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    free(tmp2[i][j]);
		}
		free(tmp2[i]);
    }
	free(tmp2);
	tmp2 = NULL;
    tmp2 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp2[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp2[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    H = (H - K5_H)/K5_STEP + 1;
    W = (W - K5_W)/K5_STEP + 1;
    BatchNorm(tmp1, bn, tmp2, H, W, K5_NUM);
    for (i = 0; i < K5_NUM; i++)
    {
		for(j = 0; j < K4_NUM; j++)
		{
		    for(k = 0; k < K5_H; k ++)
		    {
		        free(conv_w5[i][j][k]);
		    }
		    free(conv_w5[i][j]);
		}
		free(conv_w5[i]);
    }
	free(conv_w5);
	conv_w5 = NULL;
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    free(tmp1[i][j]);
		}
		free(tmp1[i]);
    }
	free(tmp1);
	tmp1 = NULL;
	free(co);
	co = NULL;
	free(bn);
	bn = NULL;
    max = 0;
    for(i = 0; i < K5_NUM; i++)
    {
        for(j = 0; j < H; j++)
        {
            for(k = 0; k < W; k++)
            {
                //tmp2[i][j][k] /= pow(2,K5);
                //tmp2[i][j][k] = floor(tmp2[i][j][k] + 0.5);
                if(tmp2[i][j][k] > max)
                max = tmp2[i][j][k];
                //printf("tmp1[%d][%d][%d] = %f\n",i,j,k,tmp2[i][j][k]);
            }
        }
    }
    printf("max = %f\n",max);
    printf("Conv5 done\n");

//***************************************************************//
//***************************************************************//
//**********************Pooling Layer****************************//
//***************************************************************//
//***************************************************************//
    //pool
    tmp1 = (float ***)malloc(K1_NUM * sizeof(float **));
    for(i = 0; i < K1_NUM; i++)
    {
        tmp1[i] = (float **)malloc(((IMG_H - K1_H)/K1_STEP + 1) * sizeof(float *));
        for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
        {
            tmp1[i][j] = (float *)malloc(((IMG_W - K1_W)/K1_STEP + 1) * sizeof(float));
        }
    }
    Maxpool(tmp2, tmp1, H, W, K5_NUM);
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    if(tmp2[i][j] != NULL)
		    free(tmp2[i][j]);
		}
		if(tmp2[i] != NULL)
		free(tmp2[i]);
		//printf("Free %d\n",i);
    }
    if(tmp2 != NULL)
	free(tmp2);
	tmp2 = NULL;
    max = 0;
    for(i = 0; i < K5_NUM; i++)
    {
        for(j = 0; j < H/POOL_STEP; j++)
        {
            for(k = 0; k < W/POOL_STEP; k++)
            {
                //tmp2[i][j][k] /= pow(2,K5);
                if(tmp1[i][j][k] > max)
                max = tmp1[i][j][k];
                //printf("tmp1[%d][%d][%d] = %f\n",i,j,k,tmp1[i][j][k]);
            }
        }
    }
    printf("max = %f\n",max);
    printf("Pooling done\n");

//***************************************************************//
//***************************************************************//
//**********************Flatten Layer****************************//
//***************************************************************//
//***************************************************************//
    //Flatten
    H = H / POOL_STEP;
    W = W / POOL_STEP;
    tmp3 = (float *)malloc(H*W*K5_NUM * sizeof(float));
    Flatten(tmp1, tmp3, H, W, K5_NUM);
    for (i = 0; i < K1_NUM; i++)
    {
		for(j = 0; j < ((IMG_H - K1_H)/K1_STEP + 1); j++)
		{
		    if(tmp1[i] != NULL)
		    free(tmp1[i][j]);
		    //printf("Free %d\n",j);
		}
		if(tmp1[i] != NULL)
		free(tmp1[i]);
		//printf("Free %d\n",i);
    }
    if(tmp1 != NULL)
	free(tmp1);
	tmp1 = NULL;
    /*for(i = 0; i < 32; i++)
    {
        for(j = 0; j < 19; j++)
        {
            printf("img[%d][%d] = %f,%f\n",i,j,tmp1[i][j][0],tmp3[i*19 + j]);
        }
    }*/
    max = 0;
    for(i = 0; i < H*W*K5_NUM; i++)
    {
        if(tmp3[i] > max)
        max = tmp3[i];
    }
    printf("max = %f\n",max);
    printf("Flatten done\n");

//***************************************************************//
//***************************************************************//
//***********************Linear Layer****************************//
//***************************************************************//
//***************************************************************//
    //Linear
    fcneff = fopen("fcn_para.txt", "r");
    for(i = 0; i < (H*W*K5_NUM + 1)*OUT_LEN; i++)
    {
        fscanf(fcneff, "%f", &fcn[i]);
        //printf("coeff[%d] = %f\n",i,fcn[i]);
    }
    w = (float **)malloc(H*W*K5_NUM * sizeof(float *));
	for (i = 0; i < H*W*K5_NUM; i++)
	{
		w[i] = (float *)malloc(OUT_LEN * sizeof(float));
		for(j = 0; j < OUT_LEN; j++)
		{
		    w[i][j] = fcn[i + j * H*W*K5_NUM];
		}
	}
	bia = (float *)malloc(OUT_LEN * sizeof(float));
	for(i = 0; i < OUT_LEN; i++)
	{
	    bia[i] = fcn[i + OUT_LEN * H*W*K5_NUM];
	}
    output = (float *)malloc(OUT_LEN * sizeof(float));
    Linear(tmp3, w, bia, output, H*W*K5_NUM, OUT_LEN);
    for (i = 0; i < H*W*K5_NUM; i++)
    {
		if(w[i] != NULL)
		free(w[i]);
    }
	if(w != NULL)
	free(w);
	w = NULL;
	if(tmp3 != NULL)
	free(tmp3);
	tmp3 = NULL;
	free(bia);
	bia = NULL;
	free(fcn);
	fcn = NULL;
    /*for(i = 0; i < 608; i++)
    {
        for(j = 0; j < 1; j++)
        {
            printf("img[%d][%d] = %f\n",i,j,tmp3[i]);
        }
    }*/
    printf("Linear done\n");

//***************************************************************//
//***************************************************************//
//***********************Output Layer****************************//
//***************************************************************//
//***************************************************************//
    //output
    if(output[0] < output[1])
    printf("Current Voice is 'Hi Verisilicon'\n");
    //printf("Result = [%f,%f]\n\n",fabs(output[0])/(fabs(output[0]) + fabs(output[1])),fabs(output[1])/(fabs(output[0]) + fabs(output[1])));
    printf("Result = [%f,%f]\n\n",output[0],output[1]);

    //post process
    for (i = 0; i < IMG_H; i++)
    {
		if(img[i] != NULL)
		free(img[i]);
    }
    if(img != NULL)
	free(img);
	img = NULL;
	if(output != NULL)
    free(output);
    output = NULL;

    fclose(input);
    fclose(coeff);
    fclose(bneff);
    fclose(fcneff);

    return 0;
}
