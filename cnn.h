//**************************************************************************//
// Copyright (c) 2023, Duan Jingkun.
//**************************************************************************//
// FILE NAME : cnn.h
// FILE FUNCTION : Head File for DSCNN
// DEPARTMENT : School of Microelectronics, SJTU
// AUTHOR : Duan Jingkun
// AUTHOR¡ÇS EMAIL : djk355@sjtu.edu.cn
//**************************************************************************//
// Release history
// Time           Version          Content
// 2023.08.07     Version 1.0      Build Network
// 2023.08.10     Version 1.1      Correct BatchNormlization
// 2023.08.15     Version 1.2      Updata BatchNormlization and Linear
// 2023.08.15     Version 2.0      Load Network Coefficient
//**************************************************************************//

#ifndef CNN_H
#define CNN_H

//Include Library
#include<math.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

//Macro Definition
//************INPUT****************//
#define IMG_W 22 //Input column
#define IMG_H 200 //Input row
#define IMG_CH 1 //Input channel

//************LAYER 1**************//
#define K1_W 4 //Kernel 1 colunm
#define K1_H 10 //Kernel 1 row
#define K1_NUM 32 //Kernel 1 number
#define K1_STEP 2 //Conv 1 step
#define K1 16 //Quan Coeff

//************LAYER 2**************//
#define K2_W 4 //Kernel 2 column
#define K2_H 18 //Kernel 2 row
#define K2_NUM 32 //Kernel 2 number
#define K2_STEP 2 //Conv 2 step
#define K2 16 //Quan Coeff

//************LAYER 3**************//
#define K3_W 1 //Kernel 3 column
#define K3_H 1 //Kernel 3 row
#define K3_NUM 32 //Kernel 3 number
#define K3_STEP 1 //Conv 3 step
#define K3 14 //Quan Coeff

//************LAYER 4**************//
#define K4_W 3 //Kernel 4 column
#define K4_H 3 //Kernel 4 row
#define K4_NUM 32 //Kernel 4 number
#define K4_STEP 1 //Conv 4 step
#define K4 15 //Quan Coeff

//************LAYER 5**************//
#define K5_W 1 //Kernel 5 column
#define K5_H 1 //Kernel 5 row
#define K5_NUM 32 //Kernel 5 number
#define K5_STEP 1 //Conv 5 number
#define K5 13 //Quan Coeff

//***********POOLING***************//
#define POOL_STEP 2 //Pooling step

//************OUTPUT***************//
#define OUT_LEN 2 //Output length

//Function Declaration
//*******************************Activation Function**********************************//
//************************************************************************************//
//                                 / x  ,  if x > 0
//                       ReLU(x) =
//                                 \ 0  ,  else
//************************************************************************************//
//************************************************************************************//
float ReLU(float input);

//**********************************2D Convolution************************************//
//************************************************************************************//
//      -------                                                -----
//     |       |                    --                        |     |
//     |       |         *         |  |           =           |     |
//     |       |                    --                        |     |
//      -------                                                -----
//       Input                    kernel                       Output
//************************************************************************************//
//Input:IMG_SIZE_W * IMG_SIZE_H * IMG_SIZE_CH
//kernel:CONV_W_W *CONV_W_H * IMG_SIZE_CH * CONV_O_CH
//Output:((IMG_SIZE_W-CONV_W_W)/STEP+1) * ((IMG_SIZE_H-CONV_W_H)/STEP+1) * CONV_O_CH
//************************************************************************************//
//************************************************************************************//
void Conv(float*** img, float**** conv_w, float*** out, int IMG_SIZE_W, int IMG_SIZE_H, int IMG_SIZE_CH, int CONV_W_W, int CONV_W_H, int CONV_O_CH,int STEP);

//*****************************Deepwise 2D Convolution********************************//
//************************************************************************************//
//      -------                                                -----
//     |       |                    --                        |     |
//     |       |         *         |  |           =           |     |
//     |       |                    --                        |     |
//      -------                                                -----
//       Input                    kernel                       Output
//************************************************************************************//
//Input:IMG_SIZE_W * IMG_SIZE_H * IMG_SIZE_CH
//kernel:CONV_W_W *CONV_W_H * IMG_SIZE_CH
//Output:((IMG_SIZE_W-CONV_W_W)/STEP+1) * ((IMG_SIZE_H-CONV_W_H)/STEP+1) * IMG_SIZE_CH
//************************************************************************************//
//************************************************************************************//
void DPConv(float*** img, float*** conv_w, float*** out, int IMG_SIZE_W, int IMG_SIZE_H, int IMG_SIZE_CH, int CONV_W_W, int CONV_W_H, int STEP);

//******************************2D Batch Normalization********************************//
//************************************************************************************//
//                                 x - mean(x)
//                       BN(x) =  ------------
//                                  stdvar(x)
//************************************************************************************//
//************************************************************************************//
void BatchNorm(float*** in, float* weight, float*** out,int IN_W, int IN_H, int CH);

//************************************Max Pooling*************************************//
//************************************************************************************//
//         -- --
//        | a| b|                            --
//         -- --    =>   Max Pooling   =>   | x|     x = max(a, b, c, d)
//        | c| d|                            --
//         -- --
//************************************************************************************//
//************************************************************************************//
void Maxpool(float*** in, float*** out,int IN_W, int IN_H, int CH);

//**************************************Flatten***************************************//
//************************************************************************************//
//      -------
//     |       |                              ------------------------
//     |       |     =>    Flatten    =>     |                        |
//     |       |                              ------------------------
//      -------
//************************************************************************************//
//************************************************************************************//
void Flatten(float*** in, float* out, int IN_W, int IN_H, int IN_CH);

//***************************************Linear***************************************//
//************************************************************************************//
//                                      -----
//    ------------------------         |     |           ----
//   |                        |   *    |     |    =     |    |
//    ------------------------         |     |           ----
//                                      -----
//************************************************************************************//
//************************************************************************************//
void Linear(float* in, float** w, float* bia, float* out, int IN_LN, int OUT_LN);

#endif
