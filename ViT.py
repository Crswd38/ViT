#################################################################################################################################################################
#
# https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb#scrollTo=yQ6MYlAfDpyI
#
#
# 터미널에 한번씩 치고 시작
# pip install timm
# pip install torch
# pip install torchvision
# pip install wget
# wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt -O ilsvrc2012_wordnet_lemmas.txt
# wget https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/santorini.png?raw=true -O santorini.png
#
##################################################################################################################################################################
# 준비하기

import os
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from timm import create_model

##################################################################################################################################################################
# 모델 및 데이터 준비

model_name = "vit_base_patch16_224"  # 모델 이름 정하기
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # CUDA 지원 GPU 사용이 가능하면 cuda, 불가능하면 cpu
print("device = ", device)
# create a ViT model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
model = create_model(model_name, pretrained=True).to(device)  # 모델을 생성하고 CPU나 CUDA로 모델을 이동한다

# 테스트용 변환 정의
IMG_SIZE = (224, 224)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
transforms = [
              T.Resize(IMG_SIZE),  # 모델이 기대하는 입력 이미지 크기와 일치하도록 이미지 크기를 조정한다
              T.ToTensor(),  # 이미지를 모델에 입력할 수 있도록 numPy 배열에서 PyTorch 텐서로 변환한다
              T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),  # 입력 이미지의 값 범위를 일정하게 조정하여 모델의 학습을 안정화시키고 성능을 향상시키도록 각 채널의 평균과 표준편차를 0.5로 이미지를 정규화한다
              ]
transforms = T.Compose(transforms)  # 변환을 하나로 결합하여 이미지에 여러 변환을 연속적으로 적용 할 수 있게 만든다

# 이미지넷(대규모 이미지 데이터베이스) 레이블
# wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt
imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))

# 데모 이미지
# wget https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/santorini.png?raw=true -O santorini.png
img = PIL.Image.open('santorini.png')
img_tensor = transforms(img).unsqueeze(0).to(device)  # 다운로드한 이미지를 이전에 정의한 변환(transform)을 사용하여 전처리한다

##################################################################################################################################################################
# 간단한 테스트

# end-to-end 추론 (시스템이 입력에서 출력까지의 전체 프로세스를 단일 시스템으로 처리하는 방식)
output = model(img_tensor)  # ViT 모델을 사용하여 전처리된 이미지에 대한 추론을 수행하고, 모델의 출력을 변수 output에 저장한다

print("Inference Result:")
print(imagenet_labels[int(torch.argmax(output))])  # 모델의 출력값 중에서 가장 큰 값의 인덱스를 구하고, 이를 ImageNet 레이블 파일에서 해당하는 클래스 이름을 찾아 출력한다
plt.imshow(img)  # 전처리된 이미지를 시각화한다. (모델이 이미지 예측을 올바르게 수행했는지 테스트)

##################################################################################################################################################################
# 비전 트랜스포머 탐구


# 이미지를 패치로 분할

patches = model.patch_embed(img_tensor)  # 입력 이미지 텐서를 ViT 모델의 패치 임베딩으로 변환한다. 이는 이미지의 각 패치를 임베딩하여 모델에 입력할 수 있도록 한다
print("Image tensor: ", img_tensor.shape)  # 이미지 텐서의 모양을 출력해서 올바르게 생성되었는지 확인한다
print("Patch embeddings: ", patches.shape)  # 패치 임베딩의 모양을 출력해서 패치 임베딩이 올바르게 생성되었는지 확인한다

# 이 코드는 파이프라인의 일부가 아니며, 시각적으로 패치를 표시하기 위한 것이다
fig = plt.figure(figsize=(8, 8))  # 새로운 Matplotlib figure를 생성한다. 이 figure의 크기는 8x8로 설정된다
fig.suptitle("Visualization of Patches", fontsize=24)  # figure의 전체 제목을 설정한다.
img = np.asarray(img)  # 이미지 numpy 변환
for i in range(0, 196):
    x = i % 14
    y = i // 14  # 현재 반복 인덱스를 14로 나눈 나머지와 몫을 계산하여 패치의 위치를 결정한다.
    patch = img[y*16:(y+1)*16, x*16:(x+1)*16]  # 이미지에서 현재 패치를 잘라낸다. 이 때 각 패치는 16x16 크기로 설정된다.
    ax = fig.add_subplot(14, 14, i+1)  # Matplotlib subplot을 추가한다.
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)  # subplot의 x 및 y 축을 숨긴다.
    ax.imshow(patch) # 현재 패치를 subplot에 이미지로 표시한다.


# 위치 임베딩 추가

pos_embed = model.pos_embed  # ViT 모델에서 위치 임베딩을 가져온다. 이는 이미 학습된 모델의 위치 임베딩을 의미한다.
print(pos_embed.shape)  # 위치 임베딩 텐서의 모양을 확인하여 크기를 파악한다.

# 위치 임베딩 유사성 시각화
# 한 셀은 임베딩과 다른 모든 임베딩 간의 코사인 유사성을 보여준다.
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)  # 코사인 유사도를 계산하기 위한 PyTorch의 CosineSimilarity 클래스를 생성한다. dim = 어느 dimension으로 cosine similarity를 구할 지 정해줌
fig = plt.figure(figsize=(8, 8))  # 새로운 Matplotlib figure를 생성한다
fig.suptitle("Visualization of position embedding similarities", fontsize=24)  # figure의 전체 제목을 설정한다
for i in range(1, pos_embed.shape[1]):
    sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)  # 각 위치 임베딩과 나머지 모든 위치 임베딩 간의 코사인 유사도를 계산한다. 이는 하나의 임베딩과 나머지 모든 임베딩 간의 유사도를 계산하는 것이다
    sim = sim.reshape((14, 14)).detach().cpu().numpy()  # 유사도를 14x14 행렬로 재구성하고, 이를 넘파이 배열로 변환한다.
    ax = fig.add_subplot(14, 14, i)  # Matplotlib subplot을 추가한다. 각 위치 임베딩의 유사도를 시각화하기 위한 작업이다
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)  # subplot의 x 및 y 축을 숨긴다.
    ax.imshow(sim)  # 유사도 행렬을 subplot에 이미지로 표시


# Transformer Input 생성

transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed
# 모델의 클래스 토큰과 패치 임베딩을 이어붙인다. 이때, dim=1은 두 텐서가 가로 방향으로 이어붙이도록 한다. 따라서 클래스 토큰과 패치 임베딩이 가로 방향으로 결합된다.
# 위치 임베딩을 이전 단계에서 생성한 결과에 더한다. 이렇게 함으로써 클래스 토큰과 패치 임베딩에 위치 정보가 추가된다.
print("Transformer input: ", transformer_input.shape) # 생성된 트랜스포머 입력의 모양을 출력한다.



print("Input tensor to Transformer (z0): ", transformer_input.shape)
x = transformer_input.clone()
for i, blk in enumerate(model.blocks):
    print("Entering the Transformer Encoder {}".format(i))
    x = blk(x)
x = model.norm(x)
transformer_output = x[:, 0]
print("Output vector from Transformer (z12-0):", transformer_output.shape)

print("Transformer Multi-head Attention block:")
attention = model.blocks[0].attn
print(attention)
print("input of the transformer encoder:", transformer_input.shape)

# fc layer to expand the dimension
transformer_input_expanded = attention.qkv(transformer_input)[0]
print("expanded to: ", transformer_input_expanded.shape)

# Split qkv into mulitple q, k, and v vectors for multi-head attantion
qkv = transformer_input_expanded.reshape(197, 3, 12, 64)  # (N=197, (qkv), H=12, D/H=64)
print("split qkv : ", qkv.shape)
q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)
print("transposed ks: ", kT.shape)

# Attention Matrix
attention_matrix = q @ kT
print("attention matrix: ", attention_matrix.shape)
plt.imshow(attention_matrix[3].detach().cpu().numpy())

# Visualize attention matrix
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Visualization of Attention", fontsize=24)
img = np.asarray(img)
ax = fig.add_subplot(2, 4, 1)
ax.imshow(img)
for i in range(7):  # visualize the 100th rows of attention matrices in the 0-7th heads
    attn_heatmap = attention_matrix[i, 100, 1:].reshape((14, 14)).detach().cpu().numpy()
    ax = fig.add_subplot(2, 4, i+2)
    ax.imshow(attn_heatmap)

print("Classification head: ", model.head)
result = model.head(transformer_output)
result_label_id = int(torch.argmax(result))
plt.plot(result.detach().cpu().numpy()[0])
plt.title("Classification result")
plt.xlabel("class id")
print("Inference result : id = {}, label name = {}".format(
    result_label_id, imagenet_labels[result_label_id]))

