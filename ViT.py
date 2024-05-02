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
plt.show()


##################################################################################################################################################################
# 비전 트랜스포머 탐구

##################################################################################################################################################################
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
    try:
        ax.imshow(patch) # 현재 패치를 subplot에 이미지로 표시한다.
    except:
        pass
plt.show()

##################################################################################################################################################################
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
plt.show()

# 트랜스포머 Input 생성
transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed
# 모델의 클래스 토큰과 패치 임베딩을 이어붙인다. 이때, dim=1은 두 텐서가 가로 방향으로 이어붙이도록 한다. 따라서 클래스 토큰과 패치 임베딩이 가로 방향으로 결합된다.
# 위치 임베딩을 이전 단계에서 생성한 결과에 더한다. 이렇게 함으로써 클래스 토큰과 패치 임베딩에 위치 정보가 추가된다.
print("Transformer input: ", transformer_input.shape) # 생성된 트랜스포머 입력의 모양을 출력한다.

##################################################################################################################################################################
# 트랜스포머 인코더

# 연속된 트랜스포머 인코더들
print("Input tensor to Transformer (z0): ", transformer_input.shape)  # 트랜스포머의 입력으로 사용될 텐서의 모양을 확인한다.
x = transformer_input.clone()  # 입력 텐서를 복제하여 트랜스포머 인코더에 입력으로 사용할 변수 x를 정의한다.
for i, blk in enumerate(model.blocks):  # ViT 모델의 각 블록(트랜스포머 인코더)에 대해 반복한다. enumerate() 함수는 블록의 인덱스와 블록 자체를 순회한다.
    print("Entering the Transformer Encoder {}".format(i))
    x = blk(x)  # 현재 블록에 입력 텐서 x를 전달하여 트랜스포머 인코더를 거친다.
x = model.norm(x)  # 모델의 LayerNorm을 통해 출력 텐서를 정규화한다.
transformer_output = x[:, 0]  # 트랜스포머의 출력으로 사용될 변수 transformer_output을 정의한다. 이는 트랜스포머의 출력 중 첫 번째 위치에 있는 벡터다.
print("Output vector from Transformer (z12-0):", transformer_output.shape)  # 트랜스포머의 출력 벡터의 모양을 출력한다.

#어텐션 작동 방식
print("Transformer Multi-head Attention block:")
attention = model.blocks[0].attn  # 모델의 첫 번째 블록에 있는 어텐션 블록을 attention 변수에 할당한다.
print(attention)  # 어텐션 블록을 출력한다. 이는 어텐션 메커니즘의 세부 구현에 대한 정보를 제공한다.
print("input of the transformer encoder:", transformer_input.shape)  # 트랜스포머 인코더에 입력되는 텐서의 모양을 출력합니다. 이는 트랜스포머 인코더에 입력되는 데이터의 모양을 확인하는 데 사용된다.

# 차원을 확장하기 위한 FC 레이어
transformer_input_expanded = attention.qkv(transformer_input)[0]  # 어텐션 블록의 qkv 메서드를 사용하여 주어진 입력 텐서를 쿼리, 키 및 값에 대한 선형 변환한다. 이후 [0]은 변환된 결과 중 쿼리에 대한 부분을 선택한다.
print("expanded to: ", transformer_input_expanded.shape)  # 변환된 결과의 모양을 출력한다. 이는 입력 텐서가 어떻게 변환되어 확장되는지를 확인하는 데 사용된다. 

# 다중 헤드 어텐션을 위해 qkv를 여러 개의 query, key 및 value 벡터로 분할합니다
qkv = transformer_input_expanded.reshape(197, 3, 12, 64)  # 변환된 입력 텐서를 197개의 샘플, 3개의 섹션(쿼리, 키, 값), 12개의 어텐션 헤드, 64개의 차원으로 변환한다. 이는 쿼리, 키 및 값으로 나뉘어진 텐서를 얻기 위함이다.
print("split qkv : ", qkv.shape)  # 쿼리, 키 및 값으로 나누어진 텐서의 모양을 출력한다.
q = qkv[:, 0].permute(1, 0, 2)  # 나누어진 텐서에서 쿼리 부분을 선택하고 차원을 재배치하여 어텐션 헤드, 샘플, 헤드 당 차원으로 조정한다.
k = qkv[:, 1].permute(1, 0, 2)  # 나누어진 텐서에서 키 부분을 선택하고 차원을 재배치하여 어텐션 헤드, 샘플, 헤드 당 차원으로 조정한다.
kT = k.permute(0, 2, 1)  # 키를 재배치하여 전치된 키를 생성한다. 이는 키를 사용하여 어텐션 스코어를 계산하는 데 사용된다.
print("transposed ks: ", kT.shape)  # 전치된 키의 모양을 출력한다. 이는 전치된 키가 어떻게 형성되는지를 확인하는 데 사용된다.

# 어텐션 행렬
attention_matrix = q @ kT  # 주어진 쿼리와 전치된 키를 곱하여 어텐션 행렬을 계산한다.
print("attention matrix: ", attention_matrix.shape)  # 어텐션 행렬의 모양을 출력한다. 이는 어텐션 행렬이 어떻게 형성되는지를 이해하는 데 도움이 된다.
plt.imshow(attention_matrix[3].detach().cpu().numpy())  # 어텐션 행렬 중 하나를 시각화한다. 여기서는 네 번째 어텐션 헤드의 행렬을 시각화한다. 텐서를 NumPy 배열로 변환한 후 imshow 함수를 사용하여 시각화한다.
plt.show()

# 어텐션 행렬 시각화
fig = plt.figure(figsize=(16, 8))  # 그림을 생성하고 크기를 설정한다.
fig.suptitle("Visualization of Attention", fontsize=24)  # 그림의 제목을 설정한다.
img = np.asarray(img)
ax = fig.add_subplot(2, 4, 1)  # 첫 번째 서브 플롯을 추가한다. 이 곳에는 원본 이미지가 표시된다.
ax.imshow(img)  # 원본 이미지를 시각화한다.
for i in range(7):  # 어텐션 행렬의 일부를 시각화하기 위한 반복문을 설정한다. 여기서는 7개의 어텐션 헤드를 시각화한다.
    attn_heatmap = attention_matrix[i, 100, 1:].reshape((14, 14)).detach().cpu().numpy()  # i번째 어텐션 헤드에서 100번째 행의 어텐션 행렬을 가져와서 시각화할 수 있는 형태로 변환한다.
    ax = fig.add_subplot(2, 4, i+2)  # 그림에 서브 플롯을 추가한다. 첫 번째 플롯은 이미지를 표시하므로, 두 번째 플롯부터 시작한다.
    ax.imshow(attn_heatmap)  # 어텐션 히트맵을 시각화한다.
plt.show()

##################################################################################################################################################################
# MLP (Classification) Head

print("Classification head: ", model.head)
result = model.head(transformer_output)  # 모델의 분류 헤드를 사용하여 트랜스포머 출력을 분류한다.
result_label_id = int(torch.argmax(result))  # 분류 결과에서 가장 높은 확률을 가진 클래스의 인덱스를 가져온다.
plt.plot(result.detach().cpu().numpy()[0])  # 분류 결과를 그래프로 시각화한다. 이 그래프는 각 클래스에 대한 예측 확률을 보여준다.
plt.title("Classification result")  # 그래프의 제목을 설정한다.
plt.xlabel("class id")  # x축 레이블을 설정한다.
print("Inference result : id = {}, label name = {}".format(
    result_label_id, imagenet_labels[result_label_id]))  # 추론 결과를 출력한다. 이는 예측된 클래스의 인덱스와 해당 클래스의 레이블 이름을 출력한다.
plt.show()
