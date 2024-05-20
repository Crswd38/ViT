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
# wget https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/watermelon.png?raw=true -O watermelon.png
#
##################################################################################################################################################################
# 준비하기

import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from timm import create_model

# 모델을 로드하는 함수
def load_model(device):
    model_name = "vit_base_patch16_224"  # 모델 이름 지정
    model = create_model(model_name, pretrained=True).to(device)  # 사전 훈련된 모델 로드
    return model

# 이미지 변환을 위한 함수
def prepare_transforms():
    IMG_SIZE = (224, 224)  # 이미지 크기 설정
    NORMALIZE_MEAN = (0.5, 0.5, 0.5)  # 정규화를 위한 평균 값
    NORMALIZE_STD = (0.5, 0.5, 0.5)  # 정규화를 위한 표준 편차
    transform_steps = [
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ]
    return T.Compose(transform_steps)

# 이미지를 처리하는 함수
def process_image(image_path, transforms, device):
    img = PIL.Image.open(image_path)  # 이미지 파일 열기
    img_tensor = transforms(img).unsqueeze(0).to(device)  # 변환 적용 및 텐서로 변환
    return img_tensor

# 이미지 추론을 수행하는 함수
def infer_image(model, img_tensor):
    patches = model.patch_embed(img_tensor)  # 이미지를 패치로 분할
    pos_embed = model.pos_embed  # 위치 임베딩 가져오기
    transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed  # 입력 데이터 준비
    x = transformer_input.clone()
    for blk in model.blocks:
        x = blk(x)  # 트랜스포머 블록을 통과
    x = model.norm(x)  # 정규화 적용
    transformer_output = x[:, 0]  # 첫 번째 출력 가져오기
    return transformer_output

# 추론 결과를 출력하는 함수
def print_inference_result(transformer_output, model):
    result = model.head(transformer_output)  # 최종 출력 계산
    result_label_id = int(torch.argmax(result))  # 결과에서 가장 높은 값을 가진 인덱스 추출
    imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))  # ImageNet 레이블 로드
    print(f"Inference result: id = {result_label_id}, label name = {imagenet_labels[result_label_id]}")  # 결과 출력

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU 사용 가능 여부에 따라 디바이스 설정
model = load_model(device)  # 모델 로드
transforms = prepare_transforms()  # 변환 설정
img_tensor = process_image('santorini.png', transforms, device)  # 이미지 처리
transformer_output = infer_image(model, img_tensor)  # 이미지 추론
print_inference_result(transformer_output, model)  # 결과 출력

