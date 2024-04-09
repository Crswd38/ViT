# ViT (Vision in Transformer)

### ViT의 목적 : Image Classification

			 input으로 새로운 이미지가 들어오면 무슨 이미지인지 구별하는 것
    
### ViT의 네트워크 구조 : 들어온 이미지를 16*16크기로 나누고 순서대로 패치들을 임베딩해서 트랜스포머 인코더에 넣는다.

					1. 들어온 이미지를 16*16크기로 나눈다
     
					2. 가장 앞부분에 포지션 임베딩을 넣어 위치값을 보존한다. 왼쪽상단은 왼쪽위가 가장 활성화되어있고 중앙은 중앙이 가장 활성화되어있다. 이런 포지션 임베딩을 각각에 맞는 패치에 더해 위치값을 보존한다.
     
					![image1](https://github.com/Crswd38/ViT/blob/main/readmeImage.png)
     
					3. 임베딩된 패치들이 input으로 들어가면 각각을 Normalization 취한다.
     
					4. 그 후 모두 Concatenation 시켜 합친다.
     
					5. Self Attention 구조에 들어가기 위해 해당 Vector에 Weight를 곱하여 쿼리, 키, 밸류로 나누고 Self Attention을 반복하기 위해 Multi Head Attention구조로 반복한다.
     
					6. Skip Connection 기법으로 Multi Head Sttention을 통과한 값과 통과하지 못한 값을 더해 기존의 값을 보존한다.
     
					7. 다시 Normalization을 취해주고 Multi Layer Perception을 통과한 값과 기존의 값을 Skip Connection으로 다시 더해준다.
     
					8. Transformer Encoder의 아웃풋은 MLP를 통해 어떤 이미지인지 Classification 해줄 수 있다.
     
					![image2](https://github.com/Crswd38/ViT/blob/main/readmeImage.png)
     
