#################################################################################################################################################################
#
# 터미널에 한번씩 치고 시작
# pip install timm
# pip install torch
# pip install torchvision
# pip install wget
# wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt -O ilsvrc2012_wordnet_lemmas.txt
# pip install PyQt5
#
##################################################################################################################################################################

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from timm import create_model

result = "1"

class CWidget(QWidget):

    def __init__(self):
        super().__init__()
 
        # 전체 폼 박스
        formbox = QHBoxLayout()
        self.setLayout(formbox)
 
        # 좌, 우 레이아웃박스
        left = QVBoxLayout()
        right = QVBoxLayout()
 
        # 그룹박스1 생성 및 좌 레이아웃 배치
        gb = QGroupBox('그리기 종류')
        left.addWidget(gb)
 
        # 그룹박스1 에서 사용할 레이아웃
        box = QVBoxLayout()
        gb.setLayout(box)
 
        # 그룹박스 1 의 라디오 버튼 배치
        text = ['직선', '곡선', '사각형', '원']
        self.radiobtns = []
 
        for i in range(len(text)):
            self.radiobtns.append(QRadioButton(text[i], self))
            self.radiobtns[i].clicked.connect(self.radioClicked)
            box.addWidget(self.radiobtns[i])
 
        self.radiobtns[1].setChecked(True)
        self.drawType = 1
         
        # 그룹박스2 펜 설정
        gb = QGroupBox('펜 설정')
        left.addWidget(gb)
 
        grid = QGridLayout()
        gb.setLayout(grid)
 
        label = QLabel('선굵기')
        grid.addWidget(label, 0, 0)
 
        self.combo = QComboBox()
        grid.addWidget(self.combo, 0, 1)
 
        for i in range(1, 21):
            self.combo.addItem(str(i))
 
        label = QLabel('선색상')
        grid.addWidget(label, 1,0)
         
        self.pencolor = QColor(0,0,0)
        self.penbtn = QPushButton()
        self.penbtn.setStyleSheet('background-color: rgb(0,0,0)')
        self.penbtn.clicked.connect(self.showColorDlg)
        grid.addWidget(self.penbtn,1, 1)
         
        # 그룹박스3 붓 설정
        gb = QGroupBox('붓 설정')
        left.addWidget(gb)
 
        hbox = QHBoxLayout()
        gb.setLayout(hbox)
 
        label = QLabel('붓색상')
        hbox.addWidget(label)
 
        self.brushcolor = QColor(255,255,255)
        self.brushbtn = QPushButton()
        self.brushbtn.setStyleSheet('background-color: rgb(255,255,255)')
        self.brushbtn.clicked.connect(self.showColorDlg)
        hbox.addWidget(self.brushbtn)
 
        # 그룹박스4 지우개
        gb = QGroupBox('지우개')
        left.addWidget(gb)
 
        grid = QGridLayout()
        gb.setLayout(grid)
         
        self.checkbox = QCheckBox('지우개')
        self.checkbox.stateChanged.connect(self.checkClicked)
        grid.addWidget(self.checkbox, 0, 0)

        self.clearbtn = QPushButton('전체 지우기')
        self.clearbtn.clicked.connect(self.clearall)
        grid.addWidget(self.clearbtn, 1, 0)
        
 
        # 그룹박스5: 그림 저장 + ViT 동작
        gb = QGroupBox('그리기 완료')
        left.addWidget(gb)

        hbox = QHBoxLayout()
        gb.setLayout(hbox)

        self.savebtn = QPushButton('저장하기')
        self.savebtn.clicked.connect(self.ViT)
        hbox.addWidget(self.savebtn)

        # 그룹박스6 분석 결과
        gb = QGroupBox('분석 결과')
        left.addWidget(gb)
 
        hbox = QHBoxLayout()
        gb.setLayout(hbox)
         
        label = QLabel(result)
        hbox.addWidget(label)
        print(result)
       
        left.addStretch(1)

        # 우 레이아웃 박스에 그래픽 뷰 추가
        self.view = CView(self)
        right.addWidget(self.view)
 
        # 전체 폼박스에 좌우 박스 배치
        formbox.addLayout(left)
        formbox.addLayout(right)
 
        formbox.setStretchFactor(left, 0)
        formbox.setStretchFactor(right, 1)
        
        self.setGeometry(100, 100, 800, 500)
        
    def radioClicked(self):
        for i in range(len(self.radiobtns)):
            if self.radiobtns[i].isChecked():
                self.drawType = i
                break

    def checkClicked(self):
        pass

    def clearall(self):
        self.view.scene.clear()
    
    def showColorDlg(self):
        
        # 색상 대화상자 생성
        color = QColorDialog.getColor()
 
        sender = self.sender()
 
        # 색상이 유효한 값이면 참, QFrame에 색 적용
        if sender == self.penbtn and color.isValid():
            self.pencolor = color
            self.penbtn.setStyleSheet('background-color: {}'.format( color.name()))
        else:
            self.brushcolor = color
            self.brushbtn.setStyleSheet('background-color: {}'.format( color.name()))
    
    # 모델을 로드하는 함수
    def load_model(self, device):
        model_name = "vit_base_patch16_224"  # 모델 이름 지정
        model = create_model(model_name, pretrained=True).to(device)  # 사전 훈련된 모델 로드
        return model

    # 이미지 변환을 위한 함수
    def prepare_transforms(self):
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
    def process_image(self, image_path, transforms, device):
        img = PIL.Image.open(image_path)  # 이미지 파일 열기
        img_tensor = transforms(img).unsqueeze(0).to(device)  # 변환 적용 및 텐서로 변환
        return img_tensor

    # 이미지 추론을 수행하는 함수
    def infer_image(self, model, img_tensor):
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
    def print_inference_result(self, transformer_output, model):
        res = model.head(transformer_output)  # 최종 출력 계산
        result_label_id = int(torch.argmax(res))  # 결과에서 가장 높은 값을 가진 인덱스 추출
        imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))  # ImageNet 레이블 로드
        result = imagenet_labels[result_label_id]
        print(f"Inference result: id = {result_label_id}, label name = {imagenet_labels[result_label_id]}")  # 결과 출력
        self.result = result
        self.view.update()

    def ViT(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "그림 저장", "", "PNG Files (*.png);;JPEG Files (*.jpeg)")
        if filePath:
            pixmap = QPixmap(self.view.viewport().size())
            self.view.viewport().render(pixmap)
            pixmap.save(filePath)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU 사용 가능 여부에 따라 디바이스 설정
            model = self.load_model(device)  # 모델 로드
            transforms = self.prepare_transforms()  # 변환 설정
            img_tensor = self.process_image(filePath, transforms, device)  # 이미지 처리
            transformer_output = self.infer_image(model, img_tensor)  # 이미지 추론
            self.print_inference_result(transformer_output, model)  # 결과 출력

# QGraphicsView display QGraphicsScene
class CView(QGraphicsView):
    
    def __init__(self, parent):
        super().__init__(parent)       
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.items = []
        self.start = QPointF()
        self.end = QPointF()
        self.setRenderHint(QPainter.HighQualityAntialiasing)
 
    def moveEvent(self, e):
        rect = QRectF(self.rect())
        rect.adjust(0,0,-2,-2)
 
        self.scene.setSceneRect(rect)
 
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            # 시작점 저장
            self.start = e.pos()
            self.end = e.pos()
 
    def mouseMoveEvent(self, e):
        # e.buttons()는 정수형 값을 리턴, e.button()은 move시 Qt.Nobutton 리턴
        if e.buttons() & Qt.LeftButton:
            self.end = e.pos()
            if self.parent().checkbox.isChecked():
                pen = QPen(QColor(255,255,255), 10)
                path = QPainterPath()
                path.moveTo(self.start)
                path.lineTo(self.end)
                self.scene.addPath(path, pen)
                self.start = e.pos()
                return None
 
            pen = QPen(self.parent().pencolor, self.parent().combo.currentIndex())
 
            # 직선 그리기
            if self.parent().drawType == 0:
                 
                # 장면에 그려진 이전 선을 제거
                if len(self.items) > 0:
                    self.scene.removeItem(self.items[-1])
                    del(self.items[-1])
 
                # 현재 선 추가
                line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())
                self.items.append(self.scene.addLine(line, pen))
 
            # 곡선 그리기
            elif self.parent().drawType == 1:
 
                # Path 이용
                path = QPainterPath()
                path.moveTo(self.start)
                path.lineTo(self.end)
                self.scene.addPath(path, pen)
 
                # Line 이용
                #line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())
                #self.scene.addLine(line, pen)
                 
                # 시작점을 다시 기존 끝점으로
                self.start = e.pos()
 
            # 사각형 그리기
            elif self.parent().drawType == 2:
                brush = QBrush(self.parent().brushcolor)
 
                if len(self.items) > 0:
                    self.scene.removeItem(self.items[-1])
                    del(self.items[-1])
 
                rect = QRectF(self.start, self.end)
                self.items.append(self.scene.addRect(rect, pen, brush))
                 
            # 원 그리기
            elif self.parent().drawType == 3:
                brush = QBrush(self.parent().brushcolor)
 
                if len(self.items) > 0:
                    self.scene.removeItem(self.items[-1])
                    del(self.items[-1])
 
                rect = QRectF(self.start, self.end)
                self.items.append(self.scene.addEllipse(rect, pen, brush))
 
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self.parent().checkbox.isChecked():
                return None
            
            pen = QPen(self.parent().pencolor, self.parent().combo.currentIndex())

            if self.parent().drawType == 0:
                self.items.clear()
                line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())
                self.scene.addLine(line, pen)

            elif self.parent().drawType == 2:
                brush = QBrush(self.parent().brushcolor)
                self.items.clear()
                rect = QRectF(self.start, self.end)
                self.scene.addRect(rect, pen, brush)

            elif self.parent().drawType == 3:
                brush = QBrush(self.parent().brushcolor)
                self.items.clear()
                rect = QRectF(self.start, self.end)
                self.scene.addEllipse(rect, pen, brush)
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CWidget()
    w.show()
    sys.exit(app.exec_())
