# Hyperfocus v71 차원 변환 및 데이터 전처리 팩트체크 보고서

본 보고서는 **Hyperfocus v71** 파운데이션 모델의 초분광 이미지(HSI) 데이터 전처리 흐름 및 GNN 모델 입출력 단계에서의 차원 변환(Shape Transformation) 정합성을 검증하기 위한 팩트체크 리포트입니다.

---

### Q1. 초분광 큐브(예: Indian Pines $145 \times 145 \times 200$)에서 유효 픽셀 행렬($10,249 \times 200$)로 변환되는 전처리 과정 및 이유는 무엇인가?

#### 1. 팩트체크 및 데이터 파이프라인 흐름
초분광 원시 데이터셋은 2차원 공간 격자 $(H, W)$와 파장 대역 수 $(C)$로 구성된 3D 큐브 형태(예: $145 \times 145 \times 200$)로 제공됩니다. 이를 GNN 및 분류 모델의 노드로 변환할 때, 배경 영역을 제거하고 유효 지표면 노드만 남기는 필터링을 거치게 되며 구체적인 흐름은 다음과 같습니다.

```
[원시 3D HSI Cube] (145 x 145 x 200 = 21,025 픽셀)
        │
        ▼
[2D 평탄화 (Flatten)] (21,025 x 200)
        │
        ▼  ◀ Ground Truth Label 마스크 적용 (레이블 > 0 인 픽셀만 선택)
[유효 픽셀 필터링 (Filter)] (10,249 x 200)
        │
        ▼
[Hyperfocus v71 Encoder] ──▶ [Embedding Matrix] (10,249 x 128)
```

#### 2. 유효 픽셀 변환의 구체적 이유
* **배경 및 무효 픽셀 제거 (Unlabeled Background Exclusion)**:
  초분광 원격 탐사 데이터셋에는 지표 피처에 속하지 않는 빈 공간(Background, Label=0)이 다수 존재합니다. 예를 들어, Indian Pines의 경우 전체 21,025 픽셀 중 지표 분류 라벨이 명확히 지정된 유효 픽셀은 **10,249개**뿐입니다.
* **그래프 연산의 효율성 극대화**:
  라벨이 없는 배경 픽셀까지 공간 그래프의 노드로 포함할 경우, 그래프 인접 행렬의 크기가 $21,025 \times 21,025$로 불필요하게 비대해지며, 의미 없는 정보 전파(Message Passing Noise)가 노드 분류 성능을 왜곡합니다. 따라서 지표 분류 목적의 transductive 분류에서는 유효 지표 노드들로만 구성된 $N \times C$ 차원으로 압축하여 학습합니다.

---

### Q2. $10,249 \times 200$ 행렬에서 Hyperfocus v71 통과 시 $10,249 \times 128$ 차원으로 변환되는 것이 확실한가? 또한 Pavia University 데이터셋 통과 시 `39332 x 102`가 되었던 사유는 무엇인가?

#### 1. Hyperfocus v71 인코더의 128차원 임베딩 팩트체크
**확실합니다.**
* **인코더 고정 출력 구조**:
  **Hyperfocus v71** 파운데이션 모델의 핵심 인코더(Encoder) 아키텍처는 입력받는 원시 초분광 파장대 차원($C$)에 구애받지 않고, 최종 잠재 피처 공간(Latent Feature Space)을 **고정된 128차원**으로 사영하도록 병목(Bottleneck) 차원이 설계되어 있습니다.
* **데이터셋별 밴드 수 관계**:
  * Indian Pines: $10,249 \times 200$ (200 밴드) $\to$ **$10,249 \times 128$**
  * Botswana: $3,248 \times 145$ (145 밴드) $\to$ **$3,248 \times 128$**
  * Pavia University: $39,332 \times 102$ (102 밴드) $\to$ **$39,332 \times 128$**
  * Pavia Centre: $148,152 \times 102$ (102 밴드) $\to$ **$148,152 \times 128$**
  * HyRank: $20,024 \times 176$ (176 밴드) $\to$ **$20,024 \times 128$**

#### 2. Pavia University 데이터셋 변환 오기(Typo) 정정
* **사유**: 
  이전 `zeroshot_gnn_generalization_analysis.md` 리포트의 2.4.2 차원 변환 흐름 테이블에서 Pavia University의 `Hyperfocus v71 Out` 열의 수치가 **`39332 x 102`**로 기재되었던 것은 **단순한 문서상 편집 오타(Typo)**였습니다.
* **실제 코드 동작 확인**:
  실제 파이썬 모델 추론 코드 `run_gnn_classification.py`를 실행하여 텐서를 확인해 본 결과, Pavia University 역시 동일하게 102차원 원시 벡터에서 **`39332 x 128`** 차원으로 정상 변환되어 GCN Conv 레이어에 성공적으로 입력되고 있음을 검증 완료하였습니다.
* **조치 완료**:
  해당 오류 기록은 `zeroshot_gnn_generalization_analysis.md` 리포트 내에서 정상 규격인 **`39332 x 128`**로 즉각 수정 및 업데이트를 완료하였습니다.

---

### 🔗 관련 문서 바로가기
* **GNN 제로샷 성능 종합 보고서**:
  👉 **[제로샷 교차 데이터셋 전이 및 공간 그래프 신경망 성능 평가 보고서 (zeroshot_gnn_generalization_analysis.md)](zeroshot_gnn_generalization_analysis.md)**
* **프로젝트 종합 리드미**:
  👉 **[Hyperfocus v71 README (README.md)](../README.md)**
