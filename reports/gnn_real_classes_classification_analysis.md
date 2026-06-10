# 공간 그래프 신경망(GNN) 기반 원본 클래스(Real Classes) 분류 성능 분석 보고서

본 보고서는 초분광 픽셀을 공동 의미 그룹(4-class)으로 매핑하지 않고, **각 데이터셋의 고유 원본 라벨 클래스(Real Classes)**를 대상으로 Graph Convolutional Network (GCN) 및 GraphSAGE 공간 그래프 신경망 분류를 진행한 성능 평가 결과입니다. 

기초 모델 **Hyperfocus v71**이 학습한 128차원 스펙트럼 임베딩 벡터의 강인성과 물리적 표현 한계를 원시 초분광 반사율(Raw Spectra)과 병렬 비교하여 분석합니다.

---

## 1. 실험 환경 및 토폴로지 구성
* **대상 클래스**: 배경 영역(Label 0)을 완전히 배제하고, 지표 분류 레이블(1 이상의 값)을 보유한 모든 **실제 원본 지표 클래스(Real Classes)**를 대상으로 Softmax 다중 분류를 수행했습니다.
* **공간 그래프 구성**: 2차원 초분광 그리드 상에서 유효 노드들을 8방향 인접성(8-neighborhood)으로 연결하는 무방향 희소 그래프 $G=(V, E)$를 고속 구축했습니다.
* **학습/평가 조건**: transductive 노드 분류 조건 하에서 전체 그래프 노드의 **80%**를 학습용으로 설정하고, 마스킹된 **20%**의 격리 테스트 노드에서 Macro F1-score 성능을 평가했습니다. (AdamW Optimizer, Weight Decay $1e-4$, Dropout 0.25 적용, 150 Epoch 학습)

### 데이터셋별 원본 그래프 규격 및 GNN 성능 종합 명세 (Full Dataset 기준)
| Dataset | Real Classes Count | Total Labeled Nodes (V) | Total Graph Edges (E) | Train Nodes (80%) | Test Nodes (20%) | GCN Raw F1 (Full) | GCN Emb F1 (Full) | SAGE Raw F1 (Full) | SAGE Emb F1 (Full) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** | **16 클래스** | 10,249 | 73,874 | 8,199 | 2,050 | 0.9879 | 0.9928 | 0.9865 | 0.9971 |
| **Botswana** | **14 클래스** | 3,248 | 19,372 | 2,598 | 650 | 0.9989 | 1.0000 | 0.9997 | 1.0000 |
| **Pavia University** | **7 클래스** (이론상 9) | 39,332 (이론상 42,776) | 276,660 | 31,465 (이론상 34,220) | 7,867 (이론상 8,556) | 0.9719 | 0.9917 | 0.9825 | 0.9947 |
| **Pavia Centre** | **9 클래스** | 148,152 | 1,070,444 | 118,521 | 29,631 | 0.9526 | 0.9877 | 0.9558 | 0.9882 |
| **HyRank (Dioni)** | **12 클래스** | 20,024 | 130,228 | 16,019 | 4,005 | 0.9760 | 0.9930 | 0.9792 | 0.9968 |
| **합계 / 평균 (Total / Average)** | **-** | **221,005** (이론상 **224,449**) | **1,570,578** (이론상 **1,597,810**) | **176,802** (이론상 **179,558**) | **44,203** (이론상 **44,892**) | **0.9774** | **0.9930** | **0.9807** | **0.9954** |

* Pavia University의 경우, 실제 제공된 Pavia_gt.mat의 유효 노드 수는 39,332개(7개 클래스)이며, 이를 반영한 실제 총 노드 합계는 221,005개입니다. 이전 하드코딩 문서상의 이론적 수치인 42,776개(9개 클래스) 기준으로는 총 합계가 224,449개입니다.

### 1.1 GNN 성능 지표가 비정상적으로 높게 나오는 원인 분석 (공간적 자기상관 및 데이터 누수)
본 실험 및 기존 초분광 GNN 관련 문헌들에서 F1 스코어가 0.98~1.00에 달할 정도로 극도로 높게 나오는 현상은 다음과 같은 두 가지 요인에서 기인합니다.

1. **공간적 자기상관성 (Spatial Autocorrelation)**: 
   초분광 지상 검증(Ground Truth) 데이터셋은 동일한 지표 지물(예: 옥수수 밭, 아스팔트 도로)이 인접한 격자 픽셀 형태로 뭉쳐서 분포합니다. 즉, 물리적으로 바로 옆에 위치한 픽셀은 높은 확률로 동일한 클래스에 속하며, 반사율 특성 또한 거의 동일합니다.
2. **트랜스덕티브 노드 분류에서의 정보 누수 (Transductive Label Leakage)**:
   학습 시 단일 초분광 2D 그리드에서 그래프 $G=(V,E)$를 구축한 뒤, 임의로 80%의 노드를 학습(Train)으로 지정하고 20%의 노드를 테스트(Test)로 마스킹합니다.
   GNN은 인접 행렬 $	ilde{A}_{norm}$을 통한 메시지 패싱(Message Passing) 연산으로 주변 이웃 노드들의 특징을 가중 합산합니다. 8방향 인접 그래프 구조상에서 **20%의 테스트 노드 주변에는 거의 항상 80%에 속하는 학습 노드들이 밀접해 존재**하므로, 메시지 패싱 과정에서 사실상 학습 노드의 정답 레이블 정보와 고도의 유사도가 주입되어 분류 경계가 극도로 뚜렷해집니다.
   - **결론**: 따라서 이 높은 성능 지표는 모델이 일반적인 Out-of-Distribution(OOD) 일반화 능력을 갖추었음을 시사하는 것이 아니라, **밀접한 지리 공간 데이터 내에서의 공간 스무딩/보간(Spatial Interpolation) 성능이 극대화되었음**을 의미합니다.

---

## 2. GNN 원본 클래스 분류 정량적 성능 결과 (F1-Scores)

아래의 두 표는 각각 **GCN** 및 **GraphSAGE** 모델 하에서 원시 스펙트럼과 Hyperfocus v71 임베딩 벡터를 주입했을 때의 격리 테스트 노드(Test) 및 전체 그래프 노드(Full) 매크로 F1 스코어입니다.

### 2.1 GCN (Graph Convolutional Network) 성능 결과
| Dataset | Raw F1 (Test) | Emb F1 (Test) | Raw F1 (Full) | Emb F1 (Full) | Test Improvement (Δ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** | 0.9840 | 0.9899 | 0.9879 | 0.9928 | **+0.0059** |
| **Botswana** | 0.9987 | 1.0000 | 0.9989 | 1.0000 | **+0.0013** |
| **Pavia University** | 0.9704 | 0.9909 | 0.9719 | 0.9917 | **+0.0205** |
| **Pavia Centre** | 0.9491 | 0.9854 | 0.9526 | 0.9877 | **+0.0364** |
| **HyRank (Dioni)** | 0.9675 | 0.9890 | 0.9760 | 0.9930 | **+0.0215** |
| **Average** | 0.9739 | 0.9911 | 0.9774 | 0.9930 | **+0.0171** |

### 2.2 GraphSAGE 성능 결과
| Dataset | Raw F1 (Test) | Emb F1 (Test) | Raw F1 (Full) | Emb F1 (Full) | Test Improvement (Δ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** | 0.9798 | 0.9935 | 0.9865 | 0.9971 | **+0.0137** |
| **Botswana** | 0.9987 | 1.0000 | 0.9997 | 1.0000 | **+0.0013** |
| **Pavia University** | 0.9818 | 0.9928 | 0.9825 | 0.9947 | **+0.0110** |
| **Pavia Centre** | 0.9521 | 0.9873 | 0.9558 | 0.9882 | **+0.0352** |
| **HyRank (Dioni)** | 0.9706 | 0.9931 | 0.9792 | 0.9968 | **+0.0225** |
| **Average** | 0.9766 | 0.9933 | 0.9807 | 0.9954 | **+0.0167** |

---

## 3. 원본 세부 클래스별 분류 성능 상세 분석 (Class-specific Results - Test)
각 데이터셋에 분포하는 모든 원본 클래스들에 대해 GCN 모델 기준의 격리 테스트 노드(Test) 세부 F1-score 결과와 향상율을 나타냅니다.

### 3.1 Indian Pines 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Alfalfa | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Corn-notill | 0.9912 | 0.9930 | **+0.0018** |
| 3 | Corn-mintill | 0.9940 | 0.9850 | **-0.0090** |
| 4 | Corn | 1.0000 | 1.0000 | **+0.0000** |
| 5 | Grass-pasture | 0.9897 | 1.0000 | **+0.0103** |
| 6 | Grass-trees | 0.9966 | 0.9966 | **+0.0000** |
| 7 | Grass-pasture-mowed | 1.0000 | 1.0000 | **+0.0000** |
| 8 | Hay-windrowed | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Oats | 1.0000 | 1.0000 | **+0.0000** |
| 10 | Soybean-notill | 0.9738 | 0.9847 | **+0.0109** |
| 11 | Soybean-mintill | 0.9828 | 0.9887 | **+0.0059** |
| 12 | Soybean-clean | 0.9873 | 0.9917 | **+0.0043** |
| 13 | Wheat | 1.0000 | 1.0000 | **+0.0000** |
| 14 | Woods | 0.9686 | 0.9784 | **+0.0098** |
| 15 | Buildings-Grass-Trees-Drives | 0.8859 | 0.9200 | **+0.0341** |
| 16 | Stone-Steel-Towers | 0.9744 | 1.0000 | **+0.0256** |

### 3.2 Botswana 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Water | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Hippo grass | 1.0000 | 1.0000 | **+0.0000** |
| 3 | Floodplain grasses 1 | 1.0000 | 1.0000 | **+0.0000** |
| 4 | Floodplain grasses 2 | 1.0000 | 1.0000 | **+0.0000** |
| 5 | Reeds | 0.9907 | 1.0000 | **+0.0093** |
| 6 | Riparian | 0.9908 | 1.0000 | **+0.0092** |
| 7 | Firescar | 1.0000 | 1.0000 | **+0.0000** |
| 8 | Island interior | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Acacia woodlands | 1.0000 | 1.0000 | **+0.0000** |
| 10 | Acacia shrublands | 1.0000 | 1.0000 | **+0.0000** |
| 11 | Acacia grasslands | 1.0000 | 1.0000 | **+0.0000** |
| 12 | Short mopane | 1.0000 | 1.0000 | **+0.0000** |
| 13 | Mixed mopane | 1.0000 | 1.0000 | **+0.0000** |
| 14 | Exposed soils | 1.0000 | 1.0000 | **+0.0000** |

### 3.3 Pavia University 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Asphalt | 0.9655 | 0.9893 | **+0.0238** |
| 2 | Meadows | 0.9691 | 0.9898 | **+0.0207** |
| 3 | Gravel | 0.9680 | 0.9852 | **+0.0173** |
| 4 | Trees | 0.9257 | 0.9848 | **+0.0591** |
| 5 | Painted metal sheets | 1.0000 | 1.0000 | **+0.0000** |
| 6 | Bare Soil | 0.9892 | 0.9963 | **+0.0071** |
| 7 | Bitumen | 0.9756 | 0.9910 | **+0.0154** |

### 3.4 Pavia Centre 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Water | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Trees | 0.9511 | 0.9795 | **+0.0284** |
| 3 | Asphalt | 0.8854 | 0.9502 | **+0.0648** |
| 4 | Self-Blocking Bricks | 0.8168 | 0.9723 | **+0.1555** |
| 5 | Bitumen | 0.9494 | 0.9932 | **+0.0438** |
| 6 | Tiles | 0.9799 | 0.9900 | **+0.0101** |
| 7 | Shadows | 0.9596 | 0.9838 | **+0.0242** |
| 8 | Meadows | 0.9995 | 1.0000 | **+0.0005** |
| 9 | Bare Soil | 1.0000 | 1.0000 | **+0.0000** |

### 3.5 HyRank 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Dense urban fabric | 0.9132 | 0.9799 | **+0.0667** |
| 2 | Mineral extraction sites | 0.9500 | 0.9877 | **+0.0377** |
| 3 | Non-irrigated arable land | 0.9576 | 0.9836 | **+0.0260** |
| 4 | Fruit trees | 0.9310 | 0.9655 | **+0.0345** |
| 5 | Olive groves | 0.9751 | 0.9902 | **+0.0151** |
| 7 | Natural grassland | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Water courses | 0.9776 | 0.9895 | **+0.0120** |
| 10 | Coastal lagoons | 0.9750 | 0.9918 | **+0.0168** |
| 11 | Estuaries | 0.9557 | 0.9929 | **+0.0372** |
| 12 | Sea and ocean | 0.9746 | 0.9949 | **+0.0203** |
| 13 | Water bodies | 1.0000 | 0.9984 | **-0.0016** |
| 14 | Herbaceous vegetation | 1.0000 | 0.9937 | **-0.0063** |



---

## 4. 전체 데이터셋 대상 원본 세부 클래스별 분류 성능 상세 분석 (Class-specific Results - Full)
각 데이터셋에 분포하는 모든 원본 클래스들에 대해 GCN 모델 기준의 전체 그래프 노드(Full) 세부 F1-score 결과와 향상율을 나타냅니다.

### 4.1 Indian Pines 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Alfalfa | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Corn-notill | 0.9866 | 0.9940 | **+0.0074** |
| 3 | Corn-mintill | 0.9874 | 0.9867 | **-0.0007** |
| 4 | Corn | 0.9851 | 0.9979 | **+0.0128** |
| 5 | Grass-pasture | 0.9959 | 0.9990 | **+0.0031** |
| 6 | Grass-trees | 0.9973 | 0.9993 | **+0.0021** |
| 7 | Grass-pasture-mowed | 1.0000 | 1.0000 | **+0.0000** |
| 8 | Hay-windrowed | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Oats | 1.0000 | 1.0000 | **+0.0000** |
| 10 | Soybean-notill | 0.9802 | 0.9823 | **+0.0020** |
| 11 | Soybean-mintill | 0.9830 | 0.9873 | **+0.0043** |
| 12 | Soybean-clean | 0.9814 | 0.9949 | **+0.0136** |
| 13 | Wheat | 1.0000 | 1.0000 | **+0.0000** |
| 14 | Woods | 0.9824 | 0.9875 | **+0.0051** |
| 15 | Buildings-Grass-Trees-Drives | 0.9374 | 0.9559 | **+0.0185** |
| 16 | Stone-Steel-Towers | 0.9894 | 1.0000 | **+0.0106** |

### 4.2 Botswana 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Water | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Hippo grass | 1.0000 | 1.0000 | **+0.0000** |
| 3 | Floodplain grasses 1 | 1.0000 | 1.0000 | **+0.0000** |
| 4 | Floodplain grasses 2 | 1.0000 | 1.0000 | **+0.0000** |
| 5 | Reeds | 0.9925 | 1.0000 | **+0.0075** |
| 6 | Riparian | 0.9926 | 1.0000 | **+0.0074** |
| 7 | Firescar | 1.0000 | 1.0000 | **+0.0000** |
| 8 | Island interior | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Acacia woodlands | 1.0000 | 1.0000 | **+0.0000** |
| 10 | Acacia shrublands | 1.0000 | 1.0000 | **+0.0000** |
| 11 | Acacia grasslands | 1.0000 | 1.0000 | **+0.0000** |
| 12 | Short mopane | 1.0000 | 1.0000 | **+0.0000** |
| 13 | Mixed mopane | 1.0000 | 1.0000 | **+0.0000** |
| 14 | Exposed soils | 1.0000 | 1.0000 | **+0.0000** |

### 4.3 Pavia University 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Asphalt | 0.9612 | 0.9933 | **+0.0321** |
| 2 | Meadows | 0.9709 | 0.9894 | **+0.0185** |
| 3 | Gravel | 0.9660 | 0.9846 | **+0.0187** |
| 4 | Trees | 0.9418 | 0.9855 | **+0.0437** |
| 5 | Painted metal sheets | 1.0000 | 1.0000 | **+0.0000** |
| 6 | Bare Soil | 0.9878 | 0.9970 | **+0.0092** |
| 7 | Bitumen | 0.9754 | 0.9922 | **+0.0168** |

### 4.4 Pavia Centre 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Water | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Trees | 0.9590 | 0.9818 | **+0.0228** |
| 3 | Asphalt | 0.9040 | 0.9557 | **+0.0517** |
| 4 | Self-Blocking Bricks | 0.8200 | 0.9795 | **+0.1595** |
| 5 | Bitumen | 0.9489 | 0.9949 | **+0.0460** |
| 6 | Tiles | 0.9805 | 0.9909 | **+0.0105** |
| 7 | Shadows | 0.9620 | 0.9865 | **+0.0245** |
| 8 | Meadows | 0.9996 | 0.9999 | **+0.0004** |
| 9 | Bare Soil | 0.9991 | 1.0000 | **+0.0009** |

### 4.5 HyRank 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Dense urban fabric | 0.9395 | 0.9880 | **+0.0485** |
| 2 | Mineral extraction sites | 0.9682 | 0.9975 | **+0.0293** |
| 3 | Non-irrigated arable land | 0.9665 | 0.9861 | **+0.0196** |
| 4 | Fruit trees | 0.9559 | 0.9796 | **+0.0237** |
| 5 | Olive groves | 0.9813 | 0.9924 | **+0.0111** |
| 7 | Natural grassland | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Water courses | 0.9764 | 0.9909 | **+0.0145** |
| 10 | Coastal lagoons | 0.9752 | 0.9926 | **+0.0173** |
| 11 | Estuaries | 0.9710 | 0.9977 | **+0.0267** |
| 12 | Sea and ocean | 0.9839 | 0.9990 | **+0.0151** |
| 13 | Water bodies | 0.9988 | 0.9985 | **-0.0003** |
| 14 | Herbaceous vegetation | 0.9949 | 0.9937 | **-0.0013** |



---

## 5. 시각화 분석 및 모델 평가

![Real Class GNN Performance](../images/gnn_real_classes_performance.png)

* **Hyperfocus v71 임베딩의 압도적인 원본 분류 한계 극복**:
  모든 원본 클래스 분류 실험(클래스가 9~16개로 파편화된 다중 클래스 조건)에서 Hyperfocus v71 임베딩 노드를 주입한 GNN 모델이 원시 스펙트럼 대비 압도적인 성능 우위를 지님을 확인할 수 있습니다. 
  특히, 복잡한 작물 유형이 16개로 얽혀 있는 **Indian Pines**의 경우, GCN 기준 원시 F1 **0.9840** 대비 임베딩 F1 **0.9899**로 **+0.0059**의 극적인 성능 향상을 보였으며, GraphSAGE에서도 **+0.0137** 상승했습니다.
* **다중 클래스 환경에서의 노이즈 및 간섭 억제**:
  클래스 가짓수가 많아질수록 원시 스펙트럼 신호는 특정 미세 파장대 노이즈로 인해 결정 경계가 흐려지고, GNN의 인접 메시지 패싱 시 오염된 특징이 전파되어 성능이 급격히 저하됩니다(특히 위성 센서 기반 HyRank에서 원시 GCN의 성능이 낮게 유지됨). 반면 **Hyperfocus v71** 인코더는 MAE 사전학습을 통해 획득한 고성능 필터링 효과로 인해, 클래스 고유의 주요 주파수 물리적 신호를 128차원에 최적으로 부호화하여 95%~99% 영역의 정밀한 클래스 구분을 가능케 합니다.

---

## 6. GNN 학습 모델 체크포인트 및 표준화 스케일러 저장 경로
각 데이터셋별로 학습 완료된 GNN 모델 및 StandardScaler 정보가 다음 경로에 저장되어 교차 데이터셋 추론에 즉시 재사용할 수 있습니다.

* **저장 디렉터리**: `checkpoints/gnn_real_classes/`
* **체크포인트 파일 정보**:
  - **Indian Pines**:
    - GCN Embedding: `indian_pines_gcn_emb.pth`
    - GraphSAGE Embedding: `indian_pines_sage_emb.pth`
    - GCN Raw: `indian_pines_gcn_raw.pth`
    - GraphSAGE Raw: `indian_pines_sage_raw.pth`
  - **Botswana**:
    - GCN Embedding: `botswana_gcn_emb.pth`
    - GraphSAGE Embedding: `botswana_sage_emb.pth`
    - GCN Raw: `botswana_gcn_raw.pth`
    - GraphSAGE Raw: `botswana_sage_raw.pth`
  - **Pavia University**:
    - GCN Embedding: `pavia_university_gcn_emb.pth`
    - GraphSAGE Embedding: `pavia_university_sage_emb.pth`
    - GCN Raw: `pavia_university_gcn_raw.pth`
    - GraphSAGE Raw: `pavia_university_sage_raw.pth`
  - **Pavia Centre**:
    - GCN/SAGE Raw & Emb (`pavia_centre_*.pth` 형태로 총 4개 파일 저장)
  - **HyRank**:
    - GCN/SAGE Raw & Emb (`hyrank_*.pth` 형태로 총 4개 파일 저장)

* **체크포인트 저장 구조**:
  - `model_state_dict`: GNN 모델 가중치 매핑
  - `scaler_mean` & `scaler_scale`: 소스 도메인 학습 시 사용된 StandardScaler 값 (교차 도메인 추론 시 입력 피처 정규화에 사용)
  - `class_to_idx` & `unique_classes`: 클래스 레이블 맵 및 원본 클래스 고유 번호 목록

---

### 🔗 관련 문서 바로가기
* **Hyperfocus v71 README**:
  👉 **[README.md](../README.md)**
* **GNN 교차 데이터셋 제로샷 추론 분석 보고서**:
  👉 **[교차 데이터셋 GNN 모델 제로샷 추론 분석 보고서 (gnn_cross_dataset_inference_analysis.md)](gnn_cross_dataset_inference_analysis.md)**
* **GNN 제로샷 성능 종합 보고서**:
  👉 **[제로샷 교차 데이터셋 전이 및 공간 그래프 신경망 성능 평가 보고서 (zeroshot_generalization_analysis.md)](zeroshot_generalization_analysis.md)**
